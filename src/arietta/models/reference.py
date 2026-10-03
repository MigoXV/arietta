"""源自 convaiinnovations/laya，Apache-2.0；见 THIRD_PARTY.md。

RL Agent shared code: config, Jev-style question rendering, model, proper-scoring rewards, metrics.

Kept Python 3.9 compatible so the same file runs on Kaggle and on a laptop smoke test.
"""

import json
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q: Dict) -> List[str]:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state,
    q: Dict,
    max_len: int,
    head_max_len: int,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
):
    """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    Returns input_ids and the positions of the per-option [MASK] markers (in the given option order).
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)[
        "input_ids"
    ]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)[
                "input_ids"
            ][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)[
        "input_ids"
    ]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


# ----------------------------------------------------------------------------- model
class DecisionModel(nn.Module):
    """Pretrained bidirectional encoder (no LLM, no LoRA) + from-scratch decision head.

    Each option gets a [MASK] marker; the head scores markers -> softmax over the question's options.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act)
        )
        self.register_buffer(
            "temperature", torch.ones(3)
        )  # per qtype, fitted post-hoc in evaluate.py
        self.head_checkpointing = False

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        detach_encoder: bool = False,
    ):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if (
                    self.head_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    h = torch.utils.checkpoint.checkpoint(
                        layer, h, None, pad, use_reentrant=False
                    )
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # act head sees the pooled sequence + detached summary of its own answer distribution
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_input = torch.cat([pooled, feats], -1).to(self.act_head[0].weight.dtype)
        act_logits = self.act_head(act_input)
        return logits, act_logits


def build_model(cfg: Dict, encoder_dir: Optional[str] = None) -> DecisionModel:
    from transformers import AutoConfig, AutoModel

    if (
        encoder_dir
    ):  # offline: architecture only, weights come from the saved state dict
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)


# ----------------------------------------------------------------------------- rewards (strictly proper)
def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    ep_group = torch.full((n,), -1, dtype=torch.long)
    group_of = {}
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    # episodes: all prefixes of the same record share a group id (used for TD(lambda) targets)
    for i, it in enumerate(items):
        if it["episode"]:
            ep_group[i] = group_of.setdefault(it.get("rec_uid", -1 - i), len(group_of))
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
        "episode": torch.tensor([it["episode"] for it in items], dtype=torch.bool),
        "ep_group": ep_group,
        "ep_step": torch.tensor([it["ep_step"] for it in items]),
        "meta": [
            {k: it[k] for k in it if k not in ("ids", "markers", "target")}
            for it in items
        ],
        "n_tokens": int(att.sum()),
    }


def temp_bucket(qtype: int, k: int) -> str:
    """Key for per-cardinality temperature fitting: a 2-option noul and a 20-option choice need different scaling."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)
