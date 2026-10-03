"""Dataset rendering policy; native runtime string semantics remain unchanged."""

from __future__ import annotations

import json
from typing import Literal

StateSerialization = Literal["verbatim", "online_json"]


def dataset_state(state, mode: StateSerialization = "verbatim"):
    if mode == "verbatim":
        return state
    if mode != "online_json":
        raise ValueError("state_serialization must be verbatim or online_json")
    try:
        value = json.loads(state) if isinstance(state, str) else state
    except json.JSONDecodeError as exc:
        raise ValueError(
            "online_json requires state to contain a JSON object or array"
        ) from exc
    if not isinstance(value, (dict, list)):
        raise ValueError("online_json requires state to contain a JSON object or array")
    # Returning the object uses the same default JSON rendering as native HTTP.
    # Preserve dictionary insertion order; never reorder the question options.
    return value
