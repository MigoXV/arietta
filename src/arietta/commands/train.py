from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.cli import LightningCLI
from arietta.configs.run_config import DriftSafeSaveConfigCallback
from arietta.models import register_models


def main():
    register_models()
    LightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        trainer_class=Trainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=DriftSafeSaveConfigCallback,
        save_config_kwargs={"config_filename": "resolved.yaml"},
    )


if __name__ == "__main__":
    main()
