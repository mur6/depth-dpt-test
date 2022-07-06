import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl

# from torchinfo import summary
from finetune.models import DPTModule

config = {
    "base_scale": 0.0000305,
    "base_shift": 0.1378,
    "batch_size": 1,
    "image_size": (384, 384),
    "base_lr": 1e-6,
    "max_lr": 1e-5,
    "num_epochs": 70,
    "early_stopping_patience": 10,
    "num_workers": 0,
    "model_path": f"./weights/dpt_hybrid-midas-501f0c75.pt",
    "dataset_path": f"./train/",
    # "weights_save_path": f"/content/drive/MyDrive/ml/dpt/models/DPT/",
    # "logs_save_path": f"/content/drive/MyDrive/ml/dpt/models/DPT/",
    # "checkpoint_path": f"/content/drive/MyDrive/ml/dpt/models/DPT/lightning_logs/version_1/checkpoints/epoch=57-step=9976.ckpt",
    "weights_save_path": f"models/DPT/",
    "logs_save_path": f"models/DPT/",
    "checkpoint_path": f"models/DPT/lightning_logs/version_1/checkpoints/epoch=57-step=9976.ckpt",
}

pl.seed_everything(42)

model = DPTModule(
    model_path=config["model_path"],
    dataset_path=config["dataset_path"],
    scale=config["base_scale"],
    shift=config["base_shift"],
    batch_size=config["batch_size"],
    base_lr=config["base_lr"],
    max_lr=config["max_lr"],
    num_workers=config["num_workers"],
    image_size=config["image_size"],
)
logger = pl.loggers.TensorBoardLogger(
    save_dir=config["logs_save_path"],
)

lr_monitor = pl.callbacks.LearningRateMonitor()
early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=config["early_stopping_patience"]
)

trainer = pl.Trainer(
    devices="auto",
    accelerator="auto",
    max_epochs=config["num_epochs"],
    logger=logger,
    callbacks=[lr_monitor, early_stopping],
    weights_save_path=config["weights_save_path"],
)

trainer.fit(model)
