import torch
import torch.nn as nn
from pathlib import Path
import pytorch_lightning as pl
from typing import Union
import os
from datetime import datetime
from models.mobilenetv3 import MobileNetV3Classifier
from utils.data import get_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_model(
        *args,
        **kwargs
) -> Union[torch.nn.Module, pl.LightningModule]:
    model_kwargs = {
        'num_classes': kwargs.get('num_classes', 2),
        'pretrained': kwargs.get('pretrained', True),
        'learning_rate': kwargs['learning_rate']
    }
    return MelanomaModel(**model_kwargs)


class MelanomaModel(MobileNetV3Classifier, pl.LightningModule):

    def __init__(self, num_classes, learning_rate, pretrained=True, **kwargs):
        super().__init__()
        
        self.save_hyperparameters('num_classes', 'learning_rate', 'pretrained')

        self.lr = learning_rate
        self.base_model = MobileNetV3Classifier(num_classes=num_classes, pretrained=pretrained)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        labels_out = self.forward(x)

        loss = self.loss_func(labels_out, labels).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch % 3 == 0:
            if batch_idx <= 20 and batch_idx % 2 == 0:
                pass
                # TODO: add logging
        del x, labels

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        labels_out = self.forward(x)

        loss = self.loss_func(labels_out, labels).mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        preds = torch.argmax(labels_out, dim=1)
        labels_argmax = torch.argmax(labels, dim=1)
        acc = (preds == labels_argmax).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_model(config: dict, data_path: Path):
    model_dir = Path(config["store_model_to"])
    # Append current timestamp to model directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = model_dir / current_time
    model_dir.mkdir(parents=True, exist_ok=True)
    print('Model directory: ', model_dir)

    checkpoint_dir = model_dir / "checkpoint"
    # custom parameters set in yaml file
    training_params = config["training_configuration"]

    base_img_dir = Path(data_path)
    csv_path = base_img_dir / "metadata_split.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at expected location: {csv_path}")
    if not base_img_dir.is_dir():
         raise FileNotFoundError(f"Base image directory not found: {base_img_dir}")

    # callback and logger
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='weights.epoch{epoch:03}-val_loss_{val_loss:.4f}',
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            auto_insert_metric_name=False,
            save_weights_only=False, 
            every_n_epochs=training_params.get("save_every_n_epochs", 1),
        )
    ]

    loggers = [
        TensorBoardLogger(
            save_dir=model_dir,
            name='board',
            version=''
        ),
    ]
    # model
    model = get_model(**training_params)

    # data
    train_dataloader = get_dataloader(
        csv_path=csv_path,
        base_img_dir=base_img_dir,
        split="train",
        **training_params
    )
    val_dataloader = get_dataloader(
        csv_path=csv_path,
        base_img_dir=base_img_dir,
        split="val",
        **training_params
    )

    # training
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=training_params["max_epochs"],
        devices=training_params.get("gpus", 1),
        accelerator=training_params.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"),
        strategy=training_params.get("strategy", 'auto'),
        gradient_clip_val=training_params.get("gradient_clip_val", 0.5),
        log_every_n_steps=training_params.get("log_every_n_steps", 50),
        precision=training_params.get("precision", 32)
    )

    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Training finished.")
