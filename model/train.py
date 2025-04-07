import torch
from pathlib import Path
import pytorch_lightning as pl
from typing import Union

from utils.data import get_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_model(
        *args,
        **kwargs
) -> Union[torch.nn.Module, pl.LightningModule]:
    return MelanomaModel(**kwargs)


class MelanomaModel(pl.LightningModule): # TODO: add model

    def __init__(self, *args, **kwargs):
        super(MelanomaModel, self).__init__(
                                           **kwargs
                                           )

        self.lr = kwargs["learning_rate"]
        self.loss_func = None

    def training_step(self, batch, batch_idx):
        x, labels = batch
        labels_out = self.forward(x)

        loss = self.loss_func(labels_out, labels).mean()
        # TODO: add logging

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
        # TODO: add logging

        if self.current_epoch % 3 == 0:
            if batch_idx <= 20 and batch_idx % 2 == 0:
                pass
                # TODO: add logging
        del x, labels_out
        del loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_model(config: dict, data_path: Path):
    model_dir = config["store_model_to"]
    print('Model directory: ', model_dir)

    # custom parameters set in yaml file
    training_params = config["training_configuration"]

    # callback and logger
    callbacks = [
        ModelCheckpoint(  # saves weights for every n epochs
            dirpath=Path(model_dir, "checkpoint"),
            filename='weights.epoch{epoch:03}-val_loss_{val_loss:.4f}',
            save_top_k=-1,
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_epochs=5,
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
        data_path,
        mlset="training",
        **training_params
    )
    val_dataloader = get_dataloader(
        data_path,
        mlset="validation",
        **training_params
    )

    # training
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=training_params["max_epochs"],
        gpus=training_params.get("gpus", 1),
        auto_select_gpus=True,
        strategy=training_params.get("strategy", None),
        gradient_clip_val=0.5,
        log_every_n_steps=5
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
