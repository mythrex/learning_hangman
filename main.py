import yaml
import wandb
import pytorch_lightning as pl
from dotenv import load_dotenv
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pl_model import BertClassifier
from dataset import CustomDataset

pl.seed_everything(42, workers=True)
load_dotenv()
params = yaml.safe_load(open("params.yaml", "r"))["trainer"]
args_list = []

for k, v in params.items():
    args_list.append(f"--{k}={v}")

if __name__ == "__main__":

    wandb.login()
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default=None)

    parser = BertClassifier.add_model_specific_args(parser)
    args = parser.parse_args(args_list)

    # Create train, validation, and test datasets
    train_dataset = CustomDataset(args.filename, split="train", max_len=args.max_len)
    val_dataset = CustomDataset(args.filename, split="val", max_len=args.max_len)
    test_dataset = CustomDataset(args.filename, split="test", max_len=args.max_len)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, num_workers=8
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.val_batch_size, num_workers=8
    )
    # Create a PyTorch Lightning Trainer
    bert_classifier = BertClassifier(args)
    wandb_logger = WandbLogger(project=args.wandb_project_name, log_model=True)
    wandb_logger.watch(bert_classifier)

    checkpoint_callback = ModelCheckpoint(
        dirpath="model_ckpt", filename="best_model", monitor="val/hit@6", mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=1,
        accelerator=args.accelerator,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True
    )
    # Train the model using the Trainer
    trainer.fit(bert_classifier, train_dataloader, val_dataloader)
