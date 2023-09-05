# Import necessary libraries
import wandb
import yaml
from dotenv import load_dotenv
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from pl_model import BertClassifier
from dataset import CustomDataset


load_dotenv()
wandb.login()

# Define your hyperparameters and their search spaces
sweep_config = {
    "name": "hangman",
    "method": "bayes",  # You can change the method to 'grid', 'bayes', etc.
    "metric": {"goal": "maximize", "name": "val/hit@6"},  # Change the metric as needed
    "parameters": {
        "hidden_size": {"values": [64, 128, 256, 512]},
        "num_hidden_layers": {"values": [4, 6, 8, 12, 16]},
        "num_attention_heads": {"values": [4, 8, 16]},
        "intermediate_size": {"values": [128, 256, 512, 1024]},
        "learning_rate": {"values": [0.01, 0.001, 0.0001]},
    },
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="hangman")


# Define your training function
def train():
    # Initialize wandb run
    with wandb.init() as run:
        parser = ArgumentParser()
        parser.add_argument("--max_epochs", type=int, default=1)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--val_batch_size", type=int, default=64)
        parser.add_argument("--accelerator", type=str, default="cpu")
        parser.add_argument("--filename", type=str, default=None)
        parser.add_argument("--wandb_project_name", type=str, default=None)
        args_list = [
            "--max_len=64",
            "--vocab_size=29",
            f"--hidden_size={wandb.config.hidden_size}",
            f"--num_hidden_layers={wandb.config.num_hidden_layers}",
            f"--num_attention_heads={wandb.config.num_attention_heads}",
            f"--intermediate_size={wandb.config.intermediate_size}",
            f"--learning_rate={wandb.config.learning_rate}",
            "--max_epochs=30",
            "--train_batch_size=256",
            "--val_batch_size=1024",
            "--accelerator=gpu",
            "--filename=sample.txt",
        ]
        parser = BertClassifier.add_model_specific_args(parser)
        args = parser.parse_args(args_list)

        bert_classifier = BertClassifier(args)
        wandb_logger = WandbLogger(project=args.wandb_project_name, log_model=True)
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            devices=1,
            accelerator=args.accelerator,
            logger=wandb_logger,
        )
        # Create train, validation, and test datasets
        train_dataset = CustomDataset(
            args.filename, split="train", max_len=args.max_len
        )
        val_dataset = CustomDataset(args.filename, split="val", max_len=args.max_len)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.val_batch_size, num_workers=8
        )
        # Train the model using the Trainer
        trainer.fit(bert_classifier, train_dataloader, val_dataloader)

        # Your training code here
        # You can access hyperparameters like config.learning_rate


# You can use wandb agent to start the sweep
# For example:
# wandb agent SWEEP_ID

# Or, if you want to run it locally for testing:
# Uncomment this and replace SWEEP_ID with the actual sweep ID
wandb.agent(sweep_id, function=train, count=50)
# train()