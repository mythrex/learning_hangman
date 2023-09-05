import torch
import pytorch_lightning as pl
from metrics import Hitrate, NDCG
from argparse import ArgumentParser

from transformers import BertModel, BertConfig

import pdb


class BertClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        # Load the BERT configuration
        self.config = config
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_len,
        )

        # Instantiate the BERT-based classifier
        self.bert = BertModel(bert_config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.negative_sample_scaler = torch.nn.Sequential(
            torch.nn.Linear(config.vocab_size, config.vocab_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.vocab_size, config.vocab_size),
            torch.nn.Sigmoid(),
        )

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

        self.train_hr1 = Hitrate(k=1)
        self.train_hr6 = Hitrate(k=6)
        self.train_ndcg6 = NDCG(k=6)

        self.val_hr1 = Hitrate(k=1)
        self.val_hr6 = Hitrate(k=6)
        self.val_ndcg6 = NDCG(k=6)

        self.test_hr1 = Hitrate(k=1)
        self.test_hr6 = Hitrate(k=6)
        self.test_ndcg6 = NDCG(k=6)

    def forward(self, input_ids, attention_mask, negative_sample):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs["last_hidden_state"]
        # print(pooled_output.shape)
        logits = self.classifier(pooled_output)
        # negative_sample = negative_sample.unsqueeze(1).repeat(1, 64, 1)

        # scaler_value = self.negative_sample_scaler(torch.cat([
        #     logits,
        #     negative_sample
        # ], dim=-1))
        B, S, H = logits.shape
        scaler_value = (
            torch.sigmoid(self.negative_sample_scaler(negative_sample)).unsqueeze(1).repeat(1, S, 1)
        )
        # pdb.set_trace()
        return logits * scaler_value

    def loss_fn(self, logits, labels):
        idx = labels != -100
        return torch.nn.functional.cross_entropy(logits[idx], labels[idx])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=self.config.max_epochs, power=0.8
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["attention_mask"], batch["negative_sample"]
        )
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        self.train_hr1(logits, labels)
        self.train_hr6(logits, labels)
        self.train_ndcg6(logits, labels)
        return loss

    def on_train_epoch_end(self):
        result = {
            "train/hit@1": self.train_hr1.compute().item(),
            "train/hit@6": self.train_hr6.compute().item(),
            "train/ndcg@6": self.train_ndcg6.compute().item(),
        }
        self.log_dict(result)

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["attention_mask"], batch["negative_sample"]
        )
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        self.val_hr1(logits, labels)
        self.val_hr6(logits, labels)
        self.val_ndcg6(logits, labels)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        result = {
            "val/hit@1": self.val_hr1.compute().item(),
            "val/hit@6": self.val_hr6.compute().item(),
            "val/ndcg@6": self.val_ndcg6.compute().item(),
        }
        self.log_dict(result)

    def test_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["attention_mask"], batch["negative_sample"]
        )
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        self.test_hr1(logits, labels)
        self.test_hr6(logits, labels)
        self.test_ndcg6(logits, labels)
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        result = {
            "test/hit@1": self.test_hr1.compute().item(),
            "test/hit@6": self.test_hr6.compute().item(),
            "test/ndcg@6": self.test_ndcg6.compute().item(),
        }
        self.log_dict(result)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Model related
        parser.add_argument("--vocab_size", type=int, default=28)
        parser.add_argument("--hidden_size", type=int, default=64)
        parser.add_argument("--num_hidden_layers", type=int, default=6)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--num_attention_heads", type=int, default=4)
        parser.add_argument("--intermediate_size", type=int, default=128)
        parser.add_argument("--max_len", type=int, default=64)
        return parser


if __name__ == "__main__":
    from dataset import CustomDataset
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--filename", type=str, default=None)
    parser = BertClassifier.add_model_specific_args(parser)
    args = parser.parse_args(
        ["--filename=words_250000_train.txt", "--val_batch_size=2"]
    )
    bert_classifier = BertClassifier(args)
    test_dataset = CustomDataset(args.filename, split="test", max_len=args.max_len)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.val_batch_size, num_workers=8
    )
    batch = next(iter(test_dataloader))
    print(batch)
    output = bert_classifier(
        batch["input_ids"], batch["attention_mask"], batch["negative_sample"]
    )
    print(output)
