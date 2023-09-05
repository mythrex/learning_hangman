import torch
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, List


class CharacterLevelTokenizer:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.vocab = {
            "<PAD>": 0,
            "<MASK>": 1,
        }
        offset = len(self.vocab)
        for k, v in zip("abcdefghijklmnopqrstuvwxyz", range(offset, 27 + offset)):
            self.vocab[k] = v

    def tokenize(
        self, word: str, labels: List[str] = None, return_pt=True
    ) -> Union[torch.Tensor, List[int]]:
        word = map(lambda x: x if x != "_" else "<MASK>", list(word))

        res = []
        for c in word:
            res.append(self.vocab[c])

        res += [self.vocab["<PAD>"]] * (self.max_len - len(res))

        if labels:
            labels = list(map(lambda x: self.vocab.get(x) if x else -100, labels))
            labels = labels + [-100] * (self.max_len - len(labels))
            labels = torch.tensor(labels).long()
            res = torch.tensor(res).long()
            return res, labels

        if return_pt:
            return torch.tensor(res).long()

        return res


def random_character_masker(word: str) -> Tuple[str, List[str]]:
    unique_chars = set(word)
    num_to_mask = random.randint(1, len(unique_chars))
    chars_to_mask = random.sample(unique_chars, num_to_mask)
    labels = [None for _ in range(len(word))]
    for c in chars_to_mask:
        for i, cc in enumerate(word):
            if c == cc:
                labels[i] = c
        word = word.replace(c, "_")
    return word, labels


def random_negative_sample(masked_word: str, label: str, vocab: dict) -> torch.Tensor:
    if random.random() < 0.5:
        set1 = list(set(masked_word.replace("_", "")))
        label = set(filter(lambda x: x is not None, label))
        negative_labels = set(vocab.keys()).difference(label)
        k = random.randint(1, 6)
        set2 = random.sample(negative_labels, k)
        negative_sample = list(map(lambda x: vocab[x], set1 + set2))
        res = [0. for i in range(len(vocab))]
        for i in negative_sample:
            res[i] = 1
        return torch.tensor(res).float()
    return torch.tensor([0.] * len(vocab))


class CustomDataset(Dataset):
    def __init__(self, word_file: str, split: str, max_len: int):
        with open(word_file, "r") as f:
            lines = f.readlines()

        lines = list(map(lambda x: x.replace("\n", ""), lines))
        self.split = split
        train, test = train_test_split(lines, test_size=0.2, random_state=42)
        del lines

        if split == "train":
            self.words = train
        else:
            val, test = train_test_split(test, test_size=0.5, random_state=42)
            if split == "val":
                self.words = val
            elif split == "test":
                self.words = test
            else:
                raise ValueError(f"Split should be in train/val/test but got {split}.")

        self.tokenizer = CharacterLevelTokenizer(max_len=max_len)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word = self.words[index]
        masked_word, label = random_character_masker(word)
        negative_sample = random_negative_sample(
            masked_word, label, self.tokenizer.vocab
        )
        masked_word, label = self.tokenizer.tokenize(masked_word, label)
        attention_mask = torch.ones_like(masked_word)
        idx = masked_word == self.tokenizer.vocab["<PAD>"]
        attention_mask[idx] = 0
        return {
            "input_ids": masked_word,
            "labels": label,
            "attention_mask": attention_mask,
            "negative_sample": negative_sample,
        }


if __name__ == "__main__":
    ds = CustomDataset(word_file="words_250000_train.txt", split="train", max_len=64)
    batch = next(iter(ds))
    print(batch)
    print(batch['labels'][batch['labels'] != -100])
    print(batch['negative_sample'][batch['labels'][batch['labels'] != -100]])
    print(len(ds.tokenizer.vocab))
