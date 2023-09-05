import torch


class Hitrate:
    def __init__(self, k: int) -> None:
        """
        Initialize Hitrate object.

        Args:
            k (int): The value of k for top-k calculation.
        """
        self.k = k
        self.hits = torch.tensor(0)
        self.n = torch.tensor(0)

    def compute(self) -> torch.Tensor:
        """
        Compute the hit rate.

        Returns:
            torch.Tensor: The computed hit rate.
        """
        return torch.tensor(self.hits / (self.n + 1e-10))

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the hit rate for a batch of logits and labels.

        Args:
            logits (torch.Tensor): The logits tensor.
            labels (torch.Tensor): The labels tensor.

        Returns:
            torch.Tensor: The computed hit rate.
        """
        idx = labels != -100
        topk = torch.softmax(logits.detach(), dim=-1)[idx].topk(self.k)
        relevant = topk.indices == labels[idx].unsqueeze(1)
        hits = relevant.sum()
        n = labels[idx].shape[0]
        self.hits += hits.cpu()
        self.n += n
        return torch.tensor(hits / (n + 1e-10))


class NDCG:
    def __init__(self, k: int) -> None:
        """
        Initialize NDCG object.

        Args:
            k (int): The value of k for top-k calculation.
        """
        self.k = k
        self.log_rank = torch.tensor(0.0)
        self.n = torch.tensor(0)

    def compute(self) -> torch.Tensor:
        """
        Compute the NDCG.

        Returns:
            torch.Tensor: The computed NDCG.
        """
        return torch.tensor(self.log_rank / (self.n + 1e-10))

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the NDCG for a batch of logits and labels.

        Args:
            logits (torch.Tensor): The logits tensor.
            labels (torch.Tensor): The labels tensor.

        Returns:
            torch.Tensor: The computed NDCG.
        """
        idx = labels != -100
        topk = torch.softmax(logits.detach(), dim=-1)[idx].topk(self.k)
        relevant = topk.indices == labels[idx].unsqueeze(1)
        B, k = relevant.shape
        rank = (torch.arange(k) + 1).unsqueeze(0).repeat(B, 1).to(relevant.device)
        log_rank = torch.log2(1 + rank) ** -1
        log_rank[~relevant] = 0
        log_rank = log_rank.sum()
        n = labels[idx].shape[0]
        self.log_rank += log_rank.cpu()
        self.n += n
        return torch.tensor(log_rank / (n + 1e-10))
