#!/usr/bin/env python3
"""PyTorch counterpart of benchmarks/mnist_mlp.cpp, for comparing accuracy and for
profiling the two side by side (Nsight Systems / Nsight Compute); it takes no timings itself.

Same data (the IDX files the build downloads into datasets/mnist, pixels scaled to [0, 1]),
same model (784 -> 256 -> 128 -> 10 with ReLU and dropout), same loss, optimizer and
hyperparameters. nn.Linear's default initialization, U(-1/sqrt(in), 1/sqrt(in)), is the one
flashgrad's Linear uses. The random streams differ, so the numbers match only statistically.

    python benchmarks/mnist_mlp_torch.py [mnist-dir]

Needs only torch and numpy (no torchvision).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:  # keep in sync with Config in benchmarks/mnist_mlp.cpp
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 128
    eval_batch_size: int = 1000
    lr: float = 1e-3
    num_workers: int = 4
    log_every: int = 100
    seed: int = 0
    required_accuracy: float = 0.97


class MnistDataset(Dataset):
    """MNIST from the four unpacked files in `root`, held in memory as two tensors."""

    def __init__(self, root: Path, train: bool = True):
        prefix = "train" if train else "t10k"
        n = 60000 if train else 10000
        self.images = self._load(root / f"{prefix}-images-idx3-ubyte", 16, (n, 1, 28, 28)) / 255.0
        self.labels = self._load(root / f"{prefix}-labels-idx1-ubyte", 8, (n,)).long()

    @staticmethod
    def _load(path: Path, header_bytes: int, shape: tuple) -> torch.Tensor:
        """Skips the header and reads one byte per pixel or label into a float tensor."""
        data = np.fromfile(path, dtype=np.uint8, offset=header_bytes)
        if data.size != np.prod(shape):
            raise ValueError(f"MnistDataset: expected {np.prod(shape)} bytes in {path}, "
                             f"got {data.size} (still gzip-compressed?)")
        return torch.from_numpy(data.astype(np.float32)).reshape(shape)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Accuracy of `model` over one pass of `loader`, in evaluation mode."""
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images.flatten(1)).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> int:
    config = Config()
    default_dir = Path(__file__).resolve().parent.parent / "datasets" / "mnist"
    mnist_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    if not torch.cuda.is_available():
        print("no CUDA device available, skipping")
        return 0
    device = torch.device("cuda")
    torch.manual_seed(config.seed)

    train_loader = DataLoader(MnistDataset(mnist_dir, train=True), batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              persistent_workers=config.num_workers > 0)
    test_loader = DataLoader(MnistDataset(mnist_dir, train=False),
                             batch_size=config.eval_batch_size, num_workers=config.num_workers,
                             persistent_workers=config.num_workers > 0)

    model = nn.Sequential(
        nn.Linear(784, config.hidden1),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden1, config.hidden2),
        nn.ReLU(),
        nn.Linear(config.hidden2, 10),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"training on {len(train_loader.dataset)} images, "
          f"{len(train_loader)} batches per epoch")

    step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # [B, 1, 28, 28] -> [B, 784]
            loss = loss_fn(model(images.flatten(1)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % config.log_every == 0:
                print(f"epoch {epoch}, step {step}: loss {loss.item():.6g}")
        print(f"epoch {epoch}: test accuracy {evaluate(model, test_loader, device):.4f}")

    test_accuracy = evaluate(model, test_loader, device)
    if test_accuracy < config.required_accuracy:
        print(f"training failed: test accuracy {test_accuracy:.4f} < {config.required_accuracy}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
