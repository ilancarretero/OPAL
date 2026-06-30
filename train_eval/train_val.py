import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Tuple


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.0
) -> Tuple[float, float]:
    """
    Perform one training epoch.

    Args:
        model: PyTorch model with fixed prototype head.
        dataloader: DataLoader for training data.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer for model parameters.
        device: Device to run computations on.
        alpha: Weight for optional sparse penalty (default 0.0).

    Returns:
        Tuple containing:
            - average_loss: Mean loss over the epoch.
            - accuracy: Top-1 accuracy percentage.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Compute classification loss
        ce_loss = criterion(logits, labels)

        # Optional sparse penalty (commented out by default)
        # embedding = model.get_embedding_train(images)
        # sparse_loss = torch.norm(embedding, 1) + torch.norm(embedding, 2) ** 2
        # loss = ce_loss + alpha * sparse_loss
        loss = ce_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        _, predicted = torch.max(logits, dim=1)
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Perform one validation/testing epoch without parameter updates.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader for validation or test data.
        criterion: Loss function.
        device: Device to run computations on.

    Returns:
        Tuple containing:
            - average_loss: Mean loss over the epoch.
            - accuracy: Top-1 accuracy percentage.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs, dim=1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    average_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy