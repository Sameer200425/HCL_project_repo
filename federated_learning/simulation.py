"""
Federated Learning Simulation for Banking Fraud Detection.
============================================================
Simulates federated training across multiple bank branches:
  - Each branch has local data (privacy-preserved)
  - Local models train independently
  - Central server aggregates using FedAvg
  - Global model distributed back to branches

Demonstrates: Privacy-preserving AI for banking compliance.
============================================================
Usage:
  python federated_learning/simulation.py
"""

import copy
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ------------------------------------------------------------------ #
#  Client (Bank Branch)
# ------------------------------------------------------------------ #
class FederatedClient:
    """
    Simulated bank branch that trains a local model on its private data.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        local_epochs: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device

    def train(self) -> Dict:
        """Run local training and return model weights + metrics."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)

        return {
            "client_id": self.client_id,
            "model_state": copy.deepcopy(self.model.state_dict()),
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
            "num_samples": total_samples,
        }

    def update_model(self, global_state: dict):
        """Load global model weights."""
        self.model.load_state_dict(global_state)


# ------------------------------------------------------------------ #
#  Server (Central Aggregator)
# ------------------------------------------------------------------ #
class FederatedServer:
    """
    Central server that aggregates local models using FedAvg.
    """

    def __init__(self, global_model: nn.Module, device: torch.device):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_results: List[Dict]) -> dict:
        """
        Federated Averaging (FedAvg): weight models by data size.

        Args:
            client_results: List of dicts from client.train().

        Returns:
            Aggregated global model state dict.
        """
        total_samples = sum(r["num_samples"] for r in client_results)
        global_state = copy.deepcopy(self.global_model.state_dict())

        # Zero out global state
        for key in global_state:
            global_state[key] = torch.zeros_like(global_state[key]).float()

        # Weighted average
        for result in client_results:
            weight = result["num_samples"] / total_samples
            client_state = result["model_state"]
            for key in global_state:
                global_state[key] += client_state[key].float() * weight

        self.global_model.load_state_dict(global_state)
        return global_state

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate global model."""
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return {"accuracy": correct / max(total, 1), "total_samples": total}


# ------------------------------------------------------------------ #
#  Federated Simulation
# ------------------------------------------------------------------ #
def simulate_federated_learning(
    model: nn.Module,
    full_dataset,
    test_loader: DataLoader,
    num_clients: int = 5,
    num_rounds: int = 20,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "results/",
) -> Dict:
    """
    Run complete federated learning simulation.

    Args:
        model: Base model architecture.
        full_dataset: Full training dataset to split across clients.
        test_loader: Global test DataLoader.
        num_clients: Number of simulated bank branches.
        num_rounds: Communication rounds.
        local_epochs: Training epochs per client per round.
        batch_size: Batch size for local training.
        lr: Learning rate.
        device: Device.
        save_dir: Results save directory.

    Returns:
        Training history and final metrics.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Federated Learning Simulation")
    print(f"  Clients (Bank Branches): {num_clients}")
    print(f"  Communication Rounds: {num_rounds}")
    print(f"  Local Epochs: {local_epochs}")
    print(f"{'='*60}\n")

    # Split dataset across clients (non-IID simulation)
    total_size = len(full_dataset)
    indices = np.random.permutation(total_size)
    client_indices = np.array_split(indices, num_clients)

    # Create server
    server = FederatedServer(model, device)

    # Create clients
    clients = []
    for i in range(num_clients):
        subset = Subset(full_dataset, client_indices[i].tolist())
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        client = FederatedClient(
            client_id=i,
            model=model,
            train_loader=loader,
            lr=lr,
            local_epochs=local_epochs,
            device=device,
        )
        clients.append(client)
        print(f"  Branch {i}: {len(client_indices[i])} documents")

    history = {
        "round": [],
        "global_accuracy": [],
        "avg_client_loss": [],
        "avg_client_acc": [],
    }

    for round_num in range(1, num_rounds + 1):
        print(f"\n  Round [{round_num}/{num_rounds}]")

        # Send global model to all clients
        global_state = server.global_model.state_dict()
        for client in clients:
            client.update_model(global_state)

        # Local training
        client_results = []
        for client in clients:
            result = client.train()
            client_results.append(result)
            print(
                f"    Branch {result['client_id']}: "
                f"Loss={result['train_loss']:.4f} | "
                f"Acc={result['train_acc']:.4f}"
            )

        # Aggregate
        server.aggregate(client_results)

        # Evaluate global model
        eval_result = server.evaluate(test_loader)

        avg_loss = np.mean([r["train_loss"] for r in client_results])
        avg_acc = np.mean([r["train_acc"] for r in client_results])

        history["round"].append(round_num)
        history["global_accuracy"].append(eval_result["accuracy"])
        history["avg_client_loss"].append(avg_loss)
        history["avg_client_acc"].append(avg_acc)

        print(
            f"  → Global Accuracy: {eval_result['accuracy']:.4f} | "
            f"Avg Client Acc: {avg_acc:.4f}"
        )

    # Save results
    plot_federated_results(history, save_path=os.path.join(save_dir, "federated_learning.png"))

    # Save best global model
    ckpt_path = os.path.join(save_dir, "federated_global_model.pth")
    torch.save({
        "model_state_dict": server.global_model.state_dict(),
        "history": history,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
    }, ckpt_path)

    best_acc = max(history["global_accuracy"])
    print(f"\n  Federated learning complete. Best global acc: {best_acc:.4f}")
    print(f"  Model saved: {ckpt_path}")

    return history


def plot_federated_results(
    history: Dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot federated learning convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rounds = history["round"]

    # Global accuracy
    axes[0].plot(rounds, history["global_accuracy"], "o-", color="#2196F3", linewidth=2)
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Global Test Accuracy")
    axes[0].set_title("Federated Learning – Global Accuracy", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Client loss
    axes[1].plot(rounds, history["avg_client_loss"], "s-", color="#F44336", linewidth=2)
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Average Client Loss")
    axes[1].set_title("Federated Learning – Training Loss", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Privacy-Preserving Federated Learning for Banking",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
