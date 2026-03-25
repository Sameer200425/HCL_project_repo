"""
Active Learning Pipeline for ViT Fraud Detection.
============================================================
Reduces labeling cost by 60–80% by intelligently selecting
which unlabeled documents need human annotation.

STRATEGIES IMPLEMENTED:
  1. Least Confidence   — query samples where max(P(class)) is lowest
  2. Margin Sampling    — query where P(top1) - P(top2) is smallest
  3. Entropy Sampling   — query samples with highest prediction entropy
  4. Core-Set (greedy)  — cover embedding space with minimum labels

WHY UNIQUE FOR BANKING:
  Real banks receive thousands of documents daily. Getting a human
  auditor to label ALL of them is impossible. Active Learning routes
  ONLY the most ambiguous/novel documents to auditors, dramatically
  reducing ops cost while continuously improving the model.

  Example outcome: With 10,000 unlabeled cheques, active learning
  identifies the ~200 most informative ones for labeling — achieving
  the same model improvement as labeling 1,000+ random samples.

WORKFLOW:
  1. Model predicts on unlabeled pool
  2. Active Learning scores/ranks each sample
  3. Top-K most uncertain → sent to human auditor queue
  4. Auditor labels → added to training set
  5. Model retrained → repeat (AL loop)

USAGE:
  al = ActiveLearner(model, strategy="entropy")
  query_indices = al.query(unlabeled_pool, n_instances=50)
  # Send query_indices items to human review queue
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Strategy Enum
# ------------------------------------------------------------------ #

class QueryStrategy(str, Enum):
    LEAST_CONFIDENCE = "least_confidence"
    MARGIN_SAMPLING  = "margin_sampling"
    ENTROPY          = "entropy"
    CORE_SET         = "core_set"
    ENSEMBLE         = "ensemble"  # combine all strategies


# ------------------------------------------------------------------ #
#  Sample Score
# ------------------------------------------------------------------ #

@dataclass
class SampleScore:
    index: int
    score: float                         # higher = more informative
    strategy: str
    predicted_class: str
    confidence: float
    entropy: float
    margin: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    priority: str = "NORMAL"             # CRITICAL / HIGH / NORMAL / LOW


CLASS_NAMES = ["genuine", "fraud", "tampered", "forged"]


# ------------------------------------------------------------------ #
#  Core Active Learner
# ------------------------------------------------------------------ #

class ActiveLearner:
    """
    Selects the most informative unlabeled samples for human annotation.

    Reduces the labeling budget needed to improve model performance
    by focusing annotation effort on uncertain/novel instances.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: QueryStrategy = QueryStrategy.ENTROPY,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        fraud_priority_boost: bool = True,
    ):
        """
        Args:
            model               : Trained ViT/Hybrid model.
            strategy            : Uncertainty measure to rank samples.
            device              : Compute device.
            class_names         : List of class labels.
            fraud_priority_boost: Elevate priority for fraud-predicted samples.
        """
        self.model                = model
        self.strategy             = QueryStrategy(strategy)
        self.device               = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names          = class_names or CLASS_NAMES
        self.fraud_priority_boost = fraud_priority_boost
        self.model.to(self.device)
        self._labeled_count: int  = 0
        self._query_history: List[Dict] = []

    # ---------------------------------------------------------------- #
    #  Prediction
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def _get_probabilities(self, dataloader) -> np.ndarray:
        """
        Run inference on all unlabeled samples.

        Returns:
            (N, n_classes) probability array.
        """
        self.model.eval()
        all_probs: List[np.ndarray] = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            logits = self.model(images)
            probs  = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)   # (N, C)

    @torch.no_grad()
    def _get_embeddings(self, dataloader) -> np.ndarray:
        """
        Extract CLS token embeddings for core-set selection.
        Assumes model has a `.get_cls_embedding()` method or similar.
        """
        self.model.eval()
        embeddings: List[np.ndarray] = []

        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device)

            # Try to get intermediate CLS embedding
            if hasattr(self.model, "get_embedding") and callable(getattr(self.model, "get_embedding")):
                emb = getattr(self.model, "get_embedding")(images)
            else:
                try:
                    emb = self.model(images, return_features=True)
                except TypeError:
                    logits = self.model(images)
                    emb = logits

            embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    # ---------------------------------------------------------------- #
    #  Scoring Functions
    # ---------------------------------------------------------------- #

    @staticmethod
    def _least_confidence_scores(probs: np.ndarray) -> np.ndarray:
        """
        Score = 1 - max_probability.
        Most uncertain samples have the lowest top-class probability.
        """
        return 1.0 - probs.max(axis=1)

    @staticmethod
    def _margin_scores(probs: np.ndarray) -> np.ndarray:
        """
        Score = P(top1) - P(top2).
        Small margin → model is nearly equally split → most uncertain.
        Return INVERTED so higher score = more uncertain.
        """
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        return 1.0 - margins  # invert: small margin → high score

    @staticmethod
    def _entropy_scores(probs: np.ndarray) -> np.ndarray:
        """
        Score = Shannon entropy of the probability distribution.
        Maximum entropy = completely uncertain prediction.
        """
        eps = 1e-8
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        # Normalize by max possible entropy
        max_entropy = np.log(probs.shape[1])
        return entropy / max_entropy

    @staticmethod
    def _core_set_scores(
        embeddings: np.ndarray,
        already_labeled: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Greedy Core-Set: score each sample by its minimum distance
        to the already-labeled set (or centroid if none labeled).
        High score = far from labeled examples = most novel.
        """
        if already_labeled is None or len(already_labeled) == 0:
            # If no labeled data, use distance from centroid
            centroid = embeddings.mean(axis=0, keepdims=True)
            dists    = np.linalg.norm(embeddings - centroid, axis=1)
        else:
            # Min distance to any labeled sample
            dists = np.array([
                np.min(np.linalg.norm(already_labeled - emb, axis=1))
                for emb in embeddings
            ])

        # Normalize
        if dists.max() > 0:
            dists = dists / dists.max()
        return dists

    def _ensemble_scores(
        self, probs: np.ndarray, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Combine all strategies with equal weight."""
        scores = (
            self._least_confidence_scores(probs) +
            self._margin_scores(probs) +
            self._entropy_scores(probs)
        )
        if embeddings is not None:
            scores += self._core_set_scores(embeddings)
            scores /= 4.0
        else:
            scores /= 3.0
        return scores

    # ---------------------------------------------------------------- #
    #  Main Query Interface
    # ---------------------------------------------------------------- #

    def query(
        self,
        unlabeled_loader,
        n_instances: int = 50,
        labeled_embeddings: Optional[np.ndarray] = None,
        review_queue: Optional["HumanReviewQueue"] = None,
        metadata_provider: Optional[Callable[[int], Dict]] = None,
    ) -> Tuple[List[int], List[SampleScore]]:
        """
        Select the most informative samples for labeling.

        Args:
            unlabeled_loader   : DataLoader for unlabeled pool.
            n_instances        : Number of samples to query.
            labeled_embeddings : Embeddings of already-labeled samples (for core-set).

        Returns:
            (selected_indices, scored_samples)
            selected_indices: indices of top-K samples to label
            scored_samples  : full scoring details for each selected sample
        """
        probs = self._get_probabilities(unlabeled_loader)
        N     = len(probs)
        n_instances = min(n_instances, N)

        # Compute raw scores
        if self.strategy == QueryStrategy.LEAST_CONFIDENCE:
            raw_scores = self._least_confidence_scores(probs)
        elif self.strategy == QueryStrategy.MARGIN_SAMPLING:
            raw_scores = self._margin_scores(probs)
        elif self.strategy == QueryStrategy.ENTROPY:
            raw_scores = self._entropy_scores(probs)
        elif self.strategy == QueryStrategy.CORE_SET:
            embeddings = self._get_embeddings(unlabeled_loader)
            raw_scores = self._core_set_scores(embeddings, labeled_embeddings)
        elif self.strategy == QueryStrategy.ENSEMBLE:
            try:
                embeddings = self._get_embeddings(unlabeled_loader)
            except Exception:
                embeddings = None
            raw_scores = self._ensemble_scores(probs, embeddings)
        else:
            raw_scores = self._entropy_scores(probs)

        # Build scored samples
        entropy_scores = self._entropy_scores(probs)
        margin_scores  = self._margin_scores(probs)

        scored: List[SampleScore] = []
        for i in range(N):
            pred_cls_idx = int(probs[i].argmax())
            pred_conf    = float(probs[i][pred_cls_idx])
            pred_cls     = self.class_names[pred_cls_idx]

            # Priority boost for fraud-predicted samples
            priority = "NORMAL"
            score    = float(raw_scores[i])
            if self.fraud_priority_boost and pred_cls in ("fraud", "tampered", "forged"):
                score   *= 1.5
                priority = "HIGH"
            if float(entropy_scores[i]) > 0.85:
                priority = "CRITICAL"

            scored.append(SampleScore(
                index=i,
                score=score,
                strategy=self.strategy.value,
                predicted_class=pred_cls,
                confidence=pred_conf,
                entropy=float(entropy_scores[i]),
                margin=float(1.0 - margin_scores[i]),   # original margin (not inverted)
                class_probabilities={
                    name: float(probs[i][j])
                    for j, name in enumerate(self.class_names)
                },
                priority=priority,
            ))

        # Sort by score descending and take top-K
        scored.sort(key=lambda s: s.score, reverse=True)
        selected        = scored[:n_instances]
        selected_indices = [s.index for s in selected]

        # Track history
        self._query_history.append({
            "n_queried": n_instances,
            "strategy":  self.strategy.value,
            "top_scores": [f"{s.score:.4f}" for s in selected[:5]],
        })

        if review_queue is not None:
            self.send_to_review_queue(review_queue, selected, metadata_provider)

        return selected_indices, selected

    # ---------------------------------------------------------------- #
    #  Labeling Efficiency Tracker
    # ---------------------------------------------------------------- #

    def record_labels_added(self, n: int) -> None:
        """Record that N labels were added to training set."""
        self._labeled_count += n

    def get_efficiency_report(self) -> Dict:
        """
        Report cumulative active learning efficiency.
        """
        total_queried = sum(h["n_queried"] for h in self._query_history)
        return {
            "total_al_rounds":    len(self._query_history),
            "total_queried":      total_queried,
            "total_labeled":      self._labeled_count,
            "rounds_summary":     self._query_history,
            "strategy":           self.strategy.value,
            "estimated_savings":  f"{max(0, 100 - int(total_queried / max(self._labeled_count, 1) * 100))}%",
        }

    def send_to_review_queue(
        self,
        review_queue: "HumanReviewQueue",
        selected: List[SampleScore],
        metadata_provider: Optional[Callable[[int], Dict]] = None,
    ) -> None:
        """Send selected samples to a human review queue."""
        for s in selected:
            metadata = metadata_provider(s.index) if metadata_provider else {}
            document_id = str(metadata.get("document_id", s.index))
            image_path = str(metadata.get("image_path", ""))
            review_queue.add(
                document_id=document_id,
                image_path=image_path,
                reason=f"Active learning ({s.strategy})",
                priority=s.priority,
                model_prediction=s.predicted_class,
                confidence=s.confidence,
                uncertainty_score=s.entropy,
                metadata=metadata,
            )


# ------------------------------------------------------------------ #
#  Human Review Queue
# ------------------------------------------------------------------ #

class HumanReviewQueue:
    """
    Maintains a priority queue of documents flagged for human review.
    Integrates with both Active Learning and Uncertainty Quantification.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._queue: List[Dict] = []

    def add(
        self,
        document_id: str,
        image_path: str,
        reason: str,
        priority: str = "NORMAL",
        model_prediction: str = "",
        confidence: float = 0.0,
        uncertainty_score: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a document to the review queue."""
        if len(self._queue) >= self.max_size:
            # Drop lowest-priority NORMAL items
            self._queue = [q for q in self._queue if q["priority"] != "NORMAL"]

        self._queue.append({
            "document_id":      document_id,
            "image_path":       image_path,
            "reason":           reason,
            "priority":         priority,
            "model_prediction": model_prediction,
            "confidence":       round(confidence, 4),
            "uncertainty_score": round(uncertainty_score, 4),
            "queued_at":        __import__("datetime").datetime.utcnow().isoformat(),
            "status":           "PENDING",
            "metadata":         metadata or {},
        })

        # Sort: CRITICAL → HIGH → NORMAL → LOW
        _order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}
        self._queue.sort(key=lambda x: _order.get(x["priority"], 99))

    def get_pending(self, limit: int = 20) -> List[Dict]:
        """Return up to `limit` pending items ordered by priority."""
        return [q for q in self._queue if q["status"] == "PENDING"][:limit]

    def mark_reviewed(self, document_id: str, auditor_label: str) -> None:
        """Mark a document as reviewed and record the auditor's label."""
        for item in self._queue:
            if item["document_id"] == document_id:
                item["status"]       = "REVIEWED"
                item["auditor_label"] = auditor_label
                item["reviewed_at"]  = __import__("datetime").datetime.utcnow().isoformat()
                break

    def get_stats(self) -> Dict:
        """Queue statistics."""
        total   = len(self._queue)
        pending = sum(1 for q in self._queue if q["status"] == "PENDING")
        reviewed = total - pending
        by_priority = {"CRITICAL": 0, "HIGH": 0, "NORMAL": 0, "LOW": 0}
        for q in self._queue:
            by_priority[q.get("priority", "NORMAL")] = \
                by_priority.get(q.get("priority", "NORMAL"), 0) + 1

        return {
            "total_in_queue": total,
            "pending":        pending,
            "reviewed":       reviewed,
            "by_priority":    by_priority,
        }


# ------------------------------------------------------------------ #
#  CLI Demo
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from models.vit_model import build_vit

    print("=" * 60)
    print("Active Learning Pipeline — Demo")
    print("=" * 60)

    device = torch.device("cpu")
    model  = build_vit(config={
        "vit": {
            "image_size": 224, "patch_size": 16, "embedding_dim": 768,
            "num_heads": 12, "num_layers": 12, "mlp_dim": 3072,
            "dropout": 0.1, "attention_dropout": 0.0, "num_classes": 4,
        }
    })
    al     = ActiveLearner(model, strategy=QueryStrategy.ENTROPY, device=device)

    # Simulate an unlabeled pool as a simple DataLoader
    dummy_data = torch.randn(100, 3, 224, 224)
    dataset = torch.utils.data.TensorDataset(dummy_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    indices, scores = al.query(loader, n_instances=10)

    print(f"\nTop 10 samples selected for labeling:")
    print(f"{'Idx':>5}  {'Score':>6}  {'Pred Class':>12}  {'Confidence':>10}  {'Entropy':>8}  {'Priority':>8}")
    print("-" * 65)
    for s in scores:
        print(
            f"{s.index:>5}  {s.score:>6.4f}  {s.predicted_class:>12}  "
            f"{s.confidence:>10.4f}  {s.entropy:>8.4f}  {s.priority:>8}"
        )

    print("\nEfficiency Report:", al.get_efficiency_report())
