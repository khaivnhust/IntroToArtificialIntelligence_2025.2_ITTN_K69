"""
early_stopping.py — Early stopping callback for PyTorch training loops.

Monitors validation loss and halts training when improvement stalls for
a configurable number of epochs (``patience``).  The best model weights
are saved to disk whenever a new best is found.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum decrease in ``val_loss`` to qualify as an improvement.
    checkpoint_path : str or Path
        Where to save the best model weights.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        checkpoint_path: str | Path = "best_model.pt",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)

        self.best_validation_loss: float = float("inf")
        self.epochs_without_improvement: int = 0
        self.best_epoch: int = 0
        self.should_stop: bool = False

    # ------------------------------------------------------------------
    def step(self, validation_loss: float, model: nn.Module, current_epoch: int) -> bool:
        """Check whether training should stop.

        Saves model weights when a new best validation loss is achieved.

        Returns
        -------
        bool
            ``True`` if training should be halted.
        """
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.epochs_without_improvement = 0
            self.best_epoch = current_epoch
            self._save_checkpoint(model)
            logger.info(
                "  New best val_loss=%.6f at epoch %d — checkpoint saved.",
                validation_loss,
                current_epoch,
            )
        else:
            self.epochs_without_improvement += 1
            logger.info(
                "  EarlyStopping counter: %d/%d  (best=%.6f @ epoch %d)",
                self.epochs_without_improvement,
                self.patience,
                self.best_validation_loss,
                self.best_epoch,
            )
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True

        return self.should_stop

    # ------------------------------------------------------------------
    def _save_checkpoint(self, model: nn.Module) -> None:
        import torch
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
