"""
Purged cross-validation with embargo for financial time series.

Follows López de Prado (Advances in Financial Machine Learning, ch. 7): labels
built from forward windows leak into adjacent folds unless training samples whose
label intervals overlap the test window are removed, and an embargo gap is kept
after each test window so serially correlated features do not bleed across the split.
"""

from __future__ import annotations

from typing import Generator, Iterable, Optional, Tuple

import numpy as np


class PurgedKFold:
    """Expanding-window folds with label purge and post-test embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon: int = 5,
        embargo: int = 5,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if label_horizon < 1:
            raise ValueError("label_horizon must be at least 1")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[Iterable] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if n_samples <= self.n_splits + 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} purged folds"
            )

        test_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            test_start = (fold + 1) * test_size
            test_end = min(test_start + test_size, n_samples)
            if test_start >= test_end:
                continue

            test_idx = np.arange(test_start, test_end)
            train_end = test_start - self.embargo
            if train_end <= 0:
                continue

            train_candidates = np.arange(0, train_end)
            purge_cutoff = test_start - self.label_horizon
            train_idx = train_candidates[train_candidates < purge_cutoff]

            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[Iterable] = None,
    ) -> int:
        return self.n_splits
