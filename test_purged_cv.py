#!/usr/bin/env python3
"""Structural check that purged CV actually purges overlapping labels.

This does not measure predictive edge or reproduce trading metrics.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtesting.purged_cv import PurgedKFold


def test_purged_kfold_purge_ran():
    """README claim: purged CV removes train rows whose labels overlap test.

    Embargo is set shorter than the label horizon so a skipped purge is
    visible. A skipped purge must fail this test. No edge is claimed.
    """
    n_samples = 200
    features = np.zeros((n_samples, 1))
    label_horizon = 12
    embargo = 3
    splitter = PurgedKFold(
        n_splits=5,
        label_horizon=label_horizon,
        embargo=embargo,
    )

    folds_checked = 0
    for train_idx, test_idx in splitter.split(features):
        test_start = int(test_idx.min())
        # Embargo already drops [test_start - embargo, test_start).
        # Purge must also drop [test_start - label_horizon, test_start - embargo).
        purge_only_zone = np.arange(test_start - label_horizon, test_start - embargo)
        assert len(purge_only_zone) > 0
        still_in_train = np.isin(purge_only_zone, train_idx)
        assert not still_in_train.any(), (
            "purge did not run: train still contains samples whose "
            "label windows overlap the test fold"
        )
        assert (train_idx < test_start - label_horizon).any()
        assert (train_idx >= test_start).sum() == 0
        folds_checked += 1

    assert folds_checked >= 1, "PurgedKFold produced no folds to check"
