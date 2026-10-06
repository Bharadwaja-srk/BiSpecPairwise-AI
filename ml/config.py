"""BSPAI shared config: env-driven paths, seeds, thresholds."""
from __future__ import annotations

import logging
import os
import random

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_pairwise_v1.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data")
ML_PORT = int(os.getenv("ML_PORT", "8001"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Expression positivity threshold (log-normalized counts).
# Paper (Zhang et al. 2024) defines P_gene = mean(I(expr > T)) but does not
# publish T; default T=0.0 is documented and configurable per call.
DEFAULT_EXPR_THRESHOLD = float(os.getenv("EXPR_THRESHOLD", "0.0"))


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
