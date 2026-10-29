# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F


def reduce_ranking_violations(positive_score, negative_scores, margin, reduction="mean"):
    """Mean hinge penalty for complete-caption scores."""
    if reduction != "mean":
        raise ValueError("The released objective uses mean negative reduction")
    return F.relu(float(margin) - positive_score + negative_scores).mean()
