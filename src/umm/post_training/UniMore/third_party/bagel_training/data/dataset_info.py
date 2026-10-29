# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .route_manifest_dataset import RouteManifestIterableDataset

DATASET_REGISTRY = {"route_manifest": RouteManifestIterableDataset}
DATASET_INFO = {"route_manifest": {}}
