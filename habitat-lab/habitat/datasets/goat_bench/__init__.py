#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_goat_dataset():
    try:
        from habitat.datasets.goat_bench.goat_dataset import (  # noqa: F401 isort:skip
            GoatDatasetV1,
        )
    except ImportError as e:
        goat_dataset_import_error = e

        @registry.register_dataset(name="Goat-v1")
        class GoatDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise goat_dataset_import_error
