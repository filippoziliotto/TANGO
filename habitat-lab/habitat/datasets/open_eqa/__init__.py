#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_open_eqa_dataset():
    try:
        from habitat.datasets.open_eqa.open_eqa_dataset import (  # noqa: F401 isort:skip
            OpenEQADatasetV1,
        )
    except ImportError as e:
        open_eqa_import_error = e

        @registry.register_dataset(name="OPENEQA-v1")
        class OpenEQADatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise open_eqa_import_error