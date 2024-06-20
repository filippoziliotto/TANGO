#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_open_eqa_task():
    try:
        from habitat.tasks.open_eqa.open_eqa import OPENEQATask  # noqa: F401
    except ImportError as e:
        open_eqatask_import_error = e

        @registry.register_task(name="OPENEQA-v0")
        class OPENEQATaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise open_eqatask_import_error
