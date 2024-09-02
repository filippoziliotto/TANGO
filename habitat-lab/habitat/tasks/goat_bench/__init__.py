#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_goat_task():
    try:
        from habitat.tasks.goat_bench.goat_task import GoatTask  # noqa: F401
    except ImportError as e:
        goat_task_import_error = e

        @registry.register_task(name="Goat-v1")
        class GoatTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise goat_task_import_error
