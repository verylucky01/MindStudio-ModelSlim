#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 调度 backend：可替换的执行策略（默认 wave）。"""

from msmodelslim.utils.distributed.task_scheduler.backend.base import DTSBackend
from msmodelslim.utils.distributed.task_scheduler.backend.wave import WaveDTSBackend

__all__ = ["DTSBackend", "WaveDTSBackend"]
