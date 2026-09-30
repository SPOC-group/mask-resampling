"""Compatibility shims for running the original MAE stack on modern PyTorch.

The repo pins timm==0.3.2 and imports torch._six, which was removed in recent
PyTorch releases. Python loads this file automatically from the repo root before
other imports, so the training scripts can still use the old dependency stack.
"""
import collections.abc
import math
import sys
import types


torch_six = types.ModuleType('torch._six')
torch_six.container_abcs = collections.abc
torch_six.inf = math.inf
torch_six.string_classes = (str, bytes)
torch_six.int_classes = (int,)

sys.modules.setdefault('torch._six', torch_six)
