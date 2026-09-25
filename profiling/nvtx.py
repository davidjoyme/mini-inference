"""
profiling/nvtx.py - label code regions so they show up in Nsight Systems timelines.

    with nvtx_range("decode"):
        ...

Does nothing unless NVTX is enabled in config (set NVTX=1 in the environment).
"""
import contextlib

import torch

import config


@contextlib.contextmanager
def nvtx_range(name: str):
    if config.NVTX_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield