# -*- coding: utf-8 -*-
"""
@inproceedings{,
  author    = {Rongkai Ma and
               Pengfei Fang and
               Gil Avraham and
               Yan Zuo and
               Tianyu Zhu and
               Tom Drummond and
               Mehrtash Harandi},
  title     = {Learning Instance and Task-Aware
               Dynamic Kernels for Few-Shot Learning},
  booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
               Addis Ababa, Ethiopia, April 26-30, 2020},
  year      = {2022},
  url       = {https://openreview.net/forum?id=rkgMkCEtPB}
}
https://arxiv.org/abs/1909.09157
"""

import torch
from torch import nn

from .meta_model import MetaModel


class INSTA(MetaModel):
    def __init__(self, init_type="normal", **kwargs):
        super().__init__(init_type, **kwargs)
