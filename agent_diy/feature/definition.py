#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls, Frame
from agent_diy.conf.conf import Config
import numpy as np
import collections
import random
import itertools
import os
import json


ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)

# ActData needs to contain d_action and d_prob, used for visualization
# ActData 需要包含 d_action 和 d_prob, 用于可视化智能体预测概率
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    d_prob=None,
    value=None,
    lstm_cell=None,
    lstm_hidden=None,
)

# SampleData for training, total dimension is sum of all data_shapes
# SampleData 用于训练，总维度是所有 data_shapes 的总和
SampleData = create_cls(
    "SampleData",
    sample=1,
)

NONE_ACTION = [0, 15, 15, 15, 15, 0]


def sample_process(collector):
    return collector.sample_process()


class FrameCollector:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def sample_process(self):
        return
