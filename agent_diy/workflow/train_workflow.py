#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import random

from agent_diy.feature.definition import (
    sample_process,
    FrameCollector,
    NONE_ACTION,
)
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # hok1v1 environment
    # hok1v1环境
    env = envs[0]

    # Number of agents, in hok1v1 the value is 2
    # 智能体数量，在hok1v1中值为2
    agent_num = len(agents)

    # Frame Collector
    # 帧收集器
    frame_collector = FrameCollector(agent_num)

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_diy/conf/train_env_conf.toml",
        logger=logger,
    )

    # Please implement your DIY algorithm flow
    # 请实现你DIY的算法流程
    # ......

    # Single environment process
    # 单局流程
    """
    while True:
        pass
    """

    return
