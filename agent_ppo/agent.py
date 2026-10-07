#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os

import numpy as np
import torch
from kaiwudrl.interface.agent import BaseAgent
from torch.optim.lr_scheduler import LambdaLR

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# Available summoner skills / 可选召唤师技能
SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}
SUMMONER_SKILL_IDS = list(SUMMONER_SKILL_MAP.keys())
FIXED_SUMMONER_SKILL_ID = GameConfig.FIXED_SUMMONER_SKILL_ID
STAGE_A_PLUS_SUMMONER_SKILL_ID = FIXED_SUMMONER_SKILL_ID


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.last_total_be_hurt_by_hero = None
        self.last_skill_hit_totals = {1: None, 2: None, 3: None}
        self.skill_hit_follow_frames = 0
        self.luban_sweep_follow_frames = 0
        self.last_action_button = None
        self.skill_prior_stats = self._new_skill_prior_stats()

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        # Select summoner skill for each hero based on hero lineup of both camps
        # 根据双方阵营英雄阵容，为己方每个英雄选择召唤师技能
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for hero_id in my_heroes:
            select_skills[hero_id] = FIXED_SUMMONER_SKILL_ID
        return select_skills

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)
        self.last_total_be_hurt_by_hero = None
        self.last_skill_hit_totals = {1: None, 2: None, 3: None}
        self.skill_hit_follow_frames = 0
        self.luban_sweep_follow_frames = 0
        self.last_action_button = None
        self.skill_prior_stats = self._new_skill_prior_stats()

    def _model_inference(self, list_obs_data, stochastic=True):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.detach().cpu().numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i], stochastic)
            list_act_data.append(
                ActData(
                    action=action if stochastic else d_action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data], stochastic=True)[0]
        self._maybe_apply_skill_exploration(observation, act_data)
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        self.last_action_button = int(action[0]) if action is not None and len(action) > 0 else None
        return action

    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data], stochastic=False)[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return d_action

    def observation_process(self, observation):
        feature = self.feature_processes.process_feature(observation)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def _maybe_apply_skill_exploration(self, observation, act_data):
        if not getattr(GameConfig, "ENABLE_SKILL_EXPLORATION", True):
            return
        context = self._skill_exploration_context(observation)
        if not context:
            return

        # 收集诊断指标
        self._collect_skill_diag_stats(observation, context)

        follow_action = self._skill_follow_action(observation, act_data.action, context)
        if follow_action is not None:
            act_data.action = follow_action
            return

        candidates = []
        if context["hero_id"] == 112:
            # 鲁班 skill1：普攻衔接技能，放宽条件
            if (
                context["enemy_visible"]
                and context["enemy_mid_close"]
                and context["self_hp_ratio"] > 0.35
                and not context["enemy_tower_danger"]
                and np.random.random() < getattr(GameConfig, "LUBAN_SKILL1_PRIOR_PROB", 0.025)
            ):
                candidates.append(("luban_s1_poke", GameConfig.BUTTON_SKILL_1, (GameConfig.TARGET_ENEMY,)))
            # 鲁班 skill2：残血收割
            if (
                context["enemy_visible"]
                and context["enemy_low_hp"]
                and (context["enemy_far"] or context["enemy_retreat_like"])
                and (context["self_hp_advantage"] > 0.05 or self.last_action_button in (GameConfig.BUTTON_ATTACK, GameConfig.BUTTON_SKILL_1))
                and np.random.random() < getattr(GameConfig, "LUBAN_SKILL2_FINISH_PRIOR_PROB", 0.045)
            ):
                candidates.append(("luban_s2_finish", GameConfig.BUTTON_SKILL_2, (GameConfig.TARGET_ENEMY,)))
            # 鲁班 skill2：贴脸自保
            if (
                context["enemy_visible"]
                and context["enemy_melee_close"]
                and (context["self_hp_ratio"] < 0.55 or context["recent_hurt_by_hero"] > 0.0)
                and np.random.random() < getattr(GameConfig, "LUBAN_SKILL2_DEFENSE_PRIOR_PROB", 0.040)
            ):
                candidates.append(("luban_s2_defense", GameConfig.BUTTON_SKILL_2, (GameConfig.TARGET_ENEMY,)))
            # 鲁班 skill3：清兵、压塔、推塔窗口、区域压制 - 大幅放宽条件
            # 条件：有兵线/敌方小兵多/推塔窗口/敌塔低血量/敌方英雄在塔下但我方有兵线
            luban_s3_context = (
                context["push_window"]  # 推塔窗口
                or context["enemy_soldier_cluster"]  # 敌方小兵数量较多
                or context["has_minion_under_tower"]  # 我方兵线正在进塔
                or context["enemy_near_enemy_tower"]  # 敌方英雄在塔下
                or context.get("enemy_tower_low_hp", False)  # 敌塔低血量
            )
            if (
                luban_s3_context
                and context["self_hp_ratio"] > 0.35
                and not context["enemy_tower_danger"]
                and np.random.random() < getattr(GameConfig, "LUBAN_SKILL3_PUSH_ZONE_PRIOR_PROB", 0.050)  # 提高概率
            ):
                candidates.append(
                    (
                        "luban_s3",
                        GameConfig.BUTTON_SKILL_3,
                        (
                            *GameConfig.TARGET_SOLDIERS,
                            GameConfig.TARGET_TOWER,
                            GameConfig.TARGET_ENEMY,
                        ),
                    )
                )
        elif context["hero_id"] == 133:
            # 狄仁杰 skill1：消耗和清线
            if (
                context["enemy_visible"]
                and context["enemy_mid_close"]
                and context["self_hp_ratio"] > 0.35
                and not context["enemy_tower_danger"]
                and np.random.random() < getattr(GameConfig, "DIRENJIE_SKILL1_POKE_PRIOR_PROB", 0.025)
            ):
                candidates.append(("direnjie_s1_poke", GameConfig.BUTTON_SKILL_1, (GameConfig.TARGET_ENEMY,)))
            # 狄仁杰 skill2：防守技能 - 大幅放宽条件
            # 条件：低血量/刚受到伤害/正在撤退/敌方靠近
            direnjie_s2_context = (
                context["self_hp_ratio"] < 0.55  # 放宽血量阈值
                or context["recent_hurt_by_hero"] > 0.0  # 刚受到伤害
                or context["retreat_window"]  # 正在撤退
                or context["enemy_melee_close"]  # 敌方靠近
            )
            if (
                direnjie_s2_context
                and np.random.random() < getattr(GameConfig, "DIRENJIE_SKILL2_DEFENSE_PRIOR_PROB", 0.050)  # 提高概率
            ):
                candidates.append(("direnjie_s2_defense", GameConfig.BUTTON_SKILL_2, (GameConfig.TARGET_SELF, GameConfig.TARGET_NONE)))
            # 狄仁杰 skill3：控制技能，中距离 target enemy
            if (
                context["enemy_visible"]
                and context["enemy_distance_suitable"]
                and context["self_hp_ratio"] > 0.45  # 自己血量不低
                and not context["enemy_tower_danger"]
                and np.random.random() < getattr(GameConfig, "DIRENJIE_SKILL3_CONTROL_PRIOR_PROB", 0.035)
            ):
                candidates.append(("direnjie_s3", GameConfig.BUTTON_SKILL_3, (GameConfig.TARGET_ENEMY,)))

        # === 探索兜底机制：3%~5% 概率尝试技能 ===
        if not candidates:
            candidates = self._skill_probe_candidates(observation, context)

        for prior_key, button, preferred_targets in candidates:
            if prior_key == "luban_s3":
                preferred_targets = self._luban_s3_targets(context)
            if not self._prior_allowed(observation, context, prior_key, button, preferred_targets):
                continue
            self._record_prior(prior_key, "try")
            explored_action = self._build_explore_action(observation, act_data.action, button, preferred_targets)
            if explored_action is not None:
                self._record_prior(prior_key, "apply")
                act_data.action = explored_action
                return

    def _new_skill_prior_stats(self):
        return {
            "luban_s2_prior_try": 0.0,
            "luban_s2_prior_apply": 0.0,
            "luban_s3_prior_try": 0.0,
            "luban_s3_prior_apply": 0.0,
            "direnjie_s2_defense_prior_try": 0.0,
            "direnjie_s2_defense_prior_apply": 0.0,
            "direnjie_s3_prior_try": 0.0,
            "direnjie_s3_prior_apply": 0.0,
            "prior_blocked_by_cd": 0.0,
            "prior_blocked_by_mask": 0.0,
            "prior_blocked_by_tower_risk": 0.0,
            "prior_blocked_by_low_hp_retreat": 0.0,
            "luban_s2_prior_block_cd": 0.0,
            "luban_s2_prior_block_mask": 0.0,
            "luban_s2_prior_block_tower_risk": 0.0,
            "luban_s2_prior_block_retreat": 0.0,
            "luban_s3_prior_block_cd": 0.0,
            "luban_s3_prior_block_mask": 0.0,
            "luban_s3_prior_block_tower_risk": 0.0,
            "luban_s3_prior_block_retreat": 0.0,
            "direnjie_s2_prior_block_cd": 0.0,
            "direnjie_s2_prior_block_mask": 0.0,
            "direnjie_s2_prior_block_tower_risk": 0.0,
            "direnjie_s2_prior_block_retreat": 0.0,
            "direnjie_s3_prior_block_cd": 0.0,
            "direnjie_s3_prior_block_mask": 0.0,
            "direnjie_s3_prior_block_tower_risk": 0.0,
            "direnjie_s3_prior_block_retreat": 0.0,
            # 诊断指标：ready/legal/context 率（按英雄分开统计帧数）
            "luban_s2_ready": 0.0,
            "luban_s2_legal": 0.0,
            "luban_s2_context": 0.0,
            "luban_s3_ready": 0.0,
            "luban_s3_legal": 0.0,
            "luban_s3_context": 0.0,
            "luban_frame_count": 0.0,  # 鲁班出场帧数
            "direnjie_s2_ready": 0.0,
            "direnjie_s2_legal": 0.0,
            "direnjie_s2_context": 0.0,
            "direnjie_s3_ready": 0.0,
            "direnjie_s3_legal": 0.0,
            "direnjie_s3_context": 0.0,
            "direnjie_frame_count": 0.0,  # 狄仁杰出场帧数
        }

    def _skill_probe_candidates(self, observation, context):
        """探索兜底机制：当 skill2/skill3 ready 且 legal，以 3%~5% 概率尝试
        
        重要：每帧最多只做一次 forced probe，不要多个技能重复抽样
        """
        candidates = []
        max_probe_prob = getattr(GameConfig, "MAX_SKILL_PROBE_RATE", 0.05)
        
        # 先抽一次总概率，决定本帧是否做 probe
        if np.random.random() >= max_probe_prob:
            return candidates  # 本帧不做 probe
        
        # 安全条件检查
        if context["self_hp_ratio"] < 0.30 and not context["retreat_window"]:
            return candidates  # 低血量且非撤退窗口不乱放
        if context["enemy_tower_danger"]:
            return candidates  # 敌塔危险不乱冲
        
        # 收集所有可用的 probe 候选
        available_probes = []
        
        if context["hero_id"] == 112:
            # 鲁班 skill2 probe：优先 target enemy
            if (
                context["skill2_cd_ratio"] <= 0.02
                and self._legal_button(observation, GameConfig.BUTTON_SKILL_2)
            ):
                available_probes.append(("luban_s2_probe", GameConfig.BUTTON_SKILL_2, (GameConfig.TARGET_ENEMY,)))
            # 鲁班 skill3 probe：优先 target soldier/tower，其次 enemy
            if (
                context["skill3_cd_ratio"] <= 0.02
                and self._legal_button(observation, GameConfig.BUTTON_SKILL_3)
            ):
                available_probes.append(("luban_s3_probe", GameConfig.BUTTON_SKILL_3, (*GameConfig.TARGET_SOLDIERS, GameConfig.TARGET_TOWER, GameConfig.TARGET_ENEMY)))
        elif context["hero_id"] == 133:
            # 狄仁杰 skill2 probe：优先 target self/none（防守）
            if (
                context["skill2_cd_ratio"] <= 0.02
                and self._legal_button(observation, GameConfig.BUTTON_SKILL_2)
            ):
                available_probes.append(("direnjie_s2_probe", GameConfig.BUTTON_SKILL_2, (GameConfig.TARGET_SELF, GameConfig.TARGET_NONE)))
            # 狄仁杰 skill3 probe：优先 target enemy
            if (
                context["skill3_cd_ratio"] <= 0.02
                and self._legal_button(observation, GameConfig.BUTTON_SKILL_3)
                and context["enemy_visible"]
            ):
                available_probes.append(("direnjie_s3_probe", GameConfig.BUTTON_SKILL_3, (GameConfig.TARGET_ENEMY,)))
        
        # 从可用候选中随机选一个（每帧最多一次 probe）
        if available_probes:
            candidates.append(available_probes[np.random.randint(len(available_probes))])
        
        return candidates

    def _record_prior(self, prior_key, suffix):
        metric_key = {
            "luban_s2_finish": f"luban_s2_prior_{suffix}",
            "luban_s2_defense": f"luban_s2_prior_{suffix}",
            "luban_s2_probe": f"luban_s2_prior_{suffix}",
            "luban_s3": f"luban_s3_prior_{suffix}",
            "luban_s3_probe": f"luban_s3_prior_{suffix}",
            "direnjie_s2_defense": f"direnjie_s2_defense_prior_{suffix}",
            "direnjie_s2_probe": f"direnjie_s2_defense_prior_{suffix}",
            "direnjie_s3": f"direnjie_s3_prior_{suffix}",
            "direnjie_s3_probe": f"direnjie_s3_prior_{suffix}",
        }.get(prior_key)
        if metric_key in self.skill_prior_stats:
            self.skill_prior_stats[metric_key] += 1.0

    def _record_prior_block(self, prior_key, block_key):
        metric_key = f"prior_blocked_by_{block_key}"
        if metric_key in self.skill_prior_stats:
            self.skill_prior_stats[metric_key] += 1.0
        prefix = {
            "luban_s2_finish": "luban_s2",
            "luban_s2_defense": "luban_s2",
            "luban_s2_probe": "luban_s2",
            "luban_s3": "luban_s3",
            "luban_s3_probe": "luban_s3",
            "direnjie_s2_defense": "direnjie_s2",
            "direnjie_s2_probe": "direnjie_s2",
            "direnjie_s3": "direnjie_s3",
            "direnjie_s3_probe": "direnjie_s3",
        }.get(prior_key)
        block_suffix = "retreat" if block_key == "low_hp_retreat" else block_key
        detail_key = f"{prefix}_prior_block_{block_suffix}" if prefix else None
        if detail_key in self.skill_prior_stats:
            self.skill_prior_stats[detail_key] += 1.0

    def _collect_skill_diag_stats(self, observation, context):
        """收集诊断指标：每个英雄每个技能的 ready/legal/context 率
        
        重要：分母按当前英雄统计，避免双英雄训练时被稀释
        """
        if context["hero_id"] == 112:
            # 鲁班出场帧数
            self.skill_prior_stats["luban_frame_count"] += 1.0
            # 鲁班 skill2
            s2_ready = context["skill2_cd_ratio"] <= 0.02
            s2_legal = self._legal_button(observation, GameConfig.BUTTON_SKILL_2)
            s2_context = (
                context["enemy_visible"]
                and (context["enemy_low_hp"] or context["enemy_melee_close"])
            )
            if s2_ready:
                self.skill_prior_stats["luban_s2_ready"] += 1.0
            if s2_legal:
                self.skill_prior_stats["luban_s2_legal"] += 1.0
            if s2_context:
                self.skill_prior_stats["luban_s2_context"] += 1.0
            
            # 鲁班 skill3
            s3_ready = context["skill3_cd_ratio"] <= 0.02
            s3_legal = self._legal_button(observation, GameConfig.BUTTON_SKILL_3)
            s3_context = (
                context["push_window"]
                or context["enemy_soldier_cluster"]
                or context["has_minion_under_tower"]
                or context["enemy_near_enemy_tower"]
                or context.get("enemy_tower_low_hp", False)
            )
            if s3_ready:
                self.skill_prior_stats["luban_s3_ready"] += 1.0
            if s3_legal:
                self.skill_prior_stats["luban_s3_legal"] += 1.0
            if s3_context:
                self.skill_prior_stats["luban_s3_context"] += 1.0
        
        elif context["hero_id"] == 133:
            # 狄仁杰出场帧数
            self.skill_prior_stats["direnjie_frame_count"] += 1.0
            # 狄仁杰 skill2
            s2_ready = context["skill2_cd_ratio"] <= 0.02
            s2_legal = self._legal_button(observation, GameConfig.BUTTON_SKILL_2)
            s2_context = (
                context["self_hp_ratio"] < 0.55
                or context["recent_hurt_by_hero"] > 0.0
                or context["retreat_window"]
                or context["enemy_melee_close"]
            )
            if s2_ready:
                self.skill_prior_stats["direnjie_s2_ready"] += 1.0
            if s2_legal:
                self.skill_prior_stats["direnjie_s2_legal"] += 1.0
            if s2_context:
                self.skill_prior_stats["direnjie_s2_context"] += 1.0
            
            # 狄仁杰 skill3
            s3_ready = context["skill3_cd_ratio"] <= 0.02
            s3_legal = self._legal_button(observation, GameConfig.BUTTON_SKILL_3)
            s3_context = (
                context["enemy_visible"]
                and context["enemy_distance_suitable"]
                and context["self_hp_ratio"] > 0.45
                and not context["enemy_tower_danger"]
            )
            if s3_ready:
                self.skill_prior_stats["direnjie_s3_ready"] += 1.0
            if s3_legal:
                self.skill_prior_stats["direnjie_s3_legal"] += 1.0
            if s3_context:
                self.skill_prior_stats["direnjie_s3_context"] += 1.0

    def _luban_s3_targets(self, context):
        targets = [*GameConfig.TARGET_SOLDIERS, GameConfig.TARGET_TOWER]
        if context["enemy_low_hp"] and not context["enemy_tower_danger"]:
            targets.append(GameConfig.TARGET_ENEMY)
        return tuple(targets)

    def _prior_allowed(self, observation, context, prior_key, button, preferred_targets):
        if button in (GameConfig.BUTTON_SKILL_2, GameConfig.BUTTON_SKILL_3):
            if context["retreat_window"] and not prior_key.endswith("defense"):
                self._record_prior_block(prior_key, "low_hp_retreat")
                return False
            if context["self_hp_ratio"] <= 0.45 and not prior_key.endswith("defense"):
                self._record_prior_block(prior_key, "low_hp_retreat")
                return False
            if context["enemy_tower_danger"] and not context["has_minion_under_tower"]:
                self._record_prior_block(prior_key, "tower_risk")
                return False

        skill_no = button - GameConfig.BUTTON_SKILL_1 + 1
        if skill_no in (1, 2, 3) and context.get(f"skill{skill_no}_cd_ratio", 0.0) > 0.02:
            self._record_prior_block(prior_key, "cd")
            return False
        if not self._legal_button(observation, button):
            self._record_prior_block(prior_key, "mask")
            return False
        legal_splits = self._legal_action_splits(observation.get("legal_action", []))
        if not legal_splits:
            self._record_prior_block(prior_key, "mask")
            return False
        if self._first_legal_target(legal_splits[-1], button, preferred_targets) is None:
            self._record_prior_block(prior_key, "mask")
            return False
        return True

    def _skill_follow_action(self, observation, base_action, context):
        if self.luban_sweep_follow_frames > 0 and context["hero_id"] == 112:
            self.luban_sweep_follow_frames -= 1
            if np.random.random() >= getattr(GameConfig, "LUBAN_SWEEP_FOLLOW_ATTACK_PROB", 0.30):
                return None
            if context["enemy_visible"] and not context["enemy_tower_danger"]:
                return self._build_explore_action(observation, base_action, GameConfig.BUTTON_ATTACK, (GameConfig.TARGET_ENEMY,))
            if context["push_window"]:
                return self._build_explore_action(observation, base_action, GameConfig.BUTTON_ATTACK, (GameConfig.TARGET_TOWER,))
            if context["enemy_soldier_cluster"]:
                return self._build_explore_action(observation, base_action, GameConfig.BUTTON_ATTACK, GameConfig.TARGET_SOLDIERS)

        if self.skill_hit_follow_frames <= 0:
            return None
        self.skill_hit_follow_frames -= 1
        if not context["enemy_visible"] or context["enemy_tower_danger"]:
            return None
        if np.random.random() >= getattr(GameConfig, "SKILL_HIT_FOLLOW_ATTACK_PROB", 0.35):
            return None
        return self._build_explore_action(observation, base_action, GameConfig.BUTTON_ATTACK, (GameConfig.TARGET_ENEMY,))

    def _build_explore_action(self, observation, base_action, button, preferred_targets):
        legal_splits = self._legal_action_splits(observation.get("legal_action", []))
        if not legal_splits or not self._legal_head_value(legal_splits[0], button):
            return None
        sub_mask = observation.get("sub_action_mask", {}) or {}
        button_mask = sub_mask.get(str(button))
        if button_mask is None:
            return None
        if hasattr(button_mask, "tolist"):
            button_mask = button_mask.tolist()
        if not isinstance(button_mask, (list, tuple)) or len(button_mask) < len(self.label_size_list):
            return None

        action = list(base_action)
        action[0] = button
        target = self._first_legal_target(legal_splits[-1], button, preferred_targets)
        if target is None:
            return None
        action[-1] = target

        for head in range(1, len(self.label_size_list) - 1):
            if float(button_mask[head] or 0.0) <= 0.5:
                continue
            if self._is_skill_offset_head(button, head):
                replacement = self._center_or_first_legal(legal_splits[head])
                if replacement is None:
                    return None
                action[head] = replacement
                continue
            if self._legal_head_value(legal_splits[head], action[head]):
                continue
            replacement = self._first_legal_head_value(legal_splits[head])
            if replacement is None:
                return None
            action[head] = replacement

        if float(button_mask[-1] or 0.0) > 0.5 and not self._target_legal(legal_splits[-1], button, action[-1]):
            return None
        return action

    def _legal_action_splits(self, legal_action):
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if not isinstance(legal_action, (list, tuple)):
            return None
        label_split_size = [sum(self.legal_action_size[: index + 1]) for index in range(len(self.legal_action_size))]
        if len(legal_action) < label_split_size[-1]:
            return None
        return np.split(np.array(legal_action[: label_split_size[-1]]), label_split_size[:-1])

    def _legal_head_value(self, legal_head, value):
        try:
            value = int(value)
            return 0 <= value < len(legal_head) and float(legal_head[value]) > 0.5
        except (TypeError, ValueError):
            return False

    def _first_legal_head_value(self, legal_head):
        legal_indices = np.where(np.asarray(legal_head) > 0.5)[0]
        if len(legal_indices) == 0:
            return None
        return int(legal_indices[0])

    def _center_or_first_legal(self, legal_head):
        for preferred in getattr(GameConfig, "SKILL_OFFSET_PRIORITIES", (8, 7, 9, 6, 10)):
            if self._legal_head_value(legal_head, preferred):
                return int(preferred)
        return self._first_legal_head_value(legal_head)

    def _legal_button(self, observation, button):
        legal_action = observation.get("legal_action", [])
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if not isinstance(legal_action, (list, tuple)) or len(legal_action) <= button:
            return False
        try:
            return float(legal_action[int(button)]) > 0.5
        except (TypeError, ValueError, IndexError):
            return False

    def _is_skill_offset_head(self, button, head):
        return button in (GameConfig.BUTTON_SKILL_1, GameConfig.BUTTON_SKILL_2, GameConfig.BUTTON_SKILL_3) and head in (3, 4)

    def _target_legal(self, target_legal_flat, button, target):
        try:
            target_matrix = np.asarray(target_legal_flat).reshape(
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            )
            return float(target_matrix[int(button)][int(target)]) > 0.5
        except (TypeError, ValueError, IndexError):
            return False

    def _first_legal_target(self, target_legal_flat, button, preferred_targets):
        for target in preferred_targets:
            if self._target_legal(target_legal_flat, button, target):
                return int(target)
        try:
            target_matrix = np.asarray(target_legal_flat).reshape(
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            )
            legal_targets = np.where(target_matrix[int(button)] > 0.5)[0]
        except (TypeError, ValueError, IndexError):
            return None
        if len(legal_targets) == 0:
            return None
        return int(legal_targets[0])

    def _skill_exploration_context(self, observation):
        frame_state = observation.get("frame_state", {})
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_hero = self._get_enemy_hero(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)
        my_soldiers = self._soldiers(frame_state, camp)
        enemy_soldiers = self._enemy_soldiers(frame_state, camp)
        if not main_hero:
            return None

        self_hp_ratio = self._hp_ratio(main_hero)
        enemy_hp_ratio = self._hp_ratio(enemy_hero)
        enemy_visible = enemy_hero is not None and enemy_hp_ratio > 0.0
        enemy_distance = self._distance(self._position(main_hero), self._position(enemy_hero)) if enemy_visible else 999999.0
        skill_details = self._skill_details(main_hero)
        for skill_no in (1, 2, 3):
            last_hit_total = self.last_skill_hit_totals.get(skill_no)
            hit_total = skill_details[skill_no]["hit_total"]
            if last_hit_total is not None and hit_total > last_hit_total:
                self.skill_hit_follow_frames = max(self.skill_hit_follow_frames, 10)
                if self._int_field(main_hero, "config_id", 0) == 112 and skill_no == 1:
                    self.luban_sweep_follow_frames = max(self.luban_sweep_follow_frames, 10)
            self.last_skill_hit_totals[skill_no] = hit_total
        hurt_by_hero = self._value(main_hero, "total_be_hurt_by_hero")
        recent_hurt_by_hero = 0.0
        if self.last_total_be_hurt_by_hero is not None:
            recent_hurt_by_hero = max(0.0, hurt_by_hero - self.last_total_be_hurt_by_hero)
        self.last_total_be_hurt_by_hero = hurt_by_hero

        enemy_tower_distance = self._distance(self._position(main_hero), self._position(enemy_tower)) if enemy_tower else 999999.0
        enemy_to_enemy_tower_distance = self._distance(self._position(enemy_hero), self._position(enemy_tower)) if enemy_visible and enemy_tower else 999999.0
        attack_range = self._value(main_hero, "attack_range") or 7000.0
        has_minion_under_tower = self._has_ally_minion_under_enemy_tower(my_soldiers, enemy_tower)
        in_enemy_tower_range = self._in_tower_range(main_hero, enemy_tower)
        enemy_tower_danger = in_enemy_tower_range and not has_minion_under_tower and enemy_hp_ratio > 0.25
        retreat_window = self_hp_ratio < 0.30 or (enemy_visible and self_hp_ratio - enemy_hp_ratio < -0.25 and enemy_distance <= 12000.0)
        skill_cd_ratios = self._skill_cd_ratios(main_hero)
        push_window = (
            enemy_tower is not None
            and self._value(enemy_tower, "hp") > 0.0
            and self_hp_ratio > 0.35
            and (has_minion_under_tower or enemy_tower_distance <= attack_range + 1200.0)
            and not enemy_tower_danger
        )

        # 敌塔低血量判断
        enemy_tower_hp = self._value(enemy_tower, "hp") if enemy_tower else 0.0
        enemy_tower_max_hp = self._value(enemy_tower, "max_hp") if enemy_tower else 1.0
        enemy_tower_hp_ratio = enemy_tower_hp / max(enemy_tower_max_hp, 1.0) if enemy_tower_max_hp > 0 else 0.0
        enemy_tower_low_hp = enemy_tower_hp_ratio < 0.50 and enemy_tower_hp_ratio > 0.0

        return {
            "hero_id": self._int_field(main_hero, "config_id", 0),
            "self_hp_ratio": self_hp_ratio,
            "enemy_hp_ratio": enemy_hp_ratio,
            "enemy_visible": enemy_visible,
            "enemy_close": enemy_visible and enemy_distance <= 12000.0,
            "enemy_mid_close": enemy_visible and 3500.0 <= enemy_distance <= 12000.0,
            "enemy_melee_close": enemy_visible and enemy_distance <= 6500.0,
            "enemy_distance_suitable": 4500.0 <= enemy_distance <= 12500.0,
            "enemy_far": enemy_visible and enemy_distance >= attack_range + 1800.0,
            "enemy_low_hp": enemy_visible and enemy_hp_ratio <= 0.35,
            "enemy_retreat_like": enemy_visible and enemy_hp_ratio < self_hp_ratio - 0.20,
            "self_hp_advantage": self_hp_ratio - enemy_hp_ratio,
            "recent_hurt_by_hero": recent_hurt_by_hero,
            "enemy_tower_danger": enemy_tower_danger,
            "retreat_window": retreat_window,
            "has_minion_under_tower": has_minion_under_tower,
            "push_window": push_window,
            "enemy_near_enemy_tower": enemy_to_enemy_tower_distance <= (self._value(enemy_tower, "attack_range") or 9000.0),
            "enemy_soldier_cluster": len(enemy_soldiers) >= 3,
            "enemy_tower_low_hp": enemy_tower_low_hp,
            "skill1_cd_ratio": skill_cd_ratios[1],
            "skill2_cd_ratio": skill_cd_ratios[2],
            "skill3_cd_ratio": skill_cd_ratios[3],
        }

    def _skill_details(self, hero):
        details = {
            1: {"used_total": 0.0, "hit_total": 0.0},
            2: {"used_total": 0.0, "hit_total": 0.0},
            3: {"used_total": 0.0, "hit_total": 0.0},
        }
        skill_state = hero.get("skill_state", {}) if isinstance(hero, dict) else {}
        slots = (
            skill_state.get("slot_states")
            or skill_state.get("skill_slot")
            or skill_state.get("skill_slot_state")
            or skill_state.get("slots")
            or []
        )
        for index, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue
            skill_no = self._skill_no_from_slot(slot, index)
            if skill_no not in details:
                continue
            details[skill_no]["used_total"] += self._value_any(
                slot, ("succUsedInFrame", "succ_used_in_frame", "usedTimes", "used_times")
            )
            details[skill_no]["hit_total"] += self._value_any(slot, ("hitHeroTimes", "hit_hero_times"))
        return details

    def _skill_no_from_slot(self, slot, index):
        slot_type = self._int_field(slot, "slotType", self._int_field(slot, "slot_type", -1))
        if slot_type in (1, 2, 3):
            return slot_type
        config_id = self._int_field(
            slot,
            "skillID",
            self._int_field(slot, "skill_id", self._int_field(slot, "configId", self._int_field(slot, "config_id", 0))),
        )
        skill_no = GameConfig.skill_no_from_id(config_id)
        if skill_no is not None:
            return skill_no
        if index in (0, 1, 2):
            return index + 1
        return None

    def _skill_cd_ratios(self, hero):
        ratios = {1: 0.0, 2: 0.0, 3: 0.0}
        skill_state = hero.get("skill_state", {}) if isinstance(hero, dict) else {}
        slots = (
            skill_state.get("slot_states")
            or skill_state.get("skill_slot")
            or skill_state.get("skill_slot_state")
            or skill_state.get("slots")
            or []
        )
        for index, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue
            skill_no = self._skill_no_from_slot(slot, index)
            if skill_no not in ratios:
                continue
            cooldown = self._value_any(slot, ("cooldown", "cd", "coolDown", "cooldownMs"))
            max_cooldown = self._value_any(slot, ("cooldown_max", "maxCooldown", "max_cooldown", "cooldownMax"))
            ratios[skill_no] = max(0.0, min(1.0, cooldown / max_cooldown)) if max_cooldown > 0 else 0.0
        return ratios

    def _value_any(self, unit, keys):
        for key in keys:
            value = self._value(unit, key)
            if value:
                return value
        return 0.0

    def _get_hero(self, frame_state, camp):
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == camp:
                return hero
        return None

    def _get_enemy_hero(self, frame_state, camp):
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") != camp:
                return hero
        return None

    def _get_enemy_tower(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") != camp and int(self._value(npc, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return npc
        return None

    def _soldiers(self, frame_state, camp):
        return [
            npc
            for npc in frame_state.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") == camp
            and not self._is_organ(npc)
            and self._is_lane_soldier(npc)
            and self._value(npc, "hp") > 0
        ]

    def _enemy_soldiers(self, frame_state, camp):
        return [
            npc
            for npc in frame_state.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") != camp
            and not self._is_organ(npc)
            and self._is_lane_soldier(npc)
            and self._value(npc, "hp") > 0
        ]

    def _is_organ(self, unit):
        sub_type = int(self._value(unit, "sub_type"))
        return sub_type in GameConfig.TOWER_SUB_TYPES or sub_type in GameConfig.CRYSTAL_SUB_TYPES

    def _is_lane_soldier(self, unit):
        sub_type = self._int_field(unit, "sub_type", -1)
        config_id = self._int_field(unit, "config_id", 0)
        return sub_type in GameConfig.SOLDIER_SUB_TYPES or config_id in GameConfig.LANE_SOLDIER_CONFIG_IDS

    def _hp_ratio(self, unit):
        max_hp = self._value(unit, "max_hp")
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, self._value(unit, "hp") / max_hp))

    def _in_tower_range(self, hero, tower):
        if not hero or not tower:
            return False
        tower_range = self._value(tower, "attack_range") or 9000.0
        return self._distance(self._position(hero), self._position(tower)) <= tower_range

    def _has_ally_minion_under_enemy_tower(self, soldiers, enemy_tower):
        if not enemy_tower:
            return False
        tower_range = self._value(enemy_tower, "attack_range") or 9000.0
        tower_pos = self._position(enemy_tower)
        return any(self._distance(self._position(soldier), tower_pos) <= tower_range for soldier in soldiers)

    def _position(self, unit):
        location = unit.get("location", {}) if isinstance(unit, dict) else {}
        return (
            self._value(location, "x"),
            self._value(location, "z"),
        )

    def _distance(self, p1, p2):
        return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)

    def _int_field(self, data, key, default=0):
        if not data or data.get(key) is None:
            return default
        try:
            return int(data.get(key))
        except (TypeError, ValueError):
            return default

    def _value(self, unit, key):
        if not isinstance(unit, dict):
            return 0.0
        try:
            return float(unit.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            if not os.path.exists(model_file_path):
                self.logger.warning(f"model {model_file_path} not found, use fresh initialized model")
                return
            state_dict = None
            try:
                state_dict = torch.load(model_file_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.cur_model_name = model_file_path
                self.logger.info(f"load model {model_file_path} successfully")
            except RuntimeError as e:
                self.logger.warning(f"model {model_file_path} is partially incompatible, try partial load: {e}")
                if state_dict is None:
                    return
                self._load_model_partial(model_file_path, state_dict)

    def _load_model_partial(self, model_file_path, state_dict):
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        model_state = self.model.state_dict()
        load_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key in model_state and hasattr(value, "shape") and model_state[key].shape == value.shape:
                load_state[key] = value
            else:
                skipped_keys.append(key)
        model_state.update(load_state)
        self.model.load_state_dict(model_state)
        self.cur_model_name = model_file_path
        self.logger.info(
            f"partial load model {model_file_path}: loaded {len(load_state)} / {len(model_state)}, "
            f"skipped {len(skipped_keys)}"
        )
        if skipped_keys:
            self.logger.info(f"partial load skipped keys: {skipped_keys[:20]}")

    def load_opponent_agent(self, id="1"):
        # Framework provides loading opponent agent function, no need to implement function content
        # 框架提供的加载对手模型功能，无需实现函数内容
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action, stochastic=True):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional list
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=not stochastic)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=not stochastic)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
