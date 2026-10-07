#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import tomllib
import time
import numpy as np
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
    lineup_iterator_roundrobin_lineup_pairs,
)
from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.buff_constants import (
    BERSERK_BUFF_IDS,
    LUBAN_BUFF_IDS,
    collect_buff_config_ids,
    has_luban_any_buff,
    has_luban_output_buff,
)
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


class OpponentSchedule:
    def __init__(self, config_path, logger=None):
        self.logger = logger
        self.train_pool = []
        self.eval_pool = []
        self.train_index = 0
        self.eval_index = 0
        self._load(config_path)

    def _load(self, config_path):
        try:
            with open(config_path, "rb") as file:
                episode_conf = (tomllib.load(file).get("episode") or {})
        except OSError as exc:
            if self.logger:
                self.logger.warning(f"opponent schedule config load failed: {exc}")
            return
        self.train_pool = self._normalize_pool(episode_conf.get("train_opponent_pool"))
        self.eval_pool = self._normalize_pool(episode_conf.get("eval_opponent_pool"))

    def _normalize_pool(self, value):
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [str(item) for item in value if str(item)]

    def select(self, default_opponent, is_eval):
        pool = self.eval_pool if is_eval else self.train_pool
        if not pool:
            return str(default_opponent)
        if is_eval:
            opponent = pool[self.eval_index % len(pool)]
            self.eval_index += 1
        else:
            opponent = pool[self.train_index % len(pool)]
            self.train_index += 1
        return opponent


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_path = "agent_ppo/conf/train_env_conf.toml"
    env_conf_manager = EnvConfManager(
        config_path=env_conf_path,
        logger=logger,
    )
    opponent_schedule = OpponentSchedule(env_conf_path, logger=logger)

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    # Direnjie C0 trains 133 vs 133 while keeping 112/133 feature dimensions unchanged.
    if getattr(GameConfig, "TRAIN_LINEUP_PAIRS", None):
        lineup_iterator = lineup_iterator_roundrobin_lineup_pairs(GameConfig.TRAIN_LINEUP_PAIRS)
    else:
        lineup_iterator = lineup_iterator_roundrobin_camp_heroes(GameConfig.TRAIN_HERO_IDS)

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
        opponent_schedule=opponent_schedule,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    REWARD_MONITOR_KEYS = (
        "reward_hp",
        "reward_tower",
        "reward_gold",
        "reward_exp",
        "reward_last_hit",
        "reward_death",
        "reward_kill",
        "reward_forward",
        "reward_win",
        "reward_tower_danger",
        "reward_push_assist",
        "reward_miss_push_window",
        "reward_lane_follow",
        "reward_finish",
        "reward_hero_damage",
        "reward_combat_kite",
        "reward_combat_standstill",
        "reward_attack_move",
        "reward_attack_standstill",
        "reward_low_hp_cake",
        "reward_low_hp_heal",
        "reward_low_hp_idle",
        "reward_follow_minion_after_clear",
        "reward_idle_after_clear",
        "reward_river_monster",
        "reward_idle_when_monster",
        "reward_attack_hero_in_range",
        "reward_miss_hero_in_range",
        "reward_skill_hit_hero",
        "reward_skill_damage",
        "reward_skill2_ready_unused",
        "reward_skill3_ready_unused",
        "reward_luban_buff_damage",
        "reward_skill2_finish",
        "reward_summoner_finish",
        "reward_attack_hero_under_tower",
        "reward_retreat_from_tower",
        "reward_stay_under_tower",
        "reward_base_stuck",
        "reward_early_forward",
        "reward_skill_follow_attack",
        "reward_skill_follow_tower",
        "reward_bad_skill2",
        "reward_bad_skill3",
        "reward_bad_summoner",
        "reward_push_window_target_tower",
        "reward_push_window_wrong_hero",
        "reward_retreat_to_own_tower",
        "reward_bad_attack_low_hp",
        "reward_safe_recall",
        "reward_skill1_good_cast",
        "reward_skill2_good_cast",
        "reward_skill3_good_cast",
        "reward_skill_after_attack",
        "reward_berserk_attack_follow",
        "reward_tower_poke_hero",
        "reward_tower_poke_dive",
        "reward_defend_tower_clear",
        "reward_defend_tower_ignore",
    )

    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator, opponent_schedule):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.opponent_schedule = opponent_schedule
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            opponent_agent = self.opponent_schedule.select(self.env_conf_manager.get_opponent_agent(), is_eval)
            self.logger.info(f"Episode opponent selected: opponent_agent={opponent_agent}, eval={is_eval}")

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            # Reset agents
            # 重置智能体
            self.reset_agents(observation, opponent_agent)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            episode_stats = [self._new_episode_stats() for _ in range(self.agent_num)]
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    self._update_reward_item_stats(episode_stats[i], reward)

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                    self._update_episode_stats(episode_stats[index], observation[str(index)], actions[index])

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"], actions[i])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._update_reward_item_stats(episode_stats[i], reward)
                        self._update_episode_outcome_stats(episode_stats[i], observation[str(i)])

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = self._build_monitor_data(
                            observation[str(monitor_side)],
                            reward_sum_list[monitor_side],
                            actions[monitor_side],
                            is_eval,
                            self.episode_cnt,
                            episode_stats[monitor_side],
                            self.agents[monitor_side],
                            opponent_agent,
                        )
                        if self.monitor:
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def _build_monitor_data(
        self,
        observation,
        reward_sum,
        action,
        is_eval,
        episode_cnt,
        episode_stats,
        monitor_agent=None,
        opponent_agent=None,
    ):
        frame_state = observation["frame_state"]
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_hero = self._get_enemy_hero(frame_state, camp)
        main_tower = self._get_tower(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)
        reward_items = episode_stats.get("reward_item_sum", {})
        main_config = int(self._value(main_hero, "config_id"))
        enemy_config = int(self._value(enemy_hero, "config_id"))
        frame_no = max(float(frame_state.get("frame_no", 0.0) or 0.0), 1.0)
        frames = max(float(episode_stats.get("frames", 0.0)), 1.0)
        attention_stats = self._attention_stats_from_agent(monitor_agent)
        skill_prior_stats = self._skill_prior_stats_from_agent(monitor_agent)
        self_hp_ratio = self._hp_ratio(main_hero)
        enemy_hp_ratio = self._hp_ratio(enemy_hero)
        self_tower_hp_ratio = self._hp_ratio(main_tower)
        enemy_tower_hp_ratio = self._hp_ratio(enemy_tower)
        self_money = self._resource(main_hero, "money")
        enemy_money = self._resource(enemy_hero, "money")

        enemy_tower_dead = enemy_tower is not None and self._value(enemy_tower, "hp") <= 0
        main_tower_dead = main_tower is not None and self._value(main_tower, "hp") <= 0
        if enemy_tower_dead and not main_tower_dead:
            win_rate = 1.0
        elif main_tower_dead and not enemy_tower_dead:
            win_rate = 0.0
        else:
            win_rate = 0.5 if is_eval else 0.0

        monitor_data = {
            "episode_cnt": episode_cnt,
            "frame": round(frame_no, 2),
            "reward": round(float(reward_sum), 2),
            "episode_win": win_rate,
            "attention_enabled": 1.0 if GameConfig.USE_TARGET_ATTENTION else 0.0,
            "attention_alpha": round(float(GameConfig.ATTENTION_ALPHA), 4),
            "attention_logit_scale": round(float(GameConfig.ATTENTION_LOGIT_SCALE), 4),
            "attention_entropy": round(attention_stats["attention_entropy"], 4),
            "attention_max_prob": round(attention_stats["attention_max_prob"], 4),
            "attention_target_hero_prob": round(attention_stats["attention_target_hero_prob"], 4),
            "attention_target_soldier_prob": round(attention_stats["attention_target_soldier_prob"], 4),
            "attention_target_tower_prob": round(attention_stats["attention_target_tower_prob"], 4),
            "target_changed_by_attention_rate": round(attention_stats["target_changed_by_attention_rate"], 4),
            "controlled_hero_id": main_config,
            "opponent_hero_id": enemy_config,
            "self_tower_hp": round(self._value(main_tower, "hp"), 2),
            "enemy_tower_hp": round(self._value(enemy_tower, "hp"), 2),
            "self_tower_hp_ratio": round(self_tower_hp_ratio, 4),
            "enemy_tower_hp_ratio": round(enemy_tower_hp_ratio, 4),
            "tower_hp_advantage": round(self_tower_hp_ratio - enemy_tower_hp_ratio, 4),
            "self_hp_ratio": round(self_hp_ratio, 4),
            "enemy_hp_ratio": round(enemy_hp_ratio, 4),
            "hp_advantage": round(self_hp_ratio - enemy_hp_ratio, 4),
            "self_money": round(self_money, 2),
            "enemy_money": round(enemy_money, 2),
            "gold_advantage": round(self_money - enemy_money, 2),
            "money_per_frame": round(self._resource(main_hero, "money") / frame_no, 4),
            "kill": 1.0 if enemy_hero and self._value(enemy_hero, "hp") <= 0 else 0.0,
            "death": 1.0 if main_hero and self._value(main_hero, "hp") <= 0 else 0.0,
            "lane_progress": round(self._lane_progress(main_hero, main_tower, enemy_tower), 4),
            "my_soldier_count": len(self._soldiers(frame_state, camp)),
            "enemy_soldier_count": len(self._enemy_soldiers(frame_state, camp)),
            "move_action_rate": round(episode_stats["move_action"] / frames, 4),
            "attack_action_rate": round(episode_stats["attack_action"] / frames, 4),
            "recall_action_rate": round(episode_stats["recall_action"] / frames, 4),
            "hero_target_rate": round(episode_stats["hero_target"] / frames, 4),
            "creep_target_rate": round(episode_stats["creep_target"] / frames, 4),
            "tower_target_rate": round(episode_stats["tower_target"] / frames, 4),
            "enemy_tower_range_rate": round(episode_stats["enemy_tower_range"] / frames, 4),
            "no_minion_tower_dive_rate": round(episode_stats["no_minion_tower_dive"] / frames, 4),
            "safe_push_window_rate": round(episode_stats["safe_push_window"] / frames, 4),
            "strict_push_window_rate": round(episode_stats["strict_push_window"] / frames, 4),
            "near_push_window_rate": round(episode_stats["near_push_window"] / frames, 4),
            "push_window_rate": round(episode_stats["push_window"] / frames, 4),
            "push_window_target_tower_rate": round(episode_stats["push_window_target_tower"] / frames, 4),
            "push_window_wrong_hero_rate": round(episode_stats["push_window_wrong_hero"] / frames, 4),
            "retreat_window_rate": round(episode_stats["retreat_window"] / frames, 4),
            "retreat_to_own_tower_rate": round(episode_stats["retreat_to_own_tower"] / frames, 4),
            "retreat_success_rate": round(episode_stats["retreat_to_own_tower"] / frames, 4),
            "bad_attack_low_hp_rate": round(episode_stats["bad_attack_low_hp"] / frames, 4),
            "safe_recall_rate": round(episode_stats["safe_recall"] / frames, 4),
            "low_hp_death_rate": round(episode_stats["low_hp_death"] / frames, 4),
            "tower_target_with_minion_rate": round(episode_stats["tower_target_with_minion"] / frames, 4),
            "attack_tower_with_minion_rate": round(episode_stats["attack_tower_with_minion"] / frames, 4),
            "safe_attack_tower_rate": round(episode_stats["safe_attack_tower"] / frames, 4),
            "tower_attack_with_minion_rate": round(episode_stats["attack_tower_with_minion"] / frames, 4),
            "reward_hp": round(reward_items.get("reward_hp", 0.0), 4),
            "reward_tower": round(reward_items.get("reward_tower", 0.0), 4),
            "reward_gold": round(reward_items.get("reward_gold", 0.0), 4),
            "reward_exp": round(reward_items.get("reward_exp", 0.0), 4),
            "reward_last_hit": round(reward_items.get("reward_last_hit", 0.0), 4),
            "reward_death": round(reward_items.get("reward_death", 0.0), 4),
            "reward_kill": round(reward_items.get("reward_kill", 0.0), 4),
            "reward_forward": round(reward_items.get("reward_forward", 0.0), 4),
            "reward_win": round(reward_items.get("reward_win", 0.0), 4),
            "reward_tower_danger": round(reward_items.get("reward_tower_danger", 0.0), 4),
            "reward_push_assist": round(reward_items.get("reward_push_assist", 0.0), 4),
            "reward_miss_push_window": round(reward_items.get("reward_miss_push_window", 0.0), 4),
            "reward_lane_follow": round(reward_items.get("reward_lane_follow", 0.0), 4),
            "reward_finish": round(reward_items.get("reward_finish", 0.0), 4),
            "reward_hero_damage": round(reward_items.get("reward_hero_damage", 0.0), 4),
            "reward_combat_kite": round(reward_items.get("reward_combat_kite", 0.0), 4),
            "reward_combat_standstill": round(reward_items.get("reward_combat_standstill", 0.0), 4),
            "reward_attack_move": round(reward_items.get("reward_attack_move", 0.0), 4),
            "reward_attack_standstill": round(reward_items.get("reward_attack_standstill", 0.0), 4),
            "reward_low_hp_cake": round(reward_items.get("reward_low_hp_cake", 0.0), 4),
            "reward_low_hp_heal": round(reward_items.get("reward_low_hp_heal", 0.0), 4),
            "reward_low_hp_idle": round(reward_items.get("reward_low_hp_idle", 0.0), 4),
            "reward_follow_minion_after_clear": round(reward_items.get("reward_follow_minion_after_clear", 0.0), 4),
            "reward_idle_after_clear": round(reward_items.get("reward_idle_after_clear", 0.0), 4),
            "reward_river_monster": round(reward_items.get("reward_river_monster", 0.0), 4),
            "reward_idle_when_monster": round(reward_items.get("reward_idle_when_monster", 0.0), 4),
            "reward_attack_hero_in_range": round(reward_items.get("reward_attack_hero_in_range", 0.0), 4),
            "reward_miss_hero_in_range": round(reward_items.get("reward_miss_hero_in_range", 0.0), 4),
            "reward_skill_hit_hero": round(reward_items.get("reward_skill_hit_hero", 0.0), 4),
            "reward_skill_damage": round(reward_items.get("reward_skill_damage", 0.0), 4),
            "reward_skill2_ready_unused": round(reward_items.get("reward_skill2_ready_unused", 0.0), 4),
            "reward_skill3_ready_unused": round(reward_items.get("reward_skill3_ready_unused", 0.0), 4),
            "reward_luban_buff_damage": round(reward_items.get("reward_luban_buff_damage", 0.0), 4),
            "reward_skill2_finish": round(reward_items.get("reward_skill2_finish", 0.0), 4),
            "reward_summoner_finish": round(reward_items.get("reward_summoner_finish", 0.0), 4),
            "reward_attack_hero_under_tower": round(reward_items.get("reward_attack_hero_under_tower", 0.0), 4),
            "reward_retreat_from_tower": round(reward_items.get("reward_retreat_from_tower", 0.0), 4),
            "reward_stay_under_tower": round(reward_items.get("reward_stay_under_tower", 0.0), 4),
            "reward_base_stuck": round(reward_items.get("reward_base_stuck", 0.0), 4),
            "reward_early_forward": round(reward_items.get("reward_early_forward", 0.0), 4),
            "reward_skill_follow_attack": round(reward_items.get("reward_skill_follow_attack", 0.0), 4),
            "reward_skill_follow_tower": round(reward_items.get("reward_skill_follow_tower", 0.0), 4),
            "reward_bad_skill2": round(reward_items.get("reward_bad_skill2", 0.0), 4),
            "reward_bad_skill3": round(reward_items.get("reward_bad_skill3", 0.0), 4),
            "reward_bad_summoner": round(reward_items.get("reward_bad_summoner", 0.0), 4),
            "reward_push_window_target_tower": round(reward_items.get("reward_push_window_target_tower", 0.0), 4),
            "reward_push_window_wrong_hero": round(reward_items.get("reward_push_window_wrong_hero", 0.0), 4),
            "reward_retreat_to_own_tower": round(reward_items.get("reward_retreat_to_own_tower", 0.0), 4),
            "reward_bad_attack_low_hp": round(reward_items.get("reward_bad_attack_low_hp", 0.0), 4),
            "reward_safe_recall": round(reward_items.get("reward_safe_recall", 0.0), 4),
            "reward_skill1_good_cast": round(reward_items.get("reward_skill1_good_cast", 0.0), 4),
            "reward_skill2_good_cast": round(reward_items.get("reward_skill2_good_cast", 0.0), 4),
            "reward_skill3_good_cast": round(reward_items.get("reward_skill3_good_cast", 0.0), 4),
            "reward_skill_after_attack": round(reward_items.get("reward_skill_after_attack", 0.0), 4),
            "reward_berserk_attack_follow": round(reward_items.get("reward_berserk_attack_follow", 0.0), 4),
            "reward_tower_poke_hero": round(reward_items.get("reward_tower_poke_hero", 0.0), 4),
            "reward_tower_poke_dive": round(reward_items.get("reward_tower_poke_dive", 0.0), 4),
            "reward_defend_tower_clear": round(reward_items.get("reward_defend_tower_clear", 0.0), 4),
            "reward_defend_tower_ignore": round(reward_items.get("reward_defend_tower_ignore", 0.0), 4),
            "reward_stage_a": 1.0 if reward_items.get("reward_stage") == "A" else 0.0,
            "reward_stage_cmin": 1.0 if GameConfig.REWARD_STAGE == "Cmin" else 0.0,
            "reward_stage_cplus": 1.0 if GameConfig.REWARD_STAGE == "Cplus" else 0.0,
            "reward_stage_d": 1.0 if GameConfig.REWARD_STAGE == "D" else 0.0,
            "reward_push_assist_trigger_rate": round(episode_stats["reward_push_assist_trigger"] / frames, 4),
            "reward_miss_push_window_trigger_rate": round(episode_stats["reward_miss_push_window_trigger"] / frames, 4),
            "reward_lane_follow_trigger_rate": round(episode_stats["reward_lane_follow_trigger"] / frames, 4),
            "reward_finish_trigger_rate": round(episode_stats["reward_finish_trigger"] / frames, 4),
            "reward_hero_damage_trigger_rate": round(episode_stats["reward_hero_damage_trigger"] / frames, 4),
            "reward_combat_kite_trigger_rate": round(episode_stats["reward_combat_kite_trigger"] / frames, 4),
            "reward_combat_standstill_trigger_rate": round(episode_stats["reward_combat_standstill_trigger"] / frames, 4),
            "reward_attack_move_trigger_rate": round(episode_stats["reward_attack_move_trigger"] / frames, 4),
            "reward_attack_standstill_trigger_rate": round(episode_stats["reward_attack_standstill_trigger"] / frames, 4),
            "reward_low_hp_cake_trigger_rate": round(episode_stats["reward_low_hp_cake_trigger"] / frames, 4),
            "reward_low_hp_heal_trigger_rate": round(episode_stats["reward_low_hp_heal_trigger"] / frames, 4),
            "reward_low_hp_idle_trigger_rate": round(episode_stats["reward_low_hp_idle_trigger"] / frames, 4),
            "reward_follow_minion_after_clear_trigger_rate": round(episode_stats["reward_follow_minion_after_clear_trigger"] / frames, 4),
            "reward_idle_after_clear_trigger_rate": round(episode_stats["reward_idle_after_clear_trigger"] / frames, 4),
            "reward_river_monster_trigger_rate": round(episode_stats["reward_river_monster_trigger"] / frames, 4),
            "reward_idle_when_monster_trigger_rate": round(episode_stats["reward_idle_when_monster_trigger"] / frames, 4),
            "reward_attack_hero_in_range_trigger_rate": round(episode_stats["reward_attack_hero_in_range_trigger"] / frames, 4),
            "reward_miss_hero_in_range_trigger_rate": round(episode_stats["reward_miss_hero_in_range_trigger"] / frames, 4),
            "reward_skill_hit_hero_trigger_rate": round(episode_stats["reward_skill_hit_hero_trigger"] / frames, 4),
            "reward_skill2_ready_unused_trigger_rate": round(episode_stats["reward_skill2_ready_unused_trigger"] / frames, 4),
            "reward_skill3_ready_unused_trigger_rate": round(episode_stats["reward_skill3_ready_unused_trigger"] / frames, 4),
            "reward_luban_buff_damage_trigger_rate": round(episode_stats["reward_luban_buff_damage_trigger"] / frames, 4),
            "reward_skill2_finish_trigger_rate": round(episode_stats["reward_skill2_finish_trigger"] / frames, 4),
            "reward_summoner_finish_trigger_rate": round(episode_stats["reward_summoner_finish_trigger"] / frames, 4),
            "reward_base_stuck_trigger_rate": round(episode_stats["reward_base_stuck_trigger"] / frames, 4),
            "reward_early_forward_trigger_rate": round(episode_stats["reward_early_forward_trigger"] / frames, 4),
            "reward_skill_follow_attack_trigger_rate": round(episode_stats["reward_skill_follow_attack_trigger"] / frames, 4),
            "reward_skill_follow_tower_trigger_rate": round(episode_stats["reward_skill_follow_tower_trigger"] / frames, 4),
            "reward_bad_skill2_trigger_rate": round(episode_stats["reward_bad_skill2_trigger"] / frames, 4),
            "reward_bad_skill3_trigger_rate": round(episode_stats["reward_bad_skill3_trigger"] / frames, 4),
            "reward_bad_summoner_trigger_rate": round(episode_stats["reward_bad_summoner_trigger"] / frames, 4),
            "reward_push_window_target_tower_trigger_rate": round(episode_stats["reward_push_window_target_tower_trigger"] / frames, 4),
            "reward_push_window_wrong_hero_trigger_rate": round(episode_stats["reward_push_window_wrong_hero_trigger"] / frames, 4),
            "reward_retreat_to_own_tower_trigger_rate": round(episode_stats["reward_retreat_to_own_tower_trigger"] / frames, 4),
            "reward_bad_attack_low_hp_trigger_rate": round(episode_stats["reward_bad_attack_low_hp_trigger"] / frames, 4),
            "reward_safe_recall_trigger_rate": round(episode_stats["reward_safe_recall_trigger"] / frames, 4),
            "reward_skill1_good_cast_trigger_rate": round(episode_stats["reward_skill1_good_cast_trigger"] / frames, 4),
            "reward_skill2_good_cast_trigger_rate": round(episode_stats["reward_skill2_good_cast_trigger"] / frames, 4),
            "reward_skill3_good_cast_trigger_rate": round(episode_stats["reward_skill3_good_cast_trigger"] / frames, 4),
            "reward_skill_after_attack_trigger_rate": round(episode_stats["reward_skill_after_attack_trigger"] / frames, 4),
            "reward_berserk_attack_follow_trigger_rate": round(episode_stats["reward_berserk_attack_follow_trigger"] / frames, 4),
            "reward_tower_poke_hero_trigger_rate": round(episode_stats["reward_tower_poke_hero_trigger"] / frames, 4),
            "reward_tower_poke_dive_trigger_rate": round(episode_stats["reward_tower_poke_dive_trigger"] / frames, 4),
            "reward_defend_tower_clear_trigger_rate": round(episode_stats["reward_defend_tower_clear_trigger"] / frames, 4),
            "reward_defend_tower_ignore_trigger_rate": round(episode_stats["reward_defend_tower_ignore_trigger"] / frames, 4),
            "skill_action_rate": round(episode_stats["skill_action"] / frames, 4),
            "skill1_cd_ratio": round(episode_stats["skill1_cd_ratio"] / frames, 4),
            "skill2_cd_ratio": round(episode_stats["skill2_cd_ratio"] / frames, 4),
            "skill3_cd_ratio": round(episode_stats["skill3_cd_ratio"] / frames, 4),
            "summoner_cd_ratio": round(episode_stats["summoner_cd_ratio"] / frames, 4),
            "summoner_action_rate": round(episode_stats["summoner_action"] / frames, 4),
            "summoner_use_rate": round(episode_stats["summoner_action"] / frames, 4),
            "summoner_legal_rate": round(episode_stats["summoner_legal"] / frames, 4),
            "summoner_target_hero_rate": round(episode_stats["summoner_target_hero"] / frames, 4),
            "summoner_real_cmd_rate": round(episode_stats["summoner_real_cmd"] / frames, 4),
            "berserk_active_rate": round(episode_stats["berserk_active"] / frames, 4),
            "berserk_attack_hero_rate": round(episode_stats["berserk_attack_hero"] / frames, 4),
            "berserk_death_after_use_rate": round(episode_stats["berserk_death_after_use"] / frames, 4),
            "berserk_hurt_to_hero": round(episode_stats["berserk_hurt_to_hero"], 2),
            "berserk_hurt_by_hero": round(episode_stats["berserk_hurt_by_hero"], 2),
            "berserk_hp_advantage_delta": round(episode_stats["berserk_hp_advantage_delta"], 4),
            "low_movement_rate": round(episode_stats["low_movement"] / frames, 4),
            "skill_hit_hero_rate": round(episode_stats["skill_hit_hero"] / frames, 4),
            "skill2_hit_hero_rate": round(episode_stats["skill2_hit_hero"] / frames, 4),
            "skill1_cmd_rate": round(episode_stats["skill1_cmd"] / frames, 4),
            "skill2_cmd_rate": round(episode_stats["skill2_cmd"] / frames, 4),
            "skill3_cmd_rate": round(episode_stats["skill3_cmd"] / frames, 4),
            "skill1_use_rate": round(episode_stats["skill1_use"] / frames, 4),
            "skill2_use_rate": round(episode_stats["skill2_use"] / frames, 4),
            "skill3_use_rate": round(episode_stats["skill3_use"] / frames, 4),
            "skill1_real_cmd_rate": round(episode_stats["skill1_real_cmd"] / frames, 4),
            "skill2_real_cmd_rate": round(episode_stats["skill2_real_cmd"] / frames, 4),
            "skill3_real_cmd_rate": round(episode_stats["skill3_real_cmd"] / frames, 4),
            "skill2_target_hero_rate": round(episode_stats["skill2_target_hero"] / frames, 4),
            "skill1_hit_hero_rate": round(episode_stats["skill1_hit_hero"] / frames, 4),
            "skill3_hit_hero_rate": round(episode_stats["skill3_hit_hero"] / frames, 4),
            "skill3_hit_soldier_rate": round(episode_stats["skill3_hit_soldier"] / frames, 4),
            "skill3_after_tower_attack_rate": round(episode_stats["skill3_after_tower_attack"] / frames, 4),
            "skill1_legal_rate": round(episode_stats["skill1_legal"] / frames, 4),
            "skill2_legal_rate": round(episode_stats["skill2_legal"] / frames, 4),
            "skill3_legal_rate": round(episode_stats["skill3_legal"] / frames, 4),
            "s2_cmd_legal": round(episode_stats["skill2_cmd_legal"] / frames, 4),
            "s3_cmd_legal": round(episode_stats["skill3_cmd_legal"] / frames, 4),
            "s2_cmd_blocked_by_cd": round(episode_stats["skill2_cmd_blocked_by_cd"] / frames, 4),
            "s3_cmd_blocked_by_cd": round(episode_stats["skill3_cmd_blocked_by_cd"] / frames, 4),
            "s2_cmd_blocked_by_mask": round(episode_stats["skill2_cmd_blocked_by_mask"] / frames, 4),
            "s3_cmd_blocked_by_mask": round(episode_stats["skill3_cmd_blocked_by_mask"] / frames, 4),
            "luban_s1_follow_attack_rate": round(episode_stats["luban_s1_follow_attack"] / frames, 4),
            "luban_s2_finish_try_rate": round(episode_stats["luban_s2_finish_try"] / frames, 4),
            "luban_s2_defense_try_rate": round(episode_stats["luban_s2_defense_try"] / frames, 4),
            "luban_s3_push_zone_try_rate": round(episode_stats["luban_s3_push_zone_try"] / frames, 4),
            "direnjie_s1_poke_try_rate": round(episode_stats["direnjie_s1_poke_try"] / frames, 4),
            "direnjie_s2_defense_try_rate": round(episode_stats["direnjie_s2_defense_try"] / frames, 4),
            "direnjie_s3_control_try_rate": round(episode_stats["direnjie_s3_control_try"] / frames, 4),
            "skill_after_hit_follow_attack_rate": round(episode_stats["skill_after_hit_follow_attack"] / frames, 4),
            "skill_interrupt_sweep_rate": round(episode_stats["skill_interrupt_sweep"] / frames, 4),
            "bad_skill_under_enemy_tower_rate": round(episode_stats["bad_skill_under_enemy_tower"] / frames, 4),
            "luban_s2_prior_try_rate": round(skill_prior_stats["luban_s2_prior_try"] / frames, 4),
            "luban_s2_prior_apply_rate": round(skill_prior_stats["luban_s2_prior_apply"] / frames, 4),
            "luban_s3_prior_try_rate": round(skill_prior_stats["luban_s3_prior_try"] / frames, 4),
            "luban_s3_prior_apply_rate": round(skill_prior_stats["luban_s3_prior_apply"] / frames, 4),
            "direnjie_s2_defense_prior_try_rate": round(skill_prior_stats["direnjie_s2_defense_prior_try"] / frames, 4),
            "direnjie_s2_defense_prior_apply_rate": round(skill_prior_stats["direnjie_s2_defense_prior_apply"] / frames, 4),
            "direnjie_s3_prior_try_rate": round(skill_prior_stats["direnjie_s3_prior_try"] / frames, 4),
            "direnjie_s3_prior_apply_rate": round(skill_prior_stats["direnjie_s3_prior_apply"] / frames, 4),
            "prior_blocked_by_cd_rate": round(skill_prior_stats["prior_blocked_by_cd"] / frames, 4),
            "prior_blocked_by_mask_rate": round(skill_prior_stats["prior_blocked_by_mask"] / frames, 4),
            "prior_blocked_by_tower_risk_rate": round(skill_prior_stats["prior_blocked_by_tower_risk"] / frames, 4),
            "prior_blocked_by_low_hp_retreat_rate": round(skill_prior_stats["prior_blocked_by_low_hp_retreat"] / frames, 4),
            "luban_s2_prior_block_cd_rate": round(skill_prior_stats["luban_s2_prior_block_cd"] / frames, 4),
            "luban_s2_prior_block_mask_rate": round(skill_prior_stats["luban_s2_prior_block_mask"] / frames, 4),
            "luban_s2_prior_block_tower_risk_rate": round(skill_prior_stats["luban_s2_prior_block_tower_risk"] / frames, 4),
            "luban_s2_prior_block_retreat_rate": round(skill_prior_stats["luban_s2_prior_block_retreat"] / frames, 4),
            "luban_s3_prior_block_cd_rate": round(skill_prior_stats["luban_s3_prior_block_cd"] / frames, 4),
            "luban_s3_prior_block_mask_rate": round(skill_prior_stats["luban_s3_prior_block_mask"] / frames, 4),
            "luban_s3_prior_block_tower_risk_rate": round(skill_prior_stats["luban_s3_prior_block_tower_risk"] / frames, 4),
            "luban_s3_prior_block_retreat_rate": round(skill_prior_stats["luban_s3_prior_block_retreat"] / frames, 4),
            "direnjie_s2_prior_block_cd_rate": round(skill_prior_stats["direnjie_s2_prior_block_cd"] / frames, 4),
            "direnjie_s2_prior_block_mask_rate": round(skill_prior_stats["direnjie_s2_prior_block_mask"] / frames, 4),
            "direnjie_s2_prior_block_tower_risk_rate": round(skill_prior_stats["direnjie_s2_prior_block_tower_risk"] / frames, 4),
            "direnjie_s2_prior_block_retreat_rate": round(skill_prior_stats["direnjie_s2_prior_block_retreat"] / frames, 4),
            "direnjie_s3_prior_block_cd_rate": round(skill_prior_stats["direnjie_s3_prior_block_cd"] / frames, 4),
            "direnjie_s3_prior_block_mask_rate": round(skill_prior_stats["direnjie_s3_prior_block_mask"] / frames, 4),
            "direnjie_s3_prior_block_tower_risk_rate": round(skill_prior_stats["direnjie_s3_prior_block_tower_risk"] / frames, 4),
            "direnjie_s3_prior_block_retreat_rate": round(skill_prior_stats["direnjie_s3_prior_block_retreat"] / frames, 4),
            "skill_follow_attack_trigger_rate": round(episode_stats["reward_skill_follow_attack_trigger"] / frames, 4),
            "skill_follow_tower_trigger_rate": round(episode_stats["reward_skill_follow_tower_trigger"] / frames, 4),
            "skill_after_attack_rate": round(episode_stats["reward_skill_after_attack_trigger"] / frames, 4),
            "summoner_finish_rate": round(episode_stats["reward_summoner_finish_trigger"] / frames, 4),
            "bad_summoner_rate": round(episode_stats["reward_bad_summoner_trigger"] / frames, 4),
            "bad_skill2_rate": round(episode_stats["reward_bad_skill2_trigger"] / frames, 4),
            "base_stuck_rate": round(episode_stats["base_stuck"] / frames, 4),
            "early_forward_rate": round(episode_stats["early_forward"] / frames, 4),
            "early_low_movement_rate": round(episode_stats["early_low_movement"] / frames, 4),
            "has_luban_any_buff_rate": round(episode_stats["has_luban_any_buff"] / frames, 4),
            "has_luban_output_buff_rate": round(episode_stats["has_luban_output_buff"] / frames, 4),
            "luban_output_buff_damage_hero_rate": round(episode_stats["reward_luban_buff_damage_trigger"] / frames, 4),
            "luban_output_buff_attack_hero_rate": round(episode_stats["luban_output_buff_attack_hero"] / frames, 4),
            "top_luban_buff_id_count": len(episode_stats["seen_luban_buff_ids"]),
            "has_luban_buff_rate": round(episode_stats["has_luban_any_buff"] / frames, 4),
            "luban_buff_damage_hero_rate": round(episode_stats["reward_luban_buff_damage_trigger"] / frames, 4),
            "luban_buff_attack_hero_rate": round(episode_stats["luban_output_buff_attack_hero"] / frames, 4),
            "attack_hero_under_enemy_tower_rate": round(episode_stats["attack_hero_under_enemy_tower"] / frames, 4),
            "tower_poke_window_rate": round(episode_stats["tower_poke_window"] / frames, 4),
            "tower_poke_hero_rate": round(episode_stats["tower_poke_hero"] / frames, 4),
            "defend_tower_window_rate": round(episode_stats["defend_tower_window"] / frames, 4),
            "defend_tower_clear_rate": round(episode_stats["defend_tower_clear"] / frames, 4),
            "enemy_tower_hp_drop": round(episode_stats["enemy_tower_hp_drop"], 2),
            "hurt_to_hero_delta": round(episode_stats["hurt_to_hero_delta"], 2),
            "hurt_by_hero_delta": round(episode_stats["hurt_by_hero_delta"], 2),
            "gold_from_hero": round(episode_stats["gold_from_hero"], 2),
            "gold_from_soldier": round(episode_stats["gold_from_soldier"], 2),
            "gold_from_other": round(episode_stats["gold_from_other"], 2),
            "gold_unattributed": round(episode_stats["gold_unattributed"], 2),
            "gold_passive_or_unknown": round(episode_stats["gold_unattributed"], 2),
            "gold_total_delta": round(episode_stats["gold_total_delta"], 2),
            "gold_event_total": round(episode_stats["gold_event_total"], 2),
            "soldier_kill_count": round(episode_stats["soldier_kill_count"], 2),
            "soldier_last_hit_count": round(episode_stats["soldier_kill_count"], 2),
            "hero_kill_count": round(episode_stats["hero_kill_count"], 2),
            "hero_kill_event_count": round(episode_stats["hero_kill_count"], 2),
        }
        if is_eval:
            eval_id = opponent_agent or GameConfig.EVAL_OPPONENT_MODEL_ID
            monitor_data["eval_win_rate"] = win_rate
            monitor_data["win_rate_eval_model"] = win_rate
            self._add_opponent_monitor_aliases(monitor_data, eval_id, win_rate, "eval")
            self._add_matchup_monitor_aliases(monitor_data, main_config, enemy_config, win_rate, "eval")
        else:
            train_id = opponent_agent or GameConfig.TRAIN_OPPONENT_MODEL_ID
            monitor_data["train_win_rate"] = win_rate
            self._add_opponent_monitor_aliases(monitor_data, train_id, win_rate, "train")
            self._add_matchup_monitor_aliases(monitor_data, main_config, enemy_config, win_rate, "train")
        self._add_hero_monitor_aliases(monitor_data, main_config)
        return monitor_data

    def _add_matchup_monitor_aliases(self, monitor_data, self_hero_id, enemy_hero_id, win_rate, phase):
        matchup = f"{self_hero_id}_vs_{enemy_hero_id}"
        monitor_data[f"win_rate_{matchup}"] = win_rate
        monitor_data[f"win_rate_{phase}_{matchup}"] = win_rate
        for key in (
            "self_tower_hp_ratio",
            "enemy_tower_hp_ratio",
            "enemy_tower_hp_drop",
            "money_per_frame",
            "kill",
            "death",
            "hurt_to_hero_delta",
            "hurt_by_hero_delta",
            "tower_poke_hero_rate",
            "defend_tower_clear_rate",
            "skill2_use_rate",
            "skill3_use_rate",
        ):
            monitor_data[f"{key}_{matchup}"] = monitor_data[key]
            monitor_data[f"{key}_{phase}_{matchup}"] = monitor_data[key]

    def _add_opponent_monitor_aliases(self, monitor_data, model_id, win_rate, phase):
        model_id = str(model_id)
        monitor_data[f"win_rate_{phase}_{model_id}"] = win_rate
        monitor_data[f"kill_{phase}_{model_id}"] = monitor_data["kill"]
        monitor_data[f"death_{phase}_{model_id}"] = monitor_data["death"]
        monitor_data[f"hurt_to_hero_{phase}_{model_id}"] = monitor_data["hurt_to_hero_delta"]
        monitor_data[f"hurt_by_hero_{phase}_{model_id}"] = monitor_data["hurt_by_hero_delta"]
        for key in (
            "win_rate",
            "self_tower_hp",
            "enemy_tower_hp",
            "self_tower_hp_ratio",
            "enemy_tower_hp_ratio",
            "enemy_tower_hp_drop",
            "money_per_frame",
            "kill",
            "death",
            "hurt_to_hero_delta",
            "hurt_by_hero_delta",
        ):
            value = win_rate if key == "win_rate" else monitor_data[key]
            monitor_data[f"{key}_{model_id}"] = value
            monitor_data[f"{key}_{phase}_{model_id}"] = value
        monitor_data[f"frame_{model_id}"] = monitor_data["frame"]
        monitor_data[f"frame_{phase}_{model_id}"] = monitor_data["frame"]

    def _add_hero_monitor_aliases(self, monitor_data, hero_id):
        for skill_no in (1, 2, 3):
            monitor_data[f"s{skill_no}_cmd_{hero_id}"] = monitor_data[f"skill{skill_no}_cmd_rate"]
            monitor_data[f"s{skill_no}_use_{hero_id}"] = monitor_data[f"skill{skill_no}_real_cmd_rate"]
            monitor_data[f"s{skill_no}_hit_{hero_id}"] = monitor_data[f"skill{skill_no}_hit_hero_rate"]
        for skill_no in (2, 3):
            monitor_data[f"s{skill_no}_cmd_legal_{hero_id}"] = monitor_data[f"s{skill_no}_cmd_legal"]
            monitor_data[f"s{skill_no}_cmd_blocked_by_cd_{hero_id}"] = monitor_data[f"s{skill_no}_cmd_blocked_by_cd"]
            monitor_data[f"s{skill_no}_cmd_blocked_by_mask_{hero_id}"] = monitor_data[f"s{skill_no}_cmd_blocked_by_mask"]

    def _attention_stats_from_agent(self, agent):
        empty = {
            "attention_entropy": 0.0,
            "attention_max_prob": 0.0,
            "attention_target_hero_prob": 0.0,
            "attention_target_soldier_prob": 0.0,
            "attention_target_tower_prob": 0.0,
            "target_changed_by_attention_rate": 0.0,
        }
        model = getattr(agent, "model", None)
        stats = getattr(model, "last_attention_stats", None)
        if not isinstance(stats, dict):
            return empty
        return {key: float(stats.get(key, 0.0) or 0.0) for key in empty}

    def _skill_prior_stats_from_agent(self, agent):
        empty = {
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
        }
        stats = getattr(agent, "skill_prior_stats", None)
        if not isinstance(stats, dict):
            return empty
        return {key: float(stats.get(key, 0.0) or 0.0) for key in empty}

    def _new_episode_stats(self):
        return {
            "frames": 0.0,
            "move_action": 0.0,
            "attack_action": 0.0,
            "recall_action": 0.0,
            "hero_target": 0.0,
            "creep_target": 0.0,
            "tower_target": 0.0,
            "enemy_tower_range": 0.0,
            "no_minion_tower_dive": 0.0,
            "safe_push_window": 0.0,
            "tower_attack_with_minion": 0.0,
            "tower_target_with_minion": 0.0,
            "attack_tower_with_minion": 0.0,
            "safe_attack_tower": 0.0,
            "skill_action": 0.0,
            "summoner_action": 0.0,
            "low_movement": 0.0,
            "reward_item_sum": {key: 0.0 for key in self.REWARD_MONITOR_KEYS},
            "reward_push_assist_trigger": 0.0,
            "reward_miss_push_window_trigger": 0.0,
            "reward_finish_trigger": 0.0,
            "reward_lane_follow_trigger": 0.0,
            "reward_hero_damage_trigger": 0.0,
            "reward_combat_kite_trigger": 0.0,
            "reward_combat_standstill_trigger": 0.0,
            "reward_attack_move_trigger": 0.0,
            "reward_attack_standstill_trigger": 0.0,
            "reward_low_hp_cake_trigger": 0.0,
            "reward_low_hp_heal_trigger": 0.0,
            "reward_low_hp_idle_trigger": 0.0,
            "reward_follow_minion_after_clear_trigger": 0.0,
            "reward_idle_after_clear_trigger": 0.0,
            "reward_river_monster_trigger": 0.0,
            "reward_idle_when_monster_trigger": 0.0,
            "reward_attack_hero_in_range_trigger": 0.0,
            "reward_miss_hero_in_range_trigger": 0.0,
            "reward_skill_hit_hero_trigger": 0.0,
            "reward_skill2_ready_unused_trigger": 0.0,
            "reward_skill3_ready_unused_trigger": 0.0,
            "reward_luban_buff_damage_trigger": 0.0,
            "reward_skill2_finish_trigger": 0.0,
            "reward_summoner_finish_trigger": 0.0,
            "reward_base_stuck_trigger": 0.0,
            "reward_early_forward_trigger": 0.0,
            "reward_skill_follow_attack_trigger": 0.0,
            "reward_skill_follow_tower_trigger": 0.0,
            "reward_bad_skill2_trigger": 0.0,
            "reward_bad_skill3_trigger": 0.0,
            "reward_bad_summoner_trigger": 0.0,
            "reward_push_window_target_tower_trigger": 0.0,
            "reward_push_window_wrong_hero_trigger": 0.0,
            "reward_retreat_to_own_tower_trigger": 0.0,
            "reward_bad_attack_low_hp_trigger": 0.0,
            "reward_safe_recall_trigger": 0.0,
            "reward_skill1_good_cast_trigger": 0.0,
            "reward_skill2_good_cast_trigger": 0.0,
            "reward_skill3_good_cast_trigger": 0.0,
            "reward_skill_after_attack_trigger": 0.0,
            "reward_berserk_attack_follow_trigger": 0.0,
            "reward_tower_poke_hero_trigger": 0.0,
            "reward_tower_poke_dive_trigger": 0.0,
            "reward_defend_tower_clear_trigger": 0.0,
            "reward_defend_tower_ignore_trigger": 0.0,
            "last_hero_pos": None,
            "last_lane_progress": None,
            "last_self_tower_distance": None,
            "last_enemy_tower_distance": None,
            "last_enemy_tower_hp": None,
            "last_hurt_to_hero": None,
            "last_hurt_by_hero": None,
            "last_self_money": None,
            "last_skill_hit_total": None,
            "last_skill2_hit_total": None,
            "last_skill_hits": {1: None, 2: None, 3: None},
            "skill_after_hit_window": 0,
            "luban_sweep_window": 0,
            "skill3_recent_frames": 0,
            "berserk_recent_frames": 0,
            "last_hp_advantage": None,
            "last_berserk_active": False,
            "seen_death_events": set(),
            "seen_luban_buff_ids": set(),
            "enemy_tower_hp_drop": 0.0,
            "hurt_to_hero_delta": 0.0,
            "hurt_by_hero_delta": 0.0,
            "gold_from_hero": 0.0,
            "gold_from_soldier": 0.0,
            "gold_from_other": 0.0,
            "gold_unattributed": 0.0,
            "gold_total_delta": 0.0,
            "gold_event_total": 0.0,
            "soldier_kill_count": 0.0,
            "hero_kill_count": 0.0,
            "skill_hit_hero": 0.0,
            "skill2_hit_hero": 0.0,
            "skill1_cmd": 0.0,
            "skill2_cmd": 0.0,
            "skill3_cmd": 0.0,
            "skill1_use": 0.0,
            "skill2_use": 0.0,
            "skill3_use": 0.0,
            "skill2_target_hero": 0.0,
            "skill1_real_cmd": 0.0,
            "skill2_real_cmd": 0.0,
            "skill3_real_cmd": 0.0,
            "skill2_cmd_legal": 0.0,
            "skill3_cmd_legal": 0.0,
            "skill2_cmd_blocked_by_cd": 0.0,
            "skill3_cmd_blocked_by_cd": 0.0,
            "skill2_cmd_blocked_by_mask": 0.0,
            "skill3_cmd_blocked_by_mask": 0.0,
            "luban_s1_follow_attack": 0.0,
            "luban_s2_finish_try": 0.0,
            "luban_s2_defense_try": 0.0,
            "luban_s3_push_zone_try": 0.0,
            "direnjie_s1_poke_try": 0.0,
            "direnjie_s2_defense_try": 0.0,
            "direnjie_s3_control_try": 0.0,
            "skill_after_hit_follow_attack": 0.0,
            "skill_interrupt_sweep": 0.0,
            "bad_skill_under_enemy_tower": 0.0,
            "skill1_hit_hero": 0.0,
            "skill3_hit_hero": 0.0,
            "skill3_hit_soldier": 0.0,
            "skill3_after_tower_attack": 0.0,
            "skill1_legal": 0.0,
            "skill2_legal": 0.0,
            "skill3_legal": 0.0,
            "skill1_cd_ratio": 0.0,
            "skill2_cd_ratio": 0.0,
            "skill3_cd_ratio": 0.0,
            "summoner_cd_ratio": 0.0,
            "summoner_legal": 0.0,
            "summoner_target_hero": 0.0,
            "summoner_real_cmd": 0.0,
            "berserk_active": 0.0,
            "berserk_attack_hero": 0.0,
            "berserk_death_after_use": 0.0,
            "berserk_hurt_to_hero": 0.0,
            "berserk_hurt_by_hero": 0.0,
            "berserk_hp_advantage_delta": 0.0,
            "base_stuck": 0.0,
            "early_forward": 0.0,
            "early_low_movement": 0.0,
            "strict_push_window": 0.0,
            "near_push_window": 0.0,
            "push_window": 0.0,
            "push_window_target_tower": 0.0,
            "push_window_wrong_hero": 0.0,
            "retreat_window": 0.0,
            "retreat_to_own_tower": 0.0,
            "bad_attack_low_hp": 0.0,
            "safe_recall": 0.0,
            "low_hp_death": 0.0,
            "has_luban_any_buff": 0.0,
            "has_luban_output_buff": 0.0,
            "luban_output_buff_attack_hero": 0.0,
            "attack_hero_under_enemy_tower": 0.0,
            "tower_poke_window": 0.0,
            "tower_poke_hero": 0.0,
            "defend_tower_window": 0.0,
            "defend_tower_clear": 0.0,
        }

    def _update_reward_item_stats(self, stats, reward):
        reward_item_sum = stats.get("reward_item_sum", {})
        for key in reward_item_sum:
            reward_item_sum[key] += float(reward.get(key, 0.0) or 0.0)
        if float(reward.get("reward_push_assist", 0.0) or 0.0) != 0.0:
            stats["reward_push_assist_trigger"] += 1.0
        if float(reward.get("reward_miss_push_window", 0.0) or 0.0) != 0.0:
            stats["reward_miss_push_window_trigger"] += 1.0
        if float(reward.get("reward_lane_follow", 0.0) or 0.0) != 0.0:
            stats["reward_lane_follow_trigger"] += 1.0
        if float(reward.get("reward_finish", 0.0) or 0.0) != 0.0:
            stats["reward_finish_trigger"] += 1.0
        if float(reward.get("reward_hero_damage", 0.0) or 0.0) != 0.0:
            stats["reward_hero_damage_trigger"] += 1.0
        if float(reward.get("reward_combat_kite", 0.0) or 0.0) != 0.0:
            stats["reward_combat_kite_trigger"] += 1.0
        if float(reward.get("reward_combat_standstill", 0.0) or 0.0) != 0.0:
            stats["reward_combat_standstill_trigger"] += 1.0
        if float(reward.get("reward_attack_move", 0.0) or 0.0) != 0.0:
            stats["reward_attack_move_trigger"] += 1.0
        if float(reward.get("reward_attack_standstill", 0.0) or 0.0) != 0.0:
            stats["reward_attack_standstill_trigger"] += 1.0
        if float(reward.get("reward_low_hp_cake", 0.0) or 0.0) != 0.0:
            stats["reward_low_hp_cake_trigger"] += 1.0
        if float(reward.get("reward_low_hp_heal", 0.0) or 0.0) != 0.0:
            stats["reward_low_hp_heal_trigger"] += 1.0
        if float(reward.get("reward_low_hp_idle", 0.0) or 0.0) != 0.0:
            stats["reward_low_hp_idle_trigger"] += 1.0
        if float(reward.get("reward_follow_minion_after_clear", 0.0) or 0.0) != 0.0:
            stats["reward_follow_minion_after_clear_trigger"] += 1.0
        if float(reward.get("reward_idle_after_clear", 0.0) or 0.0) != 0.0:
            stats["reward_idle_after_clear_trigger"] += 1.0
        if float(reward.get("reward_river_monster", 0.0) or 0.0) != 0.0:
            stats["reward_river_monster_trigger"] += 1.0
        if float(reward.get("reward_idle_when_monster", 0.0) or 0.0) != 0.0:
            stats["reward_idle_when_monster_trigger"] += 1.0
        if float(reward.get("reward_attack_hero_in_range", 0.0) or 0.0) != 0.0:
            stats["reward_attack_hero_in_range_trigger"] += 1.0
        if float(reward.get("reward_miss_hero_in_range", 0.0) or 0.0) != 0.0:
            stats["reward_miss_hero_in_range_trigger"] += 1.0
        if float(reward.get("reward_skill_hit_hero", 0.0) or 0.0) != 0.0:
            stats["reward_skill_hit_hero_trigger"] += 1.0
        if float(reward.get("reward_skill2_ready_unused", 0.0) or 0.0) != 0.0:
            stats["reward_skill2_ready_unused_trigger"] += 1.0
        if float(reward.get("reward_skill3_ready_unused", 0.0) or 0.0) != 0.0:
            stats["reward_skill3_ready_unused_trigger"] += 1.0
        if float(reward.get("reward_luban_buff_damage", 0.0) or 0.0) != 0.0:
            stats["reward_luban_buff_damage_trigger"] += 1.0
        if float(reward.get("reward_skill2_finish", 0.0) or 0.0) != 0.0:
            stats["reward_skill2_finish_trigger"] += 1.0
        if float(reward.get("reward_summoner_finish", 0.0) or 0.0) != 0.0:
            stats["reward_summoner_finish_trigger"] += 1.0
        if float(reward.get("reward_base_stuck", 0.0) or 0.0) != 0.0:
            stats["reward_base_stuck_trigger"] += 1.0
        if float(reward.get("reward_early_forward", 0.0) or 0.0) != 0.0:
            stats["reward_early_forward_trigger"] += 1.0
        if float(reward.get("reward_skill_follow_attack", 0.0) or 0.0) != 0.0:
            stats["reward_skill_follow_attack_trigger"] += 1.0
        if float(reward.get("reward_skill_follow_tower", 0.0) or 0.0) != 0.0:
            stats["reward_skill_follow_tower_trigger"] += 1.0
        if float(reward.get("reward_bad_skill2", 0.0) or 0.0) != 0.0:
            stats["reward_bad_skill2_trigger"] += 1.0
        if float(reward.get("reward_bad_skill3", 0.0) or 0.0) != 0.0:
            stats["reward_bad_skill3_trigger"] += 1.0
        if float(reward.get("reward_bad_summoner", 0.0) or 0.0) != 0.0:
            stats["reward_bad_summoner_trigger"] += 1.0
        if float(reward.get("reward_push_window_target_tower", 0.0) or 0.0) != 0.0:
            stats["reward_push_window_target_tower_trigger"] += 1.0
        if float(reward.get("reward_push_window_wrong_hero", 0.0) or 0.0) != 0.0:
            stats["reward_push_window_wrong_hero_trigger"] += 1.0
        if float(reward.get("reward_retreat_to_own_tower", 0.0) or 0.0) != 0.0:
            stats["reward_retreat_to_own_tower_trigger"] += 1.0
        if float(reward.get("reward_bad_attack_low_hp", 0.0) or 0.0) != 0.0:
            stats["reward_bad_attack_low_hp_trigger"] += 1.0
        if float(reward.get("reward_safe_recall", 0.0) or 0.0) != 0.0:
            stats["reward_safe_recall_trigger"] += 1.0
        if float(reward.get("reward_skill1_good_cast", 0.0) or 0.0) != 0.0:
            stats["reward_skill1_good_cast_trigger"] += 1.0
        if float(reward.get("reward_skill2_good_cast", 0.0) or 0.0) != 0.0:
            stats["reward_skill2_good_cast_trigger"] += 1.0
        if float(reward.get("reward_skill3_good_cast", 0.0) or 0.0) != 0.0:
            stats["reward_skill3_good_cast_trigger"] += 1.0
        if float(reward.get("reward_skill_after_attack", 0.0) or 0.0) != 0.0:
            stats["reward_skill_after_attack_trigger"] += 1.0
        if float(reward.get("reward_berserk_attack_follow", 0.0) or 0.0) != 0.0:
            stats["reward_berserk_attack_follow_trigger"] += 1.0
        if float(reward.get("reward_tower_poke_hero", 0.0) or 0.0) != 0.0:
            stats["reward_tower_poke_hero_trigger"] += 1.0
        if float(reward.get("reward_tower_poke_dive", 0.0) or 0.0) != 0.0:
            stats["reward_tower_poke_dive_trigger"] += 1.0
        if float(reward.get("reward_defend_tower_clear", 0.0) or 0.0) != 0.0:
            stats["reward_defend_tower_clear_trigger"] += 1.0
        if float(reward.get("reward_defend_tower_ignore", 0.0) or 0.0) != 0.0:
            stats["reward_defend_tower_ignore_trigger"] += 1.0

    def _update_episode_stats(self, stats, observation, action):
        frame_state = observation["frame_state"]
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_hero = self._get_enemy_hero(frame_state, camp)
        main_tower = self._get_tower(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)
        my_soldiers = self._soldiers(frame_state, camp)
        enemy_soldiers = self._enemy_soldiers(frame_state, camp)
        button = int(action[0]) if action is not None and len(action) > 0 else -1
        target = int(action[5]) if action is not None and len(action) > 5 else -1
        main_config = self._int_field(main_hero, "config_id", 0)
        frame_no = float(frame_state.get("frame_no", 0.0) or 0.0)
        in_enemy_tower_range = self._in_tower_range(main_hero, enemy_tower)
        has_minion_under_tower = self._has_ally_minion_under_enemy_tower(my_soldiers, enemy_tower)
        enemy_minion_under_own_tower = self._has_ally_minion_under_enemy_tower(enemy_soldiers, main_tower)
        enemy_under_own_tower = self._in_tower_range(enemy_hero, enemy_tower)
        safe_push_window = self._safe_to_push(main_hero, enemy_hero, my_soldiers, enemy_tower)
        action_attacks_or_skills = button == GameConfig.BUTTON_ATTACK or button in GameConfig.SKILL_BUTTONS
        main_has_luban_any_buff = has_luban_any_buff(main_hero)
        main_has_luban_output_buff = has_luban_output_buff(main_hero)
        summoner_real_cmd = self._has_summoner_real_cmd(main_hero)
        berserk_observed = self._has_berserk_buff(main_hero) or summoner_real_cmd or button == GameConfig.BUTTON_SUMMONER
        skill_totals = self._skill_totals(main_hero)
        skill_details = self._skill_details(main_hero)
        skill_state_features = self._skill_state_features(main_hero)
        lane_progress = self._lane_progress(main_hero, main_tower, enemy_tower)
        self_hp_ratio = self._hp_ratio(main_hero)
        enemy_hp_ratio = self._hp_ratio(enemy_hero)
        hp_advantage = self_hp_ratio - enemy_hp_ratio
        current_hurt_by_hero = self._value(main_hero, "total_be_hurt_by_hero")
        recent_hurt_by_hero = (
            max(0.0, current_hurt_by_hero - stats["last_hurt_by_hero"])
            if stats["last_hurt_by_hero"] is not None
            else 0.0
        )
        enemy_distance = self._distance(self._position(main_hero), self._position(enemy_hero)) if main_hero and enemy_hero else 999999.0
        enemy_close = enemy_hero is not None and enemy_hp_ratio > 0.0 and enemy_distance <= 12000.0
        enemy_mid_close = enemy_hero is not None and enemy_hp_ratio > 0.0 and 3500.0 <= enemy_distance <= 12000.0
        enemy_melee_close = enemy_hero is not None and enemy_hp_ratio > 0.0 and enemy_distance <= 6500.0
        enemy_tower_alive = enemy_tower is not None and self._value(enemy_tower, "hp") > 0.0
        retreat_window = self_hp_ratio < 0.30 or (hp_advantage < -0.25 and enemy_close)
        own_tower_distance = self._distance(self._position(main_hero), self._position(main_tower)) if main_hero and main_tower else 0.0
        enemy_tower_distance = (
            self._distance(self._position(main_hero), self._position(enemy_tower)) if main_hero and enemy_tower else 0.0
        )
        self_attack_range = self._value(main_hero, "attack_range") or 7000.0
        enemy_close_dangerous = self._enemy_close_dangerous(main_hero, enemy_hero)
        strict_push_window = has_minion_under_tower and self_hp_ratio > 0.35 and enemy_tower_alive
        near_push_window = (
            enemy_tower_alive
            and enemy_tower_distance <= self_attack_range + 1200.0
            and self_hp_ratio > 0.45
            and not enemy_close_dangerous
        )
        push_window = strict_push_window or near_push_window
        enemy_to_enemy_tower_distance = (
            self._distance(self._position(enemy_hero), self._position(enemy_tower))
            if enemy_hero and enemy_tower
            else 999999.0
        )
        enemy_near_enemy_tower = enemy_to_enemy_tower_distance <= (self._value(enemy_tower, "attack_range") or 9000.0)
        enemy_soldier_cluster = len(enemy_soldiers) >= 3
        skill_after_hit_window_active = stats["skill_after_hit_window"] > 0
        luban_sweep_window_active = stats["luban_sweep_window"] > 0
        moving_back = (
            button == GameConfig.BUTTON_MOVE
            and (
                (
                    stats["last_self_tower_distance"] is not None
                    and own_tower_distance < stats["last_self_tower_distance"] - 50.0
                )
                or (
                    stats["last_enemy_tower_distance"] is not None
                    and enemy_tower_distance > stats["last_enemy_tower_distance"] + 50.0
                )
                or (
                    stats["last_lane_progress"] is not None
                    and lane_progress < stats["last_lane_progress"] - 0.001
                )
            )
        )
        self_under_own_tower = self._in_tower_range(main_hero, main_tower)
        action_is_attack = button == GameConfig.BUTTON_ATTACK or button in GameConfig.SKILL_BUTTONS
        dangerous_low_hp_attack = (
            retreat_window
            and action_is_attack
            and target == GameConfig.TARGET_ENEMY
            and enemy_hp_ratio >= 0.40
            and not self_under_own_tower
        )
        safe_to_recall = self_hp_ratio < 0.25 and (enemy_distance > 7000.0 or self_under_own_tower)
        skill_attack_enemy = button in GameConfig.SKILL_BUTTONS and target == GameConfig.TARGET_ENEMY
        bad_skill_under_enemy_tower = skill_attack_enemy and in_enemy_tower_range and not has_minion_under_tower

        stats["frames"] += 1.0
        stats["move_action"] += 1.0 if button == GameConfig.BUTTON_MOVE else 0.0
        stats["attack_action"] += 1.0 if button == GameConfig.BUTTON_ATTACK else 0.0
        stats["recall_action"] += 1.0 if button == GameConfig.BUTTON_RECALL else 0.0
        stats["skill_action"] += 1.0 if button in GameConfig.SKILL_BUTTONS else 0.0
        stats["summoner_action"] += 1.0 if button == GameConfig.BUTTON_SUMMONER else 0.0
        stats["summoner_legal"] += 1.0 if self._legal_button(observation, GameConfig.BUTTON_SUMMONER) else 0.0
        stats["summoner_target_hero"] += 1.0 if button == GameConfig.BUTTON_SUMMONER and target == GameConfig.TARGET_ENEMY else 0.0
        stats["summoner_real_cmd"] += 1.0 if summoner_real_cmd else 0.0
        if berserk_observed:
            stats["berserk_recent_frames"] = 80
        berserk_active = stats["berserk_recent_frames"] > 0
        stats["last_berserk_active"] = berserk_active
        stats["berserk_active"] += 1.0 if berserk_active else 0.0
        stats["berserk_attack_hero"] += (
            1.0 if berserk_active and target == GameConfig.TARGET_ENEMY and action_attacks_or_skills else 0.0
        )
        stats["berserk_death_after_use"] += (
            1.0 if berserk_active and main_hero and self._value(main_hero, "hp") <= 0 else 0.0
        )
        if berserk_active and stats["last_hp_advantage"] is not None:
            stats["berserk_hp_advantage_delta"] += hp_advantage - stats["last_hp_advantage"]
        stats["last_hp_advantage"] = hp_advantage
        stats["skill1_cmd"] += 1.0 if button == GameConfig.BUTTON_SKILL_1 else 0.0
        stats["skill2_cmd"] += 1.0 if button == GameConfig.BUTTON_SKILL_2 else 0.0
        stats["skill3_cmd"] += 1.0 if button == GameConfig.BUTTON_SKILL_3 else 0.0
        stats["skill1_use"] += 1.0 if button == GameConfig.BUTTON_SKILL_1 else 0.0
        stats["skill2_use"] += 1.0 if button == GameConfig.BUTTON_SKILL_2 else 0.0
        stats["skill3_use"] += 1.0 if button == GameConfig.BUTTON_SKILL_3 else 0.0
        stats["skill2_target_hero"] += 1.0 if button == GameConfig.BUTTON_SKILL_2 and target == GameConfig.TARGET_ENEMY else 0.0
        stats["skill1_real_cmd"] += 1.0 if self._has_skill_real_cmd(main_hero, 1) else 0.0
        stats["skill2_real_cmd"] += 1.0 if self._has_skill_real_cmd(main_hero, 2) else 0.0
        stats["skill3_real_cmd"] += 1.0 if self._has_skill_real_cmd(main_hero, 3) else 0.0
        for skill_no, skill_button in ((2, GameConfig.BUTTON_SKILL_2), (3, GameConfig.BUTTON_SKILL_3)):
            selected_skill = button == skill_button
            legal_button = self._legal_button(observation, skill_button)
            blocked_by_cd = skill_state_features[f"skill{skill_no}_cd_ratio"] > 0.02
            blocked_by_mask = not self._target_allowed_for_button(observation, skill_button, target)
            stats[f"skill{skill_no}_cmd_legal"] += 1.0 if selected_skill and legal_button else 0.0
            stats[f"skill{skill_no}_cmd_blocked_by_cd"] += 1.0 if selected_skill and blocked_by_cd else 0.0
            stats[f"skill{skill_no}_cmd_blocked_by_mask"] += 1.0 if selected_skill and blocked_by_mask else 0.0
        stats["luban_s1_follow_attack"] += (
            1.0
            if main_config == 112
            and luban_sweep_window_active
            and button == GameConfig.BUTTON_ATTACK
            and target == GameConfig.TARGET_ENEMY
            else 0.0
        )
        stats["luban_s2_finish_try"] += (
            1.0
            if main_config == 112
            and button == GameConfig.BUTTON_SKILL_2
            and target == GameConfig.TARGET_ENEMY
            and enemy_hp_ratio <= 0.35
            and not bad_skill_under_enemy_tower
            else 0.0
        )
        stats["luban_s2_defense_try"] += (
            1.0
            if main_config == 112
            and button == GameConfig.BUTTON_SKILL_2
            and enemy_melee_close
            and (self_hp_ratio < 0.55 or recent_hurt_by_hero > 0.0)
            else 0.0
        )
        stats["luban_s3_push_zone_try"] += (
            1.0
            if main_config == 112
            and button == GameConfig.BUTTON_SKILL_3
            and push_window
            and (enemy_near_enemy_tower or enemy_soldier_cluster)
            and not bad_skill_under_enemy_tower
            else 0.0
        )
        stats["direnjie_s1_poke_try"] += (
            1.0
            if main_config == 133
            and button == GameConfig.BUTTON_SKILL_1
            and target == GameConfig.TARGET_ENEMY
            and enemy_mid_close
            and not bad_skill_under_enemy_tower
            else 0.0
        )
        stats["direnjie_s2_defense_try"] += (
            1.0
            if main_config == 133
            and button == GameConfig.BUTTON_SKILL_2
            and (self_hp_ratio < 0.50 or enemy_melee_close or retreat_window)
            and not bad_skill_under_enemy_tower
            else 0.0
        )
        stats["direnjie_s3_control_try"] += (
            1.0
            if main_config == 133
            and button == GameConfig.BUTTON_SKILL_3
            and target == GameConfig.TARGET_ENEMY
            and enemy_hp_ratio <= 0.65
            and 4500.0 <= enemy_distance <= 12500.0
            and not bad_skill_under_enemy_tower
            else 0.0
        )
        stats["skill_after_hit_follow_attack"] += (
            1.0
            if skill_after_hit_window_active and button == GameConfig.BUTTON_ATTACK and target == GameConfig.TARGET_ENEMY
            else 0.0
        )
        stats["skill_interrupt_sweep"] += (
            1.0
            if main_config == 112
            and luban_sweep_window_active
            and button in (GameConfig.BUTTON_MOVE, GameConfig.BUTTON_SKILL_1, GameConfig.BUTTON_SKILL_2, GameConfig.BUTTON_SKILL_3)
            else 0.0
        )
        stats["bad_skill_under_enemy_tower"] += 1.0 if bad_skill_under_enemy_tower else 0.0
        stats["skill3_hit_soldier"] += 1.0 if button == GameConfig.BUTTON_SKILL_3 and target in GameConfig.TARGET_SOLDIERS else 0.0
        stats["skill1_legal"] += 1.0 if self._legal_button(observation, GameConfig.BUTTON_SKILL_1) else 0.0
        stats["skill2_legal"] += 1.0 if self._legal_button(observation, GameConfig.BUTTON_SKILL_2) else 0.0
        stats["skill3_legal"] += 1.0 if self._legal_button(observation, GameConfig.BUTTON_SKILL_3) else 0.0
        stats["skill1_cd_ratio"] += skill_state_features["skill1_cd_ratio"]
        stats["skill2_cd_ratio"] += skill_state_features["skill2_cd_ratio"]
        stats["skill3_cd_ratio"] += skill_state_features["skill3_cd_ratio"]
        stats["summoner_cd_ratio"] += skill_state_features["summoner_cd_ratio"]
        stats["hero_target"] += 1.0 if target == GameConfig.TARGET_ENEMY else 0.0
        stats["creep_target"] += 1.0 if target in GameConfig.TARGET_SOLDIERS else 0.0
        stats["tower_target"] += 1.0 if target == GameConfig.TARGET_TOWER else 0.0
        stats["enemy_tower_range"] += 1.0 if in_enemy_tower_range else 0.0
        stats["no_minion_tower_dive"] += 1.0 if in_enemy_tower_range and not has_minion_under_tower else 0.0
        stats["strict_push_window"] += 1.0 if strict_push_window else 0.0
        stats["near_push_window"] += 1.0 if near_push_window else 0.0
        stats["push_window"] += 1.0 if push_window else 0.0
        stats["push_window_target_tower"] += (
            1.0 if push_window and button == GameConfig.BUTTON_ATTACK and target == GameConfig.TARGET_TOWER else 0.0
        )
        stats["push_window_wrong_hero"] += (
            1.0 if push_window and target == GameConfig.TARGET_ENEMY and enemy_hp_ratio >= 0.40 else 0.0
        )
        stats["retreat_window"] += 1.0 if retreat_window else 0.0
        stats["retreat_to_own_tower"] += 1.0 if retreat_window and moving_back else 0.0
        stats["bad_attack_low_hp"] += 1.0 if dangerous_low_hp_attack else 0.0
        stats["safe_recall"] += 1.0 if safe_to_recall and button == GameConfig.BUTTON_RECALL else 0.0
        stats["low_hp_death"] += 1.0 if self_hp_ratio < 0.30 and main_hero and self._value(main_hero, "hp") <= 0 else 0.0
        stats["attack_hero_under_enemy_tower"] += (
            1.0 if in_enemy_tower_range and target == GameConfig.TARGET_ENEMY and action_attacks_or_skills else 0.0
        )
        tower_poke_window = has_minion_under_tower and enemy_under_own_tower and not in_enemy_tower_range
        stats["tower_poke_window"] += 1.0 if tower_poke_window else 0.0
        stats["tower_poke_hero"] += (
            1.0 if tower_poke_window and target == GameConfig.TARGET_ENEMY and action_attacks_or_skills else 0.0
        )
        stats["defend_tower_window"] += 1.0 if enemy_minion_under_own_tower else 0.0
        stats["defend_tower_clear"] += (
            1.0
            if enemy_minion_under_own_tower
            and button == GameConfig.BUTTON_ATTACK
            and target in GameConfig.TARGET_SOLDIERS
            else 0.0
        )
        stats["safe_push_window"] += 1.0 if safe_push_window else 0.0
        stats["tower_target_with_minion"] += 1.0 if target == GameConfig.TARGET_TOWER and has_minion_under_tower else 0.0
        attack_tower_with_minion = (
            button == GameConfig.BUTTON_ATTACK
            and target == GameConfig.TARGET_TOWER
            and has_minion_under_tower
        )
        safe_attack_tower = (
            safe_push_window
            and button == GameConfig.BUTTON_ATTACK
            and target == GameConfig.TARGET_TOWER
        )
        if stats["skill3_recent_frames"] > 0 and safe_attack_tower:
            stats["skill3_after_tower_attack"] += 1.0
        stats["attack_tower_with_minion"] += 1.0 if attack_tower_with_minion else 0.0
        stats["safe_attack_tower"] += 1.0 if safe_attack_tower else 0.0
        stats["tower_attack_with_minion"] += 1.0 if attack_tower_with_minion else 0.0
        stats["has_luban_any_buff"] += 1.0 if main_has_luban_any_buff else 0.0
        stats["has_luban_output_buff"] += 1.0 if main_has_luban_output_buff else 0.0
        stats["seen_luban_buff_ids"].update(collect_buff_config_ids(main_hero) & LUBAN_BUFF_IDS)
        stats["luban_output_buff_attack_hero"] += (
            1.0 if main_has_luban_output_buff and target == GameConfig.TARGET_ENEMY and action_attacks_or_skills else 0.0
        )
        if stats["last_skill_hit_total"] is not None:
            stats["skill_hit_hero"] += max(0.0, skill_totals["hit_total"] - stats["last_skill_hit_total"])
        if stats["last_skill2_hit_total"] is not None:
            stats["skill2_hit_hero"] += max(0.0, skill_totals["skill2_hit_total"] - stats["last_skill2_hit_total"])
        for skill_no in (1, 2, 3):
            last_hit = stats["last_skill_hits"][skill_no]
            hit_total = skill_details[skill_no]["hit_total"]
            if last_hit is not None:
                hit_delta = max(0.0, hit_total - last_hit)
                if skill_no == 1:
                    stats["skill1_hit_hero"] += hit_delta
                elif skill_no == 3:
                    stats["skill3_hit_hero"] += hit_delta
                if hit_delta > 0.0:
                    stats["skill_after_hit_window"] = max(stats["skill_after_hit_window"], 10)
                    if main_config == 112 and skill_no == 1:
                        stats["luban_sweep_window"] = max(stats["luban_sweep_window"], 10)
            stats["last_skill_hits"][skill_no] = hit_total
        stats["last_skill_hit_total"] = skill_totals["hit_total"]
        stats["last_skill2_hit_total"] = skill_totals["skill2_hit_total"]
        hero_pos = self._position(main_hero)
        low_movement = (
            button in (GameConfig.BUTTON_NONE, GameConfig.BUTTON_NOOP, GameConfig.BUTTON_MOVE)
            and stats["last_hero_pos"] is not None
            and self._distance(hero_pos, stats["last_hero_pos"]) < 300.0
        )
        if (
            low_movement
        ):
            stats["low_movement"] += 1.0
        early_game = frame_no < 1200.0
        near_own_base = lane_progress < 0.12
        moved_forward = (
            stats["last_lane_progress"] is not None
            and lane_progress > stats["last_lane_progress"] + 0.001
        )
        if early_game and low_movement:
            stats["early_low_movement"] += 1.0
        if early_game and near_own_base and low_movement:
            stats["base_stuck"] += 1.0
        if early_game and moved_forward:
            stats["early_forward"] += 1.0
        if button == GameConfig.BUTTON_SKILL_3:
            stats["skill3_recent_frames"] = 5
        elif stats["skill3_recent_frames"] > 0:
            stats["skill3_recent_frames"] -= 1
        if stats["skill_after_hit_window"] > 0:
            stats["skill_after_hit_window"] -= 1
        if stats["luban_sweep_window"] > 0:
            stats["luban_sweep_window"] -= 1
        if not berserk_observed and stats["berserk_recent_frames"] > 0:
            stats["berserk_recent_frames"] -= 1
        stats["last_hero_pos"] = hero_pos
        stats["last_lane_progress"] = lane_progress
        stats["last_self_tower_distance"] = own_tower_distance
        stats["last_enemy_tower_distance"] = enemy_tower_distance

    def _update_episode_outcome_stats(self, stats, observation):
        frame_state = observation["frame_state"]
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)

        enemy_tower_hp = self._value(enemy_tower, "hp")
        if stats["last_enemy_tower_hp"] is not None:
            stats["enemy_tower_hp_drop"] += max(0.0, stats["last_enemy_tower_hp"] - enemy_tower_hp)
        stats["last_enemy_tower_hp"] = enemy_tower_hp

        hurt_to_hero = self._value(main_hero, "total_hurt_to_hero")
        if stats["last_hurt_to_hero"] is not None:
            hurt_to_delta = max(0.0, hurt_to_hero - stats["last_hurt_to_hero"])
            stats["hurt_to_hero_delta"] += hurt_to_delta
            if stats.get("last_berserk_active", False):
                stats["berserk_hurt_to_hero"] += hurt_to_delta
        stats["last_hurt_to_hero"] = hurt_to_hero

        hurt_by_hero = self._value(main_hero, "total_be_hurt_by_hero")
        if stats["last_hurt_by_hero"] is not None:
            hurt_by_delta = max(0.0, hurt_by_hero - stats["last_hurt_by_hero"])
            stats["hurt_by_hero_delta"] += hurt_by_delta
            if stats.get("last_berserk_active", False):
                stats["berserk_hurt_by_hero"] += hurt_by_delta
        stats["last_hurt_by_hero"] = hurt_by_hero

        self_money = self._resource(main_hero, "money")
        money_delta = 0.0
        if stats["last_self_money"] is not None:
            money_delta = max(0.0, self_money - stats["last_self_money"])
            stats["gold_total_delta"] += money_delta
        stats["last_self_money"] = self_money

        attributed_money = 0.0
        frame_action = frame_state.get("frame_action", {}) or {}
        for dead_action in frame_action.get("dead_action", []):
            death = dead_action.get("death", {}) if isinstance(dead_action, dict) else {}
            killer = dead_action.get("killer", {}) if isinstance(dead_action, dict) else {}
            if not killer or killer.get("camp") != camp:
                continue
            event_key = self._death_event_key(death, killer)
            if event_key is not None:
                if event_key in stats["seen_death_events"]:
                    continue
                stats["seen_death_events"].add(event_key)
            income = killer.get("income_info", {}) or {}
            money = float(income.get("money", 0.0) or 0.0)
            source_type = self._death_source_type(death)
            if source_type == "hero":
                stats["gold_from_hero"] += money
                stats["hero_kill_count"] += 1.0
                attributed_money += money
            elif source_type == "soldier":
                stats["gold_from_soldier"] += money
                stats["soldier_kill_count"] += 1.0
                attributed_money += money
            else:
                stats["gold_from_other"] += money
                attributed_money += money
            stats["gold_event_total"] += money
        if money_delta > attributed_money:
            stats["gold_unattributed"] += money_delta - attributed_money

    def _death_event_key(self, death, killer):
        runtime_id = int(death.get("runtime_id", 0) or 0)
        if runtime_id <= 0:
            return None
        return (
            runtime_id,
            self._int_field(death, "camp", -1),
            self._int_field(death, "config_id", -1),
            self._int_field(death, "sub_type", -1),
            self._int_field(killer, "runtime_id", 0),
        )

    def _death_source_type(self, death):
        actor_type = self._int_field(death, "actor_type", -1)
        sub_type = self._int_field(death, "sub_type", -1)
        config_id = self._int_field(death, "config_id", -1)
        if actor_type == GameConfig.ACTOR_TYPE_HERO or config_id in GameConfig.HERO_IDS:
            return "hero"
        if sub_type in GameConfig.SOLDIER_SUB_TYPES or config_id in GameConfig.LANE_SOLDIER_CONFIG_IDS:
            return "soldier"
        if self._is_organ(death):
            return "organ"
        return "other"

    def _int_field(self, data, key, default=0):
        if not data or data.get(key) is None:
            return default
        try:
            return int(data.get(key))
        except (TypeError, ValueError):
            return default

    def _lane_progress(self, hero, main_tower, enemy_tower):
        if not hero or not main_tower or not enemy_tower:
            return 0.0
        start = self._position(main_tower)
        end = self._position(enemy_tower)
        point = self._position(hero)
        vx, vz = end[0] - start[0], end[1] - start[1]
        lane_len_sq = max(vx * vx + vz * vz, 1.0)
        return max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vz) / lane_len_sq))

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

    def _get_tower(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") == camp and int(self._value(npc, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return npc
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

    def _safe_to_push(self, main_hero, enemy_hero, soldiers, enemy_tower):
        return (
            self._has_ally_minion_under_enemy_tower(soldiers, enemy_tower)
            and self._hp_ratio(main_hero) >= 0.30
            and not self._enemy_close_dangerous(main_hero, enemy_hero)
        )

    def _enemy_close_dangerous(self, main_hero, enemy_hero):
        if not main_hero or not enemy_hero or self._hp_ratio(enemy_hero) <= 0.0:
            return False
        return (
            self._distance(self._position(main_hero), self._position(enemy_hero)) <= 12000.0
            and self._hp_ratio(enemy_hero) > 0.25
        )

    def _has_summoner_real_cmd(self, hero):
        for skill_cmd in self._real_skill_cmds(hero):
            skill_id = self._int_field(skill_cmd, "skillID", self._int_field(skill_cmd, "skill_id", 0))
            slot_type = self._int_field(skill_cmd, "slotType", self._int_field(skill_cmd, "slot_type", -1))
            if skill_id == GameConfig.FIXED_SUMMONER_SKILL_ID or slot_type == GameConfig.BUTTON_SUMMONER:
                return True
        return False

    def _has_berserk_buff(self, hero):
        return bool(collect_buff_config_ids(hero) & BERSERK_BUFF_IDS)

    def _has_skill_real_cmd(self, hero, skill_no):
        for skill_cmd in self._real_skill_cmds(hero):
            slot_type = self._int_field(skill_cmd, "slotType", self._int_field(skill_cmd, "slot_type", -1))
            skill_id = self._int_field(skill_cmd, "skillID", self._int_field(skill_cmd, "skill_id", 0))
            if slot_type == skill_no or GameConfig.skill_no_from_id(skill_id) == skill_no:
                return True
        return False

    def _skill_state_features(self, hero):
        features = {
            "skill1_cd_ratio": 0.0,
            "skill2_cd_ratio": 0.0,
            "skill3_cd_ratio": 0.0,
            "summoner_cd_ratio": 0.0,
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
            prefix = self._slot_feature_prefix(slot, index)
            if not prefix:
                continue
            cooldown = self._value_any(slot, ("cooldown", "cd", "coolDown", "cooldownMs"))
            max_cooldown = self._value_any(
                slot, ("cooldown_max", "maxCooldown", "max_cooldown", "cooldownMax")
            )
            features[f"{prefix}_cd_ratio"] = max(0.0, min(1.0, cooldown / max_cooldown)) if max_cooldown > 0 else 0.0
        return features

    def _slot_feature_prefix(self, slot, index):
        slot_type = self._int_field(slot, "slotType", self._int_field(slot, "slot_type", -1))
        skill_id = self._int_field(
            slot,
            "skillID",
            self._int_field(slot, "skill_id", self._int_field(slot, "configId", self._int_field(slot, "config_id", 0))),
        )
        if slot_type in (1, 2, 3):
            return f"skill{slot_type}"
        if slot_type == GameConfig.BUTTON_SUMMONER or skill_id == GameConfig.FIXED_SUMMONER_SKILL_ID:
            return "summoner"
        skill_no = GameConfig.skill_no_from_id(skill_id)
        if skill_no is not None:
            return f"skill{skill_no}"
        if index in (0, 1, 2):
            return f"skill{index + 1}"
        if index == 3:
            return "summoner"
        return None

    def _real_skill_cmds(self, hero):
        if not isinstance(hero, dict):
            return []
        skill_cmds = []
        for cmd in hero.get("real_cmd", []) or []:
            if not isinstance(cmd, dict):
                continue
            for key in ("obj_skill", "dir_skill", "pos_skill"):
                skill_cmd = cmd.get(key)
                if isinstance(skill_cmd, dict):
                    skill_cmds.append(skill_cmd)
        return skill_cmds

    def _legal_button(self, observation, button):
        legal_action = observation.get("legal_action", [])
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if not isinstance(legal_action, (list, tuple)) or len(legal_action) <= button:
            return False
        try:
            return float(legal_action[button]) > 0.5
        except (TypeError, ValueError):
            return False

    def _target_allowed_for_button(self, observation, button, target):
        legal_action = observation.get("legal_action", [])
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if not isinstance(legal_action, (list, tuple)):
            return False

        sub_mask = observation.get("sub_action_mask", {}) or {}
        button_mask = sub_mask.get(str(button), [])
        if hasattr(button_mask, "tolist"):
            button_mask = button_mask.tolist()
        if not isinstance(button_mask, (list, tuple)):
            return False
        if len(button_mask) >= len(Config.LABEL_SIZE_LIST) and float(button_mask[-1] or 0.0) <= 0.5:
            return True

        split_points = [sum(Config.LEGAL_ACTION_SIZE_LIST[: index + 1]) for index in range(len(Config.LEGAL_ACTION_SIZE_LIST))]
        target_start = split_points[-2]
        target_end = split_points[-1]
        if len(legal_action) < target_end:
            return False
        try:
            target_matrix = np.asarray(legal_action[target_start:target_end]).reshape(
                Config.LEGAL_ACTION_SIZE_LIST[0],
                Config.LABEL_SIZE_LIST[-1],
            )
            return float(target_matrix[int(button)][int(target)]) > 0.5
        except (TypeError, ValueError, IndexError):
            return False

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

    def _skill_totals(self, hero):
        totals = {"hit_total": 0.0, "skill2_hit_total": 0.0}
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
            hit = self._value_any(slot, ("hitHeroTimes", "hit_hero_times"))
            totals["hit_total"] += hit
            if self._skill_no_from_slot(slot, index) == 2:
                totals["skill2_hit_total"] += hit
        return totals

    def _distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _position(self, unit):
        if not unit:
            return 0.0, 0.0
        location = unit.get("location", {})
        return float(location.get("x", 0.0) or 0.0), float(location.get("z", 0.0) or 0.0)

    def _value(self, unit, key):
        if not unit:
            return 0.0
        return float(unit.get(key, 0.0) or 0.0)

    def _resource(self, unit, key):
        if key == "money":
            return self._value_any(unit, ("money", "gold", "money_cnt"))
        return self._value(unit, key)

    def _value_any(self, unit, keys):
        for key in keys:
            value = self._value(unit, key)
            if value:
                return value
        return 0.0

    def reset_agents(self, observation, opponent_agent=None):
        opponent_agent = str(opponent_agent or self.env_conf_manager.get_opponent_agent())
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info("Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])
