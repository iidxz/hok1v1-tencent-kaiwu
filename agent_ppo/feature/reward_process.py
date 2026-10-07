#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from agent_ppo.conf.conf import GameConfig
from agent_ppo.feature.buff_constants import (
    BERSERK_BUFF_IDS,
    CAKE_LOCATIONS_BY_CAMP,
    RIVER_SPIRIT_CONFIG_ID,
    collect_buff_config_ids,
    has_luban_any_buff,
    has_luban_output_buff,
)


class GameRewardManager:
    def __init__(self, main_hero_player_id):
        self.main_hero_player_id = main_hero_player_id
        self.prev_stats = None
        self.reward_weights = GameConfig.REWARD_WEIGHT_DICT
        self.seen_last_hit_death_events = set()
        self.skill_recent_frames = 0

    def result(self, frame_data, action=None):
        cur_stats = self._extract_stats(frame_data, action)
        if self.prev_stats is None:
            self.prev_stats = cur_stats
            return self._zero_reward()

        reward_items = self._compute_reward_items(self.prev_stats, cur_stats)
        reward_sum = 0.0
        for name, value in reward_items.items():
            reward_sum += value * self.reward_weights.get(name, 0.0)

        self.prev_stats = cur_stats
        reward_items["reward_sum"] = self._clip(reward_sum, -100.0, 100.0)
        reward_items.update(self._legacy_aliases(reward_items))
        reward_items.update({f"{name}_weight": weight for name, weight in self.reward_weights.items()})
        reward_items["reward_stage"] = GameConfig.REWARD_STAGE
        reward_items["enable_tower_danger_reward"] = float(GameConfig.ENABLE_TOWER_DANGER_REWARD)
        reward_items["enable_push_assist_reward"] = float(GameConfig.ENABLE_PUSH_ASSIST_REWARD)
        reward_items["enable_lane_follow_reward"] = float(GameConfig.ENABLE_LANE_FOLLOW_REWARD)
        reward_items["enable_finish_reward"] = float(GameConfig.ENABLE_FINISH_REWARD)
        reward_items["enable_skill_reward"] = float(GameConfig.ENABLE_SKILL_REWARD)
        reward_items["enable_combo_reward"] = float(GameConfig.ENABLE_COMBO_REWARD)
        reward_items["enable_skill_follow_reward"] = float(GameConfig.ENABLE_SKILL_FOLLOW_REWARD)
        reward_items["enable_skill_bad_reward"] = float(GameConfig.ENABLE_SKILL_BAD_REWARD)
        reward_items["enable_berserk_reward"] = float(GameConfig.ENABLE_BERSERK_REWARD)
        reward_items["enable_push_retreat_reward"] = float(GameConfig.ENABLE_PUSH_RETREAT_REWARD)
        reward_items["enable_buff_reward"] = float(GameConfig.ENABLE_BUFF_REWARD)
        return reward_items

    def _zero_reward(self):
        reward_items = {name: 0.0 for name in self.reward_weights}
        reward_items["reward_sum"] = 0.0
        reward_items.update(self._legacy_aliases(reward_items))
        reward_items.update({f"{name}_weight": weight for name, weight in self.reward_weights.items()})
        reward_items["reward_stage"] = GameConfig.REWARD_STAGE
        reward_items["enable_tower_danger_reward"] = float(GameConfig.ENABLE_TOWER_DANGER_REWARD)
        reward_items["enable_push_assist_reward"] = float(GameConfig.ENABLE_PUSH_ASSIST_REWARD)
        reward_items["enable_lane_follow_reward"] = float(GameConfig.ENABLE_LANE_FOLLOW_REWARD)
        reward_items["enable_finish_reward"] = float(GameConfig.ENABLE_FINISH_REWARD)
        reward_items["enable_skill_reward"] = float(GameConfig.ENABLE_SKILL_REWARD)
        reward_items["enable_combo_reward"] = float(GameConfig.ENABLE_COMBO_REWARD)
        reward_items["enable_skill_follow_reward"] = float(GameConfig.ENABLE_SKILL_FOLLOW_REWARD)
        reward_items["enable_skill_bad_reward"] = float(GameConfig.ENABLE_SKILL_BAD_REWARD)
        reward_items["enable_berserk_reward"] = float(GameConfig.ENABLE_BERSERK_REWARD)
        reward_items["enable_push_retreat_reward"] = float(GameConfig.ENABLE_PUSH_RETREAT_REWARD)
        reward_items["enable_buff_reward"] = float(GameConfig.ENABLE_BUFF_REWARD)
        return reward_items

    def _compute_reward_items(self, prev, cur):
        enemy_hp_drop = prev["enemy_hp_ratio"] - cur["enemy_hp_ratio"]
        self_hp_drop = prev["self_hp_ratio"] - cur["self_hp_ratio"]
        enemy_tower_drop = prev["enemy_tower_hp_ratio"] - cur["enemy_tower_hp_ratio"]
        self_tower_drop = prev["self_tower_hp_ratio"] - cur["self_tower_hp_ratio"]
        skill_recent = self.skill_recent_frames > 0

        reward_items = {
            "reward_hp": self._clip(enemy_hp_drop - self_hp_drop, -1.0, 1.0),
            "reward_tower": self._clip(enemy_tower_drop - self_tower_drop, -1.0, 1.0),
            "reward_gold": self._clip((cur["self_money"] - prev["self_money"]) - (cur["enemy_money"] - prev["enemy_money"]), -200.0, 200.0),
            "reward_exp": self._clip((cur["self_exp"] - prev["self_exp"]) - (cur["enemy_exp"] - prev["enemy_exp"]), -200.0, 200.0),
            "reward_last_hit": self._clip(cur["last_hit_delta"], 0.0, 4.0),
            "reward_death": self._clip(cur["self_dead_cnt"] - prev["self_dead_cnt"], 0.0, 1.0),
            "reward_kill": self._clip(cur["self_kill_cnt"] - prev["self_kill_cnt"], 0.0, 1.0),
            "reward_forward": self._clip(cur["lane_progress"] - prev["lane_progress"], -1.0, 1.0),
            "reward_win": self._win_reward(cur),
            "reward_tower_danger": 0.0,
            "reward_push_assist": 0.0,
            "reward_miss_push_window": 0.0,
            "reward_lane_follow": 0.0,
            "reward_finish": 0.0,
            "reward_hero_damage": 0.0,
            "reward_combat_kite": 0.0,
            "reward_combat_standstill": 0.0,
            "reward_attack_move": 0.0,
            "reward_attack_standstill": 0.0,
            "reward_low_hp_cake": 0.0,
            "reward_low_hp_heal": 0.0,
            "reward_low_hp_idle": 0.0,
            "reward_follow_minion_after_clear": 0.0,
            "reward_idle_after_clear": 0.0,
            "reward_river_monster": 0.0,
            "reward_idle_when_monster": 0.0,
            "reward_attack_hero_in_range": 0.0,
            "reward_miss_hero_in_range": 0.0,
            "reward_skill_hit_hero": 0.0,
            "reward_skill_damage": 0.0,
            "reward_skill2_ready_unused": 0.0,
            "reward_skill3_ready_unused": 0.0,
            "reward_luban_buff_damage": 0.0,
            "reward_skill2_finish": 0.0,
            "reward_summoner_finish": 0.0,
            "reward_attack_hero_under_tower": 0.0,
            "reward_retreat_from_tower": 0.0,
            "reward_stay_under_tower": 0.0,
            "reward_base_stuck": 0.0,
            "reward_early_forward": 0.0,
            "reward_skill_follow_attack": 0.0,
            "reward_skill_follow_tower": 0.0,
            "reward_bad_skill2": 0.0,
            "reward_bad_skill3": 0.0,
            "reward_bad_summoner": 0.0,
            "reward_push_window_target_tower": 0.0,
            "reward_push_window_wrong_hero": 0.0,
            "reward_retreat_to_own_tower": 0.0,
            "reward_bad_attack_low_hp": 0.0,
            "reward_safe_recall": 0.0,
            "reward_skill1_good_cast": 0.0,
            "reward_skill2_good_cast": 0.0,
            "reward_skill3_good_cast": 0.0,
            "reward_skill_after_attack": 0.0,
            "reward_berserk_attack_follow": 0.0,
            "reward_tower_poke_hero": 0.0,
            "reward_tower_poke_dive": 0.0,
            "reward_defend_tower_clear": 0.0,
            "reward_defend_tower_ignore": 0.0,
        }

        if GameConfig.ENABLE_TOWER_DANGER_REWARD:
            reward_items["reward_tower_danger"] = self._tower_danger_reward(cur)
        if GameConfig.ENABLE_PUSH_ASSIST_REWARD:
            reward_items["reward_push_assist"] = self._push_assist_reward(prev, cur)
            reward_items["reward_miss_push_window"] = self._miss_push_window_reward(cur)
        if GameConfig.ENABLE_LANE_FOLLOW_REWARD:
            reward_items["reward_lane_follow"] = self._lane_follow_reward(prev, cur)
        if GameConfig.ENABLE_FINISH_REWARD:
            reward_items["reward_finish"] = self._finish_reward(cur)
            reward_items["reward_skill2_finish"] = self._skill2_finish_reward(prev, cur, enemy_hp_drop)
            reward_items["reward_summoner_finish"] = self._summoner_finish_reward(cur)
        if GameConfig.REWARD_STAGE == "D":
            reward_items["reward_hero_damage"] = self._hero_damage_reward(enemy_hp_drop, cur)
            reward_items.update(self._combat_movement_rewards(prev, cur))
            reward_items.update(self._attack_movement_rewards(prev, cur))
            reward_items.update(self._low_hp_recovery_rewards(prev, cur))
            reward_items.update(self._map_pressure_rewards(prev, cur))
            reward_items.update(self._hero_target_rewards(cur))
            reward_items.update(self._skill_rewards(prev, cur, enemy_hp_drop))
            reward_items.update(self._skill_pressure_rewards(cur))
            reward_items.update(self._tower_poke_rewards(cur, enemy_hp_drop))
            reward_items.update(self._defend_tower_rewards(prev, cur))
            reward_items["reward_attack_hero_under_tower"] = self._attack_hero_under_tower_reward(cur)
            if GameConfig.ENABLE_PUSH_RETREAT_REWARD:
                reward_items.update(self._tower_retreat_rewards(prev, cur, self_hp_drop))
                reward_items.update(self._push_window_target_rewards(cur))
                reward_items.update(self._retreat_window_rewards(prev, cur))
            if GameConfig.ENABLE_COMBO_REWARD:
                reward_items.update(self._early_start_rewards(prev, cur))
            if GameConfig.ENABLE_SKILL_FOLLOW_REWARD:
                reward_items.update(self._skill_follow_rewards(prev, cur, enemy_hp_drop, skill_recent))
            if GameConfig.ENABLE_SKILL_BAD_REWARD:
                reward_items["reward_bad_skill2"] = self._bad_skill2_reward(prev, cur, enemy_hp_drop)
                reward_items["reward_bad_skill3"] = self._bad_skill3_reward(prev, cur, enemy_hp_drop)
            reward_items["reward_bad_summoner"] = self._bad_summoner_reward(cur)
            reward_items.update(self._combo_rewards(cur, skill_recent))
            if GameConfig.ENABLE_BUFF_REWARD:
                reward_items["reward_luban_buff_damage"] = self._luban_buff_damage_reward(enemy_hp_drop, cur)
        self._update_skill_recent(prev, cur)
        return reward_items

    def _extract_stats(self, frame_data, action=None):
        main_hero, enemy_hero = self._split_heroes(frame_data)
        main_camp = main_hero.get("camp", -1) if main_hero else -1
        main_tower = self._get_tower(frame_data, main_camp)
        enemy_tower = self._get_enemy_tower(frame_data, main_camp)
        my_soldiers = self._soldiers(frame_data, main_camp)
        enemy_soldiers = self._enemy_soldiers(frame_data, main_camp)
        neutral_monsters = self._neutral_monsters(frame_data)
        nearest_neutral_monster = self._nearest_unit(main_hero, neutral_monsters)
        last_hit_delta = self._last_hit_delta_from_frame_action(frame_data, main_hero)
        action_button = self._action_button(action)
        action_target = self._action_target(action)
        skill_totals = self._skill_totals(main_hero)
        skill_cd_ratios = self._skill_cd_ratios(main_hero)
        self_hp_ratio = self._hp_ratio(main_hero)
        enemy_hp_ratio = self._hp_ratio(enemy_hero)
        self_in_enemy_tower_range = self._in_tower_range(main_hero, enemy_tower)
        self_under_own_tower = self._in_tower_range(main_hero, main_tower)
        enemy_visible = enemy_hero is not None and enemy_hp_ratio > 0.0
        enemy_distance = self._distance_to_unit(main_hero, enemy_hero)
        enemy_close = enemy_visible and enemy_distance <= 12000.0
        has_ally_minion_under_enemy_tower = self._has_ally_minion_under_enemy_tower(my_soldiers, enemy_tower)
        enemy_minion_under_own_tower = self._has_ally_minion_under_enemy_tower(enemy_soldiers, main_tower)
        enemy_tower_alive = enemy_tower is not None and self._value(enemy_tower, "hp") > 0.0
        hp_advantage = self_hp_ratio - enemy_hp_ratio
        enemy_tower_distance = self._distance_to_unit(main_hero, enemy_tower)
        cake_distance = self._distance_to_point(main_hero, CAKE_LOCATIONS_BY_CAMP.get(main_camp))
        neutral_monster_distance = self._distance_to_unit(main_hero, nearest_neutral_monster)
        self_attack_range = self._value(main_hero, "attack_range") or 7000.0
        enemy_in_attack_range = enemy_visible and enemy_distance <= self_attack_range + 800.0
        can_fight_hero = enemy_in_attack_range and not self_in_enemy_tower_range and self_hp_ratio > 0.35
        enemy_close_dangerous = self._enemy_close_dangerous(main_hero, enemy_hero)
        strict_push_window = has_ally_minion_under_enemy_tower and self_hp_ratio > 0.35 and enemy_tower_alive
        near_push_window = (
            enemy_tower_alive
            and enemy_tower_distance <= self_attack_range + 1200.0
            and self_hp_ratio > 0.45
            and not enemy_close_dangerous
        )
        push_window = strict_push_window or near_push_window
        retreat_window = self_hp_ratio < 0.30 or (hp_advantage < -0.25 and enemy_close)
        safe_to_recall = self_hp_ratio < 0.25 and (enemy_distance > 7000.0 or self_under_own_tower)
        lane_clear_window = not enemy_soldiers and not enemy_close_dangerous and not push_window
        river_monster_window = (
            lane_clear_window
            and nearest_neutral_monster is not None
            and self_hp_ratio > 0.35
            and neutral_monster_distance <= 18000.0
        )

        return {
            "self_hp_ratio": self_hp_ratio,
            "self_hero_id": self._int_field(main_hero, "config_id", 0),
            "enemy_hp_ratio": enemy_hp_ratio,
            "enemy_under_own_tower": self._in_tower_range(enemy_hero, enemy_tower),
            "self_tower_hp_ratio": self._hp_ratio(main_tower),
            "enemy_tower_hp_ratio": self._hp_ratio(enemy_tower),
            "self_money": self._resource(main_hero, "money"),
            "enemy_money": self._resource(enemy_hero, "money"),
            "self_exp": self._resource(main_hero, "exp"),
            "enemy_exp": self._resource(enemy_hero, "exp"),
            "self_dead_cnt": self._value(main_hero, "dead_cnt"),
            "self_kill_cnt": self._value(main_hero, "kill_cnt"),
            "lane_progress": self._lane_progress(main_hero, main_tower, enemy_tower),
            "frame_no": self._value(frame_data, "frame_no"),
            "last_hit_delta": last_hit_delta,
            "self_tower_dead": main_tower is not None and self._value(main_tower, "hp") <= 0,
            "enemy_tower_dead": enemy_tower is not None and self._value(enemy_tower, "hp") <= 0,
            "self_in_enemy_tower_range": self_in_enemy_tower_range,
            "self_under_own_tower": self_under_own_tower,
            "self_low_hp": self_hp_ratio < 0.35,
            "has_ally_minion_under_enemy_tower": has_ally_minion_under_enemy_tower,
            "enemy_low_hp": enemy_hp_ratio < 0.40,
            "enemy_tower_alive": enemy_tower_alive,
            "strict_push_window": strict_push_window,
            "near_push_window": near_push_window,
            "push_window": push_window,
            "retreat_window": retreat_window,
            "safe_to_recall": safe_to_recall,
            "enemy_distance": enemy_distance,
            "enemy_close": enemy_close,
            "enemy_in_attack_range": enemy_in_attack_range,
            "enemy_soldier_visible": bool(enemy_soldiers),
            "enemy_soldier_count": len(enemy_soldiers),
            "enemy_minion_under_own_tower": enemy_minion_under_own_tower,
            "safe_to_push": self._safe_to_push(main_hero, enemy_hero, my_soldiers, enemy_tower),
            "ally_minion_ahead": self._ally_minion_ahead(main_hero, my_soldiers, main_tower, enemy_tower),
            "enemy_close_dangerous": enemy_close_dangerous,
            "enemy_visible": enemy_visible,
            "can_fight_hero": can_fight_hero,
            "enemy_tower_distance": enemy_tower_distance,
            "enemy_tower_low_hp": self._hp_ratio(enemy_tower) < 0.50 and self._hp_ratio(enemy_tower) > 0.0,
            "self_tower_distance": self._distance_to_unit(main_hero, main_tower),
            "cake_distance": cake_distance,
            "neutral_monster_distance": neutral_monster_distance,
            "neutral_monster_visible": nearest_neutral_monster is not None,
            "lane_clear_window": lane_clear_window,
            "river_monster_window": river_monster_window,
            "has_luban_any_buff": has_luban_any_buff(main_hero),
            "has_luban_output_buff": has_luban_output_buff(main_hero),
            "berserk_active": self._has_berserk_buff(main_hero),
            "skill_used_total": skill_totals["used_total"],
            "skill_hit_total": skill_totals["hit_total"],
            "skill2_used_total": skill_totals["skill2_used_total"],
            "skill2_hit_total": skill_totals["skill2_hit_total"],
            "skill2_cd_ratio": skill_cd_ratios[2],
            "skill3_cd_ratio": skill_cd_ratios[3],
            "skill_success_used": skill_totals["succ_used_total"] > 0.0,
            "action_button": action_button,
            "action_target": action_target,
            "position": self._position(main_hero),
        }

    def _legacy_aliases(self, reward_items):
        return {
            "hp_point": reward_items.get("reward_hp", 0.0),
            "tower_hp_point": reward_items.get("reward_tower", 0.0),
            "gold_point": reward_items.get("reward_gold", 0.0),
            "exp_point": reward_items.get("reward_exp", 0.0),
            "last_hit_point": reward_items.get("reward_last_hit", 0.0),
            "death_point": reward_items.get("reward_death", 0.0),
            "kill_point": reward_items.get("reward_kill", 0.0),
            "forward": reward_items.get("reward_forward", 0.0),
            "push_assist": reward_items.get("reward_push_assist", 0.0),
            "miss_push_window": reward_items.get("reward_miss_push_window", 0.0),
            "lane_follow": reward_items.get("reward_lane_follow", 0.0),
            "finish": reward_items.get("reward_finish", 0.0),
            "hero_damage": reward_items.get("reward_hero_damage", 0.0),
            "combat_kite": reward_items.get("reward_combat_kite", 0.0),
            "combat_standstill": reward_items.get("reward_combat_standstill", 0.0),
            "attack_move": reward_items.get("reward_attack_move", 0.0),
            "attack_standstill": reward_items.get("reward_attack_standstill", 0.0),
            "low_hp_cake": reward_items.get("reward_low_hp_cake", 0.0),
            "low_hp_heal": reward_items.get("reward_low_hp_heal", 0.0),
            "low_hp_idle": reward_items.get("reward_low_hp_idle", 0.0),
            "follow_minion_after_clear": reward_items.get("reward_follow_minion_after_clear", 0.0),
            "idle_after_clear": reward_items.get("reward_idle_after_clear", 0.0),
            "river_monster": reward_items.get("reward_river_monster", 0.0),
            "idle_when_monster": reward_items.get("reward_idle_when_monster", 0.0),
            "attack_hero_in_range": reward_items.get("reward_attack_hero_in_range", 0.0),
            "miss_hero_in_range": reward_items.get("reward_miss_hero_in_range", 0.0),
            "skill_hit_hero": reward_items.get("reward_skill_hit_hero", 0.0),
            "skill_damage": reward_items.get("reward_skill_damage", 0.0),
            "skill2_ready_unused": reward_items.get("reward_skill2_ready_unused", 0.0),
            "skill3_ready_unused": reward_items.get("reward_skill3_ready_unused", 0.0),
            "luban_buff_damage": reward_items.get("reward_luban_buff_damage", 0.0),
            "skill2_finish": reward_items.get("reward_skill2_finish", 0.0),
            "summoner_finish": reward_items.get("reward_summoner_finish", 0.0),
            "base_stuck": reward_items.get("reward_base_stuck", 0.0),
            "early_forward": reward_items.get("reward_early_forward", 0.0),
            "skill_follow_attack": reward_items.get("reward_skill_follow_attack", 0.0),
            "skill_follow_tower": reward_items.get("reward_skill_follow_tower", 0.0),
            "bad_skill2": reward_items.get("reward_bad_skill2", 0.0),
            "bad_skill3": reward_items.get("reward_bad_skill3", 0.0),
            "bad_summoner": reward_items.get("reward_bad_summoner", 0.0),
            "push_window_target_tower": reward_items.get("reward_push_window_target_tower", 0.0),
            "push_window_wrong_hero": reward_items.get("reward_push_window_wrong_hero", 0.0),
            "retreat_to_own_tower": reward_items.get("reward_retreat_to_own_tower", 0.0),
            "bad_attack_low_hp": reward_items.get("reward_bad_attack_low_hp", 0.0),
            "safe_recall": reward_items.get("reward_safe_recall", 0.0),
            "skill1_good_cast": reward_items.get("reward_skill1_good_cast", 0.0),
            "skill2_good_cast": reward_items.get("reward_skill2_good_cast", 0.0),
            "skill3_good_cast": reward_items.get("reward_skill3_good_cast", 0.0),
            "skill_after_attack": reward_items.get("reward_skill_after_attack", 0.0),
            "berserk_attack_follow": reward_items.get("reward_berserk_attack_follow", 0.0),
            "tower_poke_hero": reward_items.get("reward_tower_poke_hero", 0.0),
            "tower_poke_dive": reward_items.get("reward_tower_poke_dive", 0.0),
            "defend_tower_clear": reward_items.get("reward_defend_tower_clear", 0.0),
            "defend_tower_ignore": reward_items.get("reward_defend_tower_ignore", 0.0),
        }

    def _win_reward(self, cur):
        if cur["enemy_tower_dead"] and not cur["self_tower_dead"]:
            return 1.0
        if cur["self_tower_dead"] and not cur["enemy_tower_dead"]:
            return -1.0
        return 0.0

    def _tower_danger_reward(self, cur):
        if cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"]:
            return -0.04
        return 0.0

    def _push_assist_reward(self, prev, cur):
        reward = 0.0
        push_action = (
            cur["safe_to_push"]
            and cur["action_button"] == GameConfig.BUTTON_ATTACK
            and cur["action_target"] == GameConfig.TARGET_TOWER
        )
        enemy_tower_drop = prev["enemy_tower_hp_ratio"] - cur["enemy_tower_hp_ratio"]
        push_effect = cur["safe_to_push"] and enemy_tower_drop > 0.0001
        if push_action:
            reward += 0.04
        if push_effect:
            reward += 0.04
        return self._clip(reward, 0.0, 0.08)

    def _miss_push_window_reward(self, cur):
        if (
            cur["safe_to_push"]
            and cur["action_button"] == GameConfig.BUTTON_ATTACK
            and cur["action_target"] in GameConfig.TARGET_SOLDIERS
        ):
            return -0.005
        return 0.0

    def _lane_follow_reward(self, prev, cur):
        moved_forward = cur["lane_progress"] > prev["lane_progress"] + 0.001
        if (
            cur["action_button"] == GameConfig.BUTTON_MOVE
            and moved_forward
            and cur["ally_minion_ahead"]
            and not cur["enemy_close_dangerous"]
        ):
            return 0.01
        return 0.0

    def _finish_reward(self, cur):
        if (
            cur["enemy_visible"]
            and cur["enemy_low_hp"]
            and cur["action_target"] == GameConfig.TARGET_ENEMY
        ):
            return 0.02
        return 0.0

    def _hero_damage_reward(self, enemy_hp_drop, cur):
        if cur["can_fight_hero"] and enemy_hp_drop > 0.0:
            return self._clip(enemy_hp_drop * 0.4, 0.0, 0.015)
        return 0.0

    def _combat_movement_rewards(self, prev, cur):
        moved_distance = math.dist(prev["position"], cur["position"])
        fighting = cur["can_fight_hero"] and cur["action_target"] == GameConfig.TARGET_ENEMY
        if not fighting:
            return {"reward_combat_kite": 0.0, "reward_combat_standstill": 0.0}
        if moved_distance >= 300.0 and cur["action_button"] in (GameConfig.BUTTON_MOVE, GameConfig.BUTTON_ATTACK, *GameConfig.SKILL_BUTTONS):
            return {"reward_combat_kite": 0.018, "reward_combat_standstill": 0.0}
        if moved_distance < 220.0 and cur["enemy_distance"] <= 10000.0:
            return {"reward_combat_kite": 0.0, "reward_combat_standstill": -0.025}
        return {"reward_combat_kite": 0.0, "reward_combat_standstill": 0.0}

    def _attack_movement_rewards(self, prev, cur):
        moved_distance = math.dist(prev["position"], cur["position"])
        attack_target = (
            cur["action_target"] == GameConfig.TARGET_ENEMY
            or cur["action_target"] in GameConfig.TARGET_SOLDIERS
            or cur["action_target"] == GameConfig.TARGET_MONSTER
            or cur["action_target"] == GameConfig.TARGET_TOWER
        )
        attack_action = cur["action_button"] == GameConfig.BUTTON_ATTACK or cur["action_button"] in GameConfig.SKILL_BUTTONS
        if not attack_action or not attack_target:
            return {"reward_attack_move": 0.0, "reward_attack_standstill": 0.0}
        if cur["self_in_enemy_tower_range"]:
            return {"reward_attack_move": 0.0, "reward_attack_standstill": 0.0}
        if moved_distance >= 180.0:
            return {"reward_attack_move": 0.01, "reward_attack_standstill": 0.0}
        if moved_distance < 180.0 and cur["action_target"] != GameConfig.TARGET_TOWER:
            return {"reward_attack_move": 0.0, "reward_attack_standstill": -0.014}
        return {"reward_attack_move": 0.0, "reward_attack_standstill": 0.0}

    def _low_hp_recovery_rewards(self, prev, cur):
        reward_cake = 0.0
        reward_heal = 0.0
        reward_idle = 0.0
        hp_recovered = cur["self_hp_ratio"] > prev["self_hp_ratio"] + 0.01
        moved_to_cake = cur["cake_distance"] + 150.0 < prev["cake_distance"]
        low_hp = prev["self_hp_ratio"] < 0.35 or cur["self_hp_ratio"] < 0.35
        unsafe_low_hp = low_hp and not cur["enemy_low_hp"]
        if low_hp and moved_to_cake:
            reward_cake = 0.025
        if low_hp and (hp_recovered or cur["action_button"] in (GameConfig.BUTTON_HEAL, GameConfig.BUTTON_RECALL)):
            reward_heal = 0.03
        if (
            unsafe_low_hp
            and cur["action_button"] in (GameConfig.BUTTON_NONE, GameConfig.BUTTON_NOOP, GameConfig.BUTTON_ATTACK)
            and not moved_to_cake
            and not hp_recovered
        ):
            reward_idle = -0.025
        return {
            "reward_low_hp_cake": reward_cake,
            "reward_low_hp_heal": reward_heal,
            "reward_low_hp_idle": reward_idle,
        }

    def _map_pressure_rewards(self, prev, cur):
        reward_follow = 0.0
        reward_idle_lane = 0.0
        reward_monster = 0.0
        reward_idle_monster = 0.0
        moved_forward = cur["lane_progress"] > prev["lane_progress"] + 0.001
        idle_or_noop = cur["action_button"] in (GameConfig.BUTTON_NONE, GameConfig.BUTTON_NOOP)
        low_movement = math.dist(prev["position"], cur["position"]) < 250.0
        if cur["lane_clear_window"] and cur["ally_minion_ahead"] and moved_forward:
            reward_follow = 0.03
        if cur["lane_clear_window"] and (idle_or_noop or low_movement) and not cur["river_monster_window"]:
            reward_idle_lane = -0.035
        if cur["river_monster_window"] and cur["action_button"] == GameConfig.BUTTON_ATTACK and cur["action_target"] == GameConfig.TARGET_MONSTER:
            reward_monster = 0.045
        if cur["river_monster_window"] and (idle_or_noop or low_movement):
            reward_idle_monster = -0.035
        return {
            "reward_follow_minion_after_clear": reward_follow,
            "reward_idle_after_clear": reward_idle_lane,
            "reward_river_monster": reward_monster,
            "reward_idle_when_monster": reward_idle_monster,
        }

    def _hero_target_rewards(self, cur):
        reward_attack = 0.0
        reward_miss = 0.0
        if (
            cur["can_fight_hero"]
            and not cur["push_window"]
            and cur["action_button"] == GameConfig.BUTTON_ATTACK
        ):
            if cur["action_target"] == GameConfig.TARGET_ENEMY:
                reward_attack = 0.035
            else:
                reward_miss = -0.025
        return {
            "reward_attack_hero_in_range": reward_attack,
            "reward_miss_hero_in_range": reward_miss,
        }

    def _skill_rewards(self, prev, cur, enemy_hp_drop):
        reward_skill_hit_hero = 0.0
        reward_skill_damage = 0.0
        skill_used = cur["action_button"] in GameConfig.SKILL_BUTTONS or cur["skill_used_total"] > prev["skill_used_total"]
        skill_hit_hero = cur["skill_hit_total"] > prev["skill_hit_total"]
        if skill_used and enemy_hp_drop > 0.001:
            skill_hit_hero = True
        if GameConfig.ENABLE_SKILL_REWARD and cur["can_fight_hero"] and skill_used and skill_hit_hero:
            reward_skill_hit_hero = 0.018
        if GameConfig.ENABLE_SKILL_REWARD and cur["can_fight_hero"] and skill_used and enemy_hp_drop > 0.0:
            reward_skill_damage = self._clip(enemy_hp_drop * 0.45, 0.0, 0.025)
        return {
            "reward_skill_hit_hero": reward_skill_hit_hero,
            "reward_skill_damage": reward_skill_damage,
        }

    def _skill_pressure_rewards(self, cur):
        """
        ready_unused 惩罚按英雄技能拆分：
        - 鲁班 skill2：残血收割/贴脸自保，ready 时应该用
        - 鲁班 skill3：清兵/压塔用，不要因为没打 enemy 惩罚
        - 狄仁杰 skill2：防守技能，不要因为没打 enemy 惩罚
        - 狄仁杰 skill3：控制技能，ready 时应该用
        """
        empty = {
            "reward_skill2_ready_unused": 0.0,
            "reward_skill3_ready_unused": 0.0,
        }
        if not cur["can_fight_hero"]:
            return empty
        skill2_ready = cur["skill2_cd_ratio"] <= 0.05
        skill3_ready = cur["skill3_cd_ratio"] <= 0.05
        basic_attack = cur["action_button"] == GameConfig.BUTTON_ATTACK
        enemy_low_or_close = cur["enemy_hp_ratio"] <= 0.55 or cur["enemy_distance"] <= 9000.0
        
        if cur["self_hero_id"] == 112:
            # 鲁班 skill2：残血收割/贴脸自保，ready 且敌人低血量/靠近时应该用
            luban_s2_unused = (
                skill2_ready
                and basic_attack
                and cur["action_target"] == GameConfig.TARGET_ENEMY
                and (cur["enemy_low_hp"] or cur["enemy_close"])  # 只在残血/贴脸时惩罚
            )
            # 鲁班 skill3：清兵/压塔用，不要因为没打 enemy 惩罚
            # 只有在有兵线/推塔窗口时才考虑惩罚
            luban_s3_unused = False  # 不惩罚鲁班3技能 ready 但没用
            return {
                "reward_skill2_ready_unused": -0.020 if luban_s2_unused else 0.0,
                "reward_skill3_ready_unused": -0.015 if luban_s3_unused else 0.0,
            }
        if cur["self_hero_id"] == 133:
            # 狄仁杰 skill2：防守技能，不要因为没打 enemy 惩罚
            # 狄仁杰 skill3：控制技能，ready 且敌人在合理距离时应该用
            direnjie_s3_unused = (
                skill3_ready
                and basic_attack
                and cur["action_target"] == GameConfig.TARGET_ENEMY
                and enemy_low_or_close
                and cur["self_hp_ratio"] > 0.45  # 自己血量不低
                and not cur["self_in_enemy_tower_range"]  # 不在敌塔危险区
            )
            return {
                "reward_skill2_ready_unused": 0.0,  # 狄仁杰2是防守技能，不惩罚
                "reward_skill3_ready_unused": -0.025 if direnjie_s3_unused else 0.0,
            }
        return {
            "reward_skill2_ready_unused": 0.0,
            "reward_skill3_ready_unused": 0.0,
        }

    def _tower_poke_rewards(self, cur, enemy_hp_drop):
        if not cur["has_ally_minion_under_enemy_tower"] or not cur["enemy_under_own_tower"]:
            return {
                "reward_tower_poke_hero": 0.0,
                "reward_tower_poke_dive": 0.0,
            }
        attack_enemy = (
            cur["action_target"] == GameConfig.TARGET_ENEMY
            and (cur["action_button"] == GameConfig.BUTTON_ATTACK or cur["action_button"] in GameConfig.SKILL_BUTTONS)
        )
        if not attack_enemy:
            return {
                "reward_tower_poke_hero": 0.0,
                "reward_tower_poke_dive": 0.0,
            }
        if cur["self_in_enemy_tower_range"]:
            return {
                "reward_tower_poke_hero": 0.0,
                "reward_tower_poke_dive": -0.12,
            }
        poke_reward = 0.06 if enemy_hp_drop > 0.0 else 0.025
        return {
            "reward_tower_poke_hero": poke_reward,
            "reward_tower_poke_dive": 0.0,
        }

    def _defend_tower_rewards(self, prev, cur):
        if not cur["enemy_minion_under_own_tower"]:
            return {
                "reward_defend_tower_clear": 0.0,
                "reward_defend_tower_ignore": 0.0,
            }
        soldier_count_drop = prev["enemy_soldier_count"] > cur["enemy_soldier_count"]
        attack_soldier = (
            cur["action_button"] == GameConfig.BUTTON_ATTACK
            and cur["action_target"] in GameConfig.TARGET_SOLDIERS
        )
        if attack_soldier or soldier_count_drop:
            return {
                "reward_defend_tower_clear": 0.06,
                "reward_defend_tower_ignore": 0.0,
            }
        ignore_action = (
            cur["action_button"] in (GameConfig.BUTTON_NONE, GameConfig.BUTTON_NOOP)
            or cur["action_target"] in (GameConfig.TARGET_ENEMY, GameConfig.TARGET_MONSTER, GameConfig.TARGET_TOWER)
        )
        return {
            "reward_defend_tower_clear": 0.0,
            "reward_defend_tower_ignore": -0.06 if ignore_action else 0.0,
        }

    def _luban_buff_damage_reward(self, enemy_hp_drop, cur):
        if cur["can_fight_hero"] and cur["has_luban_output_buff"] and enemy_hp_drop > 0.0:
            return 0.02
        return 0.0

    def _skill2_finish_reward(self, prev, cur, enemy_hp_drop):
        skill2_used = cur["action_button"] == GameConfig.BUTTON_SKILL_2
        skill2_hit = cur["skill2_hit_total"] > prev["skill2_hit_total"] or (skill2_used and enemy_hp_drop > 0.001)
        if cur["can_fight_hero"] and cur["enemy_hp_ratio"] < 0.35 and skill2_used and skill2_hit:
            return 0.03
        return 0.0

    def _summoner_finish_reward(self, cur):
        if GameConfig.FIXED_SUMMONER_SKILL_ID == 80110:
            return 0.0
        if (
            cur["can_fight_hero"]
            and cur["enemy_hp_ratio"] < 0.30
            and cur["action_button"] == GameConfig.BUTTON_SUMMONER
            and cur["action_target"] == GameConfig.TARGET_ENEMY
        ):
            return 0.03
        return 0.0

    def _attack_hero_under_tower_reward(self, cur):
        attack_or_skill = cur["action_button"] == GameConfig.BUTTON_ATTACK or cur["action_button"] in GameConfig.SKILL_BUTTONS
        if cur["self_in_enemy_tower_range"] and cur["action_target"] == GameConfig.TARGET_ENEMY and attack_or_skill:
            return -0.20
        return 0.0

    def _tower_retreat_rewards(self, prev, cur, self_hp_drop):
        reward_retreat = 0.0
        reward_stay = 0.0
        tower_hitting_self = cur["self_in_enemy_tower_range"] and self_hp_drop > 0.001
        moving_away = (
            cur["action_button"] == GameConfig.BUTTON_MOVE
            and cur["enemy_tower_distance"] > prev["enemy_tower_distance"] + 100.0
        )
        if tower_hitting_self and moving_away:
            reward_retreat = 0.03
        elif tower_hitting_self and not moving_away:
            reward_stay = -0.04
        return {
            "reward_retreat_from_tower": reward_retreat,
            "reward_stay_under_tower": reward_stay,
        }

    def _early_start_rewards(self, prev, cur):
        reward_base_stuck = 0.0
        reward_early_forward = 0.0
        early_game = cur["frame_no"] < 1200.0
        near_own_base = cur["lane_progress"] < 0.12
        moved_distance = math.dist(prev["position"], cur["position"])
        moved_forward = cur["lane_progress"] > prev["lane_progress"] + 0.001
        low_movement = moved_distance < 300.0
        if early_game and near_own_base and low_movement:
            reward_base_stuck = -0.01
        if early_game and cur["action_button"] == GameConfig.BUTTON_MOVE and moved_forward:
            reward_early_forward = 0.005
        return {
            "reward_base_stuck": reward_base_stuck,
            "reward_early_forward": reward_early_forward,
        }

    def _skill_follow_rewards(self, prev, cur, enemy_hp_drop, skill_recent):
        reward_follow_attack = 0.0
        reward_follow_tower = 0.0
        if (
            skill_recent
            and cur["action_button"] == GameConfig.BUTTON_ATTACK
            and cur["action_target"] == GameConfig.TARGET_ENEMY
            and enemy_hp_drop > 0.0
            and cur["self_hp_ratio"] > 0.40
            and cur["self_hp_ratio"] - cur["enemy_hp_ratio"] > -0.15
            and not cur["self_in_enemy_tower_range"]
        ):
            reward_follow_attack = 0.01
        if (
            skill_recent
            and cur["action_button"] == GameConfig.BUTTON_ATTACK
            and cur["action_target"] == GameConfig.TARGET_TOWER
            and cur["safe_to_push"]
        ):
            reward_follow_tower = 0.005
        return {
            "reward_skill_follow_attack": reward_follow_attack,
            "reward_skill_follow_tower": reward_follow_tower,
        }

    def _push_window_target_rewards(self, cur):
        reward_target_tower = 0.0
        reward_wrong_hero = 0.0
        if (
            cur["push_window"]
            and cur["action_button"] == GameConfig.BUTTON_ATTACK
            and cur["action_target"] == GameConfig.TARGET_TOWER
        ):
            reward_target_tower = 0.08
        if (
            cur["push_window"]
            and cur["action_target"] == GameConfig.TARGET_ENEMY
            and not cur["enemy_low_hp"]
        ):
            reward_wrong_hero = -0.08
        return {
            "reward_push_window_target_tower": reward_target_tower,
            "reward_push_window_wrong_hero": reward_wrong_hero,
        }

    def _retreat_window_rewards(self, prev, cur):
        reward_retreat = 0.0
        reward_bad_attack = 0.0
        reward_safe_recall = 0.0
        moving_back = (
            cur["action_button"] == GameConfig.BUTTON_MOVE
            and (
                cur["self_tower_distance"] < prev["self_tower_distance"] - 50.0
                or cur["enemy_tower_distance"] > prev["enemy_tower_distance"] + 50.0
                or cur["lane_progress"] < prev["lane_progress"] - 0.001
            )
        )
        dangerous_low_hp_attack = (
            cur["retreat_window"]
            and (cur["action_button"] == GameConfig.BUTTON_ATTACK or cur["action_button"] in GameConfig.SKILL_BUTTONS)
            and cur["action_target"] == GameConfig.TARGET_ENEMY
            and not cur["enemy_low_hp"]
            and not cur["self_under_own_tower"]
        )
        if cur["retreat_window"] and moving_back:
            reward_retreat = 0.03
        if dangerous_low_hp_attack:
            reward_bad_attack = -0.02
        if cur["safe_to_recall"] and cur["action_button"] == GameConfig.BUTTON_RECALL:
            reward_safe_recall = 0.03
        return {
            "reward_retreat_to_own_tower": reward_retreat,
            "reward_bad_attack_low_hp": reward_bad_attack,
            "reward_safe_recall": reward_safe_recall,
        }

    def _combo_rewards(self, cur, skill_recent):
        skill1_used = cur["action_button"] == GameConfig.BUTTON_SKILL_1
        skill2_used = cur["action_button"] == GameConfig.BUTTON_SKILL_2
        skill3_used = cur["action_button"] == GameConfig.BUTTON_SKILL_3
        attack_target = cur["action_target"] in (GameConfig.TARGET_ENEMY, GameConfig.TARGET_TOWER)
        skill1_good_target = cur["action_target"] == GameConfig.TARGET_ENEMY or cur["action_target"] in GameConfig.TARGET_SOLDIERS
        luban_s2_good = (
            cur["self_hero_id"] == 112
            and skill2_used
            and cur["action_target"] == GameConfig.TARGET_ENEMY
            and not (cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"])
            and (
                (cur["enemy_low_hp"] and cur["self_hp_ratio"] > 0.45)
                or (cur["enemy_close"] and cur["self_hp_ratio"] > 0.25)
            )
        )
        luban_s3_good = (
            cur["self_hero_id"] == 112
            and skill3_used
            and cur["self_hp_ratio"] > 0.35
            and not (cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"])
            and (
                cur["push_window"]
                or cur["enemy_soldier_visible"]
                or cur["has_ally_minion_under_enemy_tower"]
                or (cur["action_target"] == GameConfig.TARGET_ENEMY and cur["enemy_low_hp"])
            )
            and cur["action_target"] in (*GameConfig.TARGET_SOLDIERS, GameConfig.TARGET_TOWER, GameConfig.TARGET_ENEMY)
        )
        direnjie_s2_good = (
            cur["self_hero_id"] == 133
            and skill2_used
            and cur["action_target"] in (GameConfig.TARGET_SELF, GameConfig.TARGET_NONE)
            and (cur["self_hp_ratio"] < 0.50 or cur["retreat_window"] or cur["enemy_close"])
        )
        direnjie_s3_good = (
            cur["self_hero_id"] == 133
            and skill3_used
            and cur["action_target"] == GameConfig.TARGET_ENEMY
            and cur["self_hp_ratio"] > 0.45
            and 4500.0 <= cur["enemy_distance"] <= 12500.0
            and not (cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"])
        )
        return {
            "reward_skill1_good_cast": (
                0.018
                if GameConfig.ENABLE_COMBO_REWARD
                and skill1_used
                and not cur["self_in_enemy_tower_range"]
                and (cur["can_fight_hero"] or cur["enemy_soldier_visible"])
                and skill1_good_target
                else 0.0
            ),
            "reward_skill2_good_cast": (
                0.04
                if GameConfig.ENABLE_COMBO_REWARD
                and (luban_s2_good or direnjie_s2_good)
                else 0.0
            ),
            "reward_skill3_good_cast": (
                0.07
                if GameConfig.ENABLE_COMBO_REWARD
                and (luban_s3_good or direnjie_s3_good)
                else 0.0
            ),
            "reward_skill_after_attack": 0.018 if GameConfig.ENABLE_COMBO_REWARD and skill_recent and cur["action_button"] == GameConfig.BUTTON_ATTACK and attack_target else 0.0,
            "reward_berserk_attack_follow": (
                0.01
                if GameConfig.ENABLE_BERSERK_REWARD
                and cur["berserk_active"]
                and cur["action_button"] == GameConfig.BUTTON_ATTACK
                and cur["action_target"] == GameConfig.TARGET_ENEMY
                else 0.0
            ),
        }

    def _bad_skill2_reward(self, prev, cur, enemy_hp_drop):
        skill2_used = cur["action_button"] == GameConfig.BUTTON_SKILL_2
        skill2_hit = cur["skill2_hit_total"] > prev["skill2_hit_total"] or (skill2_used and enemy_hp_drop > 0.001)
        if not skill2_used:
            return 0.0
        if cur["self_hero_id"] == 112:
            # 鲁班 skill2：残血收割/远程压制/贴脸自保
            if cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"]:
                return -0.018  # 敌塔危险乱放
            if cur["action_target"] != GameConfig.TARGET_ENEMY:
                return -0.015  # 目标不是敌人（轻微惩罚）
            # 合理场景：残血收割、贴脸自保、狂暴后追击
            if cur["enemy_low_hp"] or cur["enemy_close"] or cur["berserk_active"]:
                return 0.0  # 合理使用不惩罚
            if cur["enemy_hp_ratio"] > 0.65 and not skill2_hit:
                return -0.010  # 高血量敌人且没命中，轻微惩罚
            return 0.0
        if cur["self_hero_id"] == 133:
            # 狄仁杰 skill2：防守技能，target self/none 是正常用法
            defensive_context = (
                cur["self_hp_ratio"] < 0.55  # 放宽血量阈值
                or cur["retreat_window"]
                or cur["enemy_close"]
                or cur.get("recent_hurt_by_hero", 0.0) > 0.0  # 刚受到伤害
            )
            if cur["action_target"] in (GameConfig.TARGET_SELF, GameConfig.TARGET_NONE):
                # 防守用法：低血量/被打/撤退窗口时不惩罚
                if defensive_context:
                    return 0.0  # 合理防守不惩罚
                return -0.008  # 完全安全时乱放，轻微惩罚
            if cur["action_target"] == GameConfig.TARGET_ENEMY:
                return -0.010  # 狄仁杰2技能打敌人是错误用法
            return -0.008
        return 0.0

    def _bad_skill3_reward(self, prev, cur, enemy_hp_drop):
        skill3_used = cur["action_button"] == GameConfig.BUTTON_SKILL_3
        skill3_missed = skill3_used and enemy_hp_drop <= 0.001
        if not skill3_used:
            return 0.0
        if cur["self_hero_id"] == 112:
            # 鲁班 skill3：清兵、压塔、推塔窗口、区域压制
            valid_target = cur["action_target"] in (*GameConfig.TARGET_SOLDIERS, GameConfig.TARGET_TOWER, GameConfig.TARGET_ENEMY)
            # 合理场景：推塔窗口、有敌方小兵、己方兵进敌塔、敌塔低血量、区域压制
            reasonable_context = (
                cur["push_window"]
                or cur["enemy_soldier_visible"]
                or cur["has_ally_minion_under_enemy_tower"]
                or cur.get("enemy_tower_low_hp", False)  # 敌塔低血量
                or (cur["action_target"] == GameConfig.TARGET_ENEMY and cur["enemy_low_hp"])
            )
            tower_danger = cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"]
            
            # 先检查危险情况，必须惩罚
            if cur["self_hp_ratio"] < 0.25:
                return -0.015  # 自己残血还在敌塔前乱放
            if tower_danger:
                return -0.020  # 敌塔危险区内强行放
            
            # 打小兵、打塔、压塔、封走位：安全情况下不惩罚
            if cur["action_target"] in GameConfig.TARGET_SOLDIERS or cur["action_target"] == GameConfig.TARGET_TOWER:
                if reasonable_context:
                    return 0.0  # 清兵/压塔合理使用
                return -0.010  # 没有兵线、没有敌人、没有塔目标时乱放
            
            # 打敌人
            if cur["action_target"] == GameConfig.TARGET_ENEMY:
                if cur["enemy_low_hp"]:
                    return 0.0  # 残血收割合理
                if reasonable_context:
                    return 0.0  # 有合理场景
                return -0.012  # 无合理场景乱放
            
            if not valid_target:
                return -0.015  # 无效目标
            if not reasonable_context:
                return -0.012  # 无合理场景
            return 0.0
        if cur["self_hero_id"] == 133:
            # 狄仁杰 skill3：控制技能，中距离打 enemy
            if cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"]:
                return -0.025  # 塔下危险乱放
            if cur["action_target"] != GameConfig.TARGET_ENEMY:
                return -0.018  # 空放
            # 中距离打 enemy 给正奖励（在 _combo_rewards 中处理）
            if skill3_missed and cur["enemy_distance"] > 9000.0:
                return -0.012  # 距离太远空放
            return 0.0
        return 0.0

    def _bad_summoner_reward(self, cur):
        if GameConfig.FIXED_SUMMONER_SKILL_ID == 80110:
            return 0.0
        if cur["action_button"] == GameConfig.BUTTON_SUMMONER and cur["enemy_hp_ratio"] > 0.35:
            return -0.005
        return 0.0

    def _has_berserk_buff(self, hero):
        return bool(collect_buff_config_ids(hero) & BERSERK_BUFF_IDS)

    def _update_skill_recent(self, prev, cur):
        skill_success = cur["skill_success_used"] or cur["skill_used_total"] > prev["skill_used_total"]
        if skill_success:
            self.skill_recent_frames = 5
        elif self.skill_recent_frames > 0:
            self.skill_recent_frames -= 1

    def _split_heroes(self, frame_data):
        heroes = frame_data.get("hero_states", [])
        main_hero = None
        for hero in heroes:
            if hero.get("player_id") == self.main_hero_player_id or hero.get("runtime_id") == self.main_hero_player_id:
                main_hero = hero
                break
        if main_hero is None and heroes:
            main_hero = heroes[0]
        enemy_hero = None
        if main_hero is not None:
            for hero in heroes:
                if hero is not main_hero and hero.get("camp") != main_hero.get("camp"):
                    enemy_hero = hero
                    break
        return main_hero, enemy_hero

    def _lane_progress(self, hero, main_tower, enemy_tower):
        if not hero or not main_tower or not enemy_tower:
            return 0.0
        start = self._position(main_tower)
        end = self._position(enemy_tower)
        point = self._position(hero)
        vx, vz = end[0] - start[0], end[1] - start[1]
        lane_len_sq = max(vx * vx + vz * vz, 1.0)
        progress = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vz) / lane_len_sq
        if self._hp_ratio(hero) < 0.25:
            progress *= 0.3
        return self._clip(progress, 0.0, 1.0)

    def _in_tower_range(self, hero, tower):
        if not hero or not tower:
            return False
        tower_range = self._value(tower, "attack_range") or 9000.0
        return math.dist(self._position(hero), self._position(tower)) <= tower_range

    def _has_ally_minion_under_enemy_tower(self, soldiers, enemy_tower):
        if not enemy_tower:
            return False
        tower_range = self._value(enemy_tower, "attack_range") or 9000.0
        tower_pos = self._position(enemy_tower)
        return any(math.dist(self._position(soldier), tower_pos) <= tower_range for soldier in soldiers)

    def _safe_to_push(self, main_hero, enemy_hero, soldiers, enemy_tower):
        return (
            self._has_ally_minion_under_enemy_tower(soldiers, enemy_tower)
            and self._hp_ratio(main_hero) >= 0.30
            and not self._enemy_close_dangerous(main_hero, enemy_hero)
        )

    def _ally_minion_ahead(self, main_hero, soldiers, main_tower, enemy_tower):
        if not main_hero or not main_tower or not enemy_tower:
            return False
        hero_progress = self._lane_progress(main_hero, main_tower, enemy_tower)
        for soldier in soldiers:
            if self._lane_progress(soldier, main_tower, enemy_tower) > hero_progress + 0.03:
                return True
        return False

    def _enemy_close_dangerous(self, main_hero, enemy_hero):
        if not main_hero or not enemy_hero or self._hp_ratio(enemy_hero) <= 0.0:
            return False
        return math.dist(self._position(main_hero), self._position(enemy_hero)) <= 12000.0 and self._hp_ratio(enemy_hero) > 0.25

    def _skill_totals(self, hero):
        totals = {
            "used_total": 0.0,
            "succ_used_total": 0.0,
            "hit_total": 0.0,
            "skill2_used_total": 0.0,
            "skill2_hit_total": 0.0,
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
            used = self._value_any(slot, ("succUsedInFrame", "succ_used_in_frame", "usedTimes", "used_times"))
            succ_used = self._value_any(slot, ("succUsedInFrame", "succ_used_in_frame"))
            hit = self._value_any(slot, ("hitHeroTimes", "hit_hero_times"))
            totals["used_total"] += used
            totals["succ_used_total"] += succ_used
            totals["hit_total"] += hit
            slot_type = self._int_field(slot, "slotType", self._int_field(slot, "slot_type", -1))
            config_id = self._int_field(
                slot,
                "skillID",
                self._int_field(slot, "skill_id", self._int_field(slot, "configId", self._int_field(slot, "config_id", 0))),
            )
            is_skill2 = slot_type == 2 or index == 1 or GameConfig.skill_no_from_id(config_id) == 2
            if is_skill2:
                totals["skill2_used_total"] += used
                totals["skill2_hit_total"] += hit
        return totals

    def _skill_cd_ratios(self, hero):
        ratios = {1: 1.0, 2: 1.0, 3: 1.0}
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
            ratios[skill_no] = self._clip(cooldown / max_cooldown, 0.0, 1.0) if max_cooldown > 0 else 0.0
        return ratios

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

    def _last_hit_delta_from_frame_action(self, frame_data, main_hero):
        if not main_hero:
            return 0.0
        main_runtime_id = self._int_field(main_hero, "runtime_id", 0)
        if main_runtime_id <= 0:
            return 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        count = 0.0
        for dead_action in frame_action.get("dead_action", []):
            death = dead_action.get("death", {}) if isinstance(dead_action, dict) else {}
            killer = dead_action.get("killer", {}) if isinstance(dead_action, dict) else {}
            if self._int_field(killer, "runtime_id", 0) != main_runtime_id:
                continue
            event_key = self._death_event_key(death, killer)
            if event_key is not None:
                if event_key in self.seen_last_hit_death_events:
                    continue
                self.seen_last_hit_death_events.add(event_key)
            sub_type = self._int_field(death, "sub_type", 0)
            config_id = self._int_field(death, "config_id", 0)
            if sub_type in GameConfig.SOLDIER_SUB_TYPES or config_id in GameConfig.LANE_SOLDIER_CONFIG_IDS:
                count += 1.0
        return count

    def _death_event_key(self, death, killer):
        runtime_id = self._int_field(death, "runtime_id", 0)
        if runtime_id <= 0:
            return None
        return (
            runtime_id,
            self._int_field(death, "camp", -1),
            self._int_field(death, "config_id", -1),
            self._int_field(death, "sub_type", -1),
            self._int_field(killer, "runtime_id", 0),
        )

    def _int_field(self, data, key, default=0):
        if not data or data.get(key) is None:
            return default
        try:
            return int(data.get(key))
        except (TypeError, ValueError):
            return default

    def _action_button(self, action):
        if action is None or len(action) <= 0:
            return -1
        return int(action[0])

    def _action_target(self, action):
        if action is None or len(action) <= 5:
            return -1
        return int(action[5])

    def _get_tower(self, frame_data, camp):
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") == camp and int(self._value(organ, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return organ
        return None

    def _get_enemy_tower(self, frame_data, camp):
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") != camp and int(self._value(organ, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return organ
        return None

    def _soldiers(self, frame_data, camp):
        return [
            npc
            for npc in frame_data.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") == camp
            and not self._is_organ(npc)
            and self._is_lane_soldier(npc)
            and self._value(npc, "hp") > 0
        ]

    def _enemy_soldiers(self, frame_data, camp):
        return [
            npc
            for npc in frame_data.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") != camp
            and not self._is_organ(npc)
            and self._is_lane_soldier(npc)
            and self._value(npc, "hp") > 0
        ]

    def _neutral_monsters(self, frame_data):
        return [
            npc
            for npc in frame_data.get("npc_states", [])
            if isinstance(npc, dict)
            and self._value(npc, "hp") > 0
            and (
                self._int_field(npc, "actor_type", -1) == GameConfig.ACTOR_TYPE_MONSTER
                or self._int_field(npc, "config_id", 0) == RIVER_SPIRIT_CONFIG_ID
                or self._int_field(npc, "sub_type", -1) == 0 and npc.get("camp") not in (GameConfig.CAMP_BLUE, GameConfig.CAMP_RED)
            )
        ]

    def _nearest_unit(self, src, units):
        if not src or not units:
            return None
        src_pos = self._position(src)
        return min(units, key=lambda unit: math.dist(src_pos, self._position(unit)))

    def _distance_to_point(self, src, point):
        if not src or point is None:
            return 999999.0
        return math.dist(self._position(src), point)

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
        return self._clip(self._value(unit, "hp") / max_hp, 0.0, 1.0)

    def _position(self, unit):
        if not unit:
            return 0.0, 0.0
        location = unit.get("location", {})
        return float(location.get("x", 0.0) or 0.0), float(location.get("z", 0.0) or 0.0)

    def _distance_to_unit(self, src, dst):
        if not src or not dst:
            return 0.0
        return math.dist(self._position(src), self._position(dst))

    def _unit_id(self, unit):
        if not unit:
            return None
        for key in ("runtime_id", "actor_runtime_id", "obj_id"):
            value = unit.get(key)
            if value is not None:
                return str(value)
        location = unit.get("location", {})
        return (
            f"{unit.get('camp')}_{unit.get('sub_type')}_{unit.get('config_id')}_"
            f"{int(location.get('x', 0) or 0)}_{int(location.get('z', 0) or 0)}"
        )

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

    def _value(self, unit, key):
        if not unit:
            return 0.0
        return float(unit.get(key, 0.0) or 0.0)

    def _clip(self, value, min_value, max_value):
        return max(min_value, min(max_value, float(value)))
