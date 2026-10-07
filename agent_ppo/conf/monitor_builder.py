#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder

from agent_ppo.conf.conf import GameConfig


def _panel(builder, name, metrics, panel_type="line", name_en=None):
    builder = builder.add_panel(name=name, name_en=name_en or name, type=panel_type)
    for metrics_name, expr in metrics:
        builder = builder.add_metric(metrics_name=metrics_name, expr=expr)
    return builder.end_panel()


def build_monitor():
    train_id = GameConfig.TRAIN_OPPONENT_MODEL_ID
    eval_id = GameConfig.EVAL_OPPONENT_MODEL_ID
    train_pool = tuple(dict.fromkeys(GameConfig.TRAIN_OPPONENT_MODEL_POOL or (train_id,)))
    eval_pool = tuple(dict.fromkeys(GameConfig.EVAL_OPPONENT_MODEL_POOL or (eval_id,)))
    opponent_pool = tuple(dict.fromkeys((*train_pool, *eval_pool)))
    matchups = tuple((self_id, enemy_id) for self_id in GameConfig.HERO_IDS for enemy_id in GameConfig.HERO_IDS)
    builder = MonitorConfigBuilder().title("hok1v1_mixed_history_opponents")

    builder = builder.add_group(group_name="result", group_name_en="result")
    builder = _panel(
        builder,
        "overall_win_rate",
        [
            ("train_win_rate", "round(avg(train_win_rate{}), 0.01)"),
            ("eval_win_rate", "round(avg(eval_win_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "matchup_win",
        [
            (f"win_rate_{self_id}_vs_{enemy_id}", f"round(avg(win_rate_{self_id}_vs_{enemy_id}{{}}), 0.01)")
            for self_id, enemy_id in matchups
        ],
    )
    builder = _panel(
        builder,
        "opponent_win",
        [(f"win_rate_vs_{model_id}", f"round(avg(win_rate_{model_id}{{}}), 0.01)") for model_id in opponent_pool],
    )
    builder = _panel(
        builder,
        "matchup_tower",
        [
            (f"enemy_tower_drop_{self_id}_vs_{enemy_id}", f"round(avg(enemy_tower_hp_drop_{self_id}_vs_{enemy_id}{{}}), 0.01)")
            for self_id, enemy_id in matchups
        ],
    )
    builder = _panel(builder, "episode_reward", [("reward_total", "round(avg(reward{}), 0.01)")])
    builder = builder.end_group()

    builder = builder.add_group(group_name="attention", group_name_en="attention")
    builder = _panel(
        builder,
        "attention_config",
        [
            ("attention_enabled", "round(avg(attention_enabled{}), 0.01)"),
            ("attention_alpha", "round(avg(attention_alpha{}), 0.01)"),
            ("attention_logit_scale", "round(avg(attention_logit_scale{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "attention_stats",
        [
            ("attention_entropy", "round(avg(attention_entropy{}), 0.01)"),
            ("attention_max_prob", "round(avg(attention_max_prob{}), 0.01)"),
            ("target_changed_by_attention_rate", "round(avg(target_changed_by_attention_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "attn_target_prob",
        [
            ("attention_target_hero_prob", "round(avg(attention_target_hero_prob{}), 0.01)"),
            ("attention_target_soldier_prob", "round(avg(attention_target_soldier_prob{}), 0.01)"),
            ("attention_target_tower_prob", "round(avg(attention_target_tower_prob{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "target_distribution",
        [
            ("hero_target_rate", "round(avg(hero_target_rate{}), 0.01)"),
            ("creep_target_rate", "round(avg(creep_target_rate{}), 0.01)"),
            ("tower_target_rate", "round(avg(tower_target_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="combat", group_name_en="combat")
    builder = _panel(
        builder,
        "combat_result",
        [
            ("kill", "round(avg(kill{}), 0.01)"),
            ("death", "round(avg(death{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "hero_damage",
        [
            ("hurt_to_hero_delta", "round(avg(hurt_to_hero_delta{}), 0.01)"),
            ("hurt_by_hero_delta", "round(avg(hurt_by_hero_delta{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "hero_state",
        [
            ("self_hp_ratio", "round(avg(self_hp_ratio{}), 0.01)"),
            ("enemy_hp_ratio", "round(avg(enemy_hp_ratio{}), 0.01)"),
            ("hp_advantage", "round(avg(hp_advantage{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="skill", group_name_en="skill")
    builder = _panel(
        builder,
        "skill_use",
        [
            ("skill1_use_rate", "round(avg(skill1_use_rate{}), 0.01)"),
            ("skill2_use_rate", "round(avg(skill2_use_rate{}), 0.01)"),
            ("skill3_use_rate", "round(avg(skill3_use_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_hit_by_id",
        [
            ("skill1_hit_hero_rate", "round(avg(skill1_hit_hero_rate{}), 0.01)"),
            ("skill2_hit_hero_rate", "round(avg(skill2_hit_hero_rate{}), 0.01)"),
            ("skill3_hit_hero_rate", "round(avg(skill3_hit_hero_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_cooldown",
        [
            ("skill1_cd_ratio", "round(avg(skill1_cd_ratio{}), 0.01)"),
            ("skill2_cd_ratio", "round(avg(skill2_cd_ratio{}), 0.01)"),
            ("skill3_cd_ratio", "round(avg(skill3_cd_ratio{}), 0.01)"),
            ("summoner_cd_ratio", "round(avg(summoner_cd_ratio{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_behavior",
        [
            ("skill1_legal_rate", "round(avg(skill1_legal_rate{}), 0.01)"),
            ("skill2_legal_rate", "round(avg(skill2_legal_rate{}), 0.01)"),
            ("skill3_legal_rate", "round(avg(skill3_legal_rate{}), 0.01)"),
            ("skill_after_attack_rate", "round(avg(skill_after_attack_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_cmd_use_112",
        [
            ("s1_cmd_112", "round(avg(s1_cmd_112{}), 0.01)"),
            ("s1_use_112", "round(avg(s1_use_112{}), 0.01)"),
            ("s1_hit_112", "round(avg(s1_hit_112{}), 0.01)"),
            ("s2_cmd_112", "round(avg(s2_cmd_112{}), 0.01)"),
            ("s2_use_112", "round(avg(s2_use_112{}), 0.01)"),
            ("s2_hit_112", "round(avg(s2_hit_112{}), 0.01)"),
            ("s3_cmd_112", "round(avg(s3_cmd_112{}), 0.01)"),
            ("s3_use_112", "round(avg(s3_use_112{}), 0.01)"),
            ("s3_hit_112", "round(avg(s3_hit_112{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_cmd_use_133",
        [
            ("s1_cmd_133", "round(avg(s1_cmd_133{}), 0.01)"),
            ("s1_use_133", "round(avg(s1_use_133{}), 0.01)"),
            ("s1_hit_133", "round(avg(s1_hit_133{}), 0.01)"),
            ("s2_cmd_133", "round(avg(s2_cmd_133{}), 0.01)"),
            ("s2_use_133", "round(avg(s2_use_133{}), 0.01)"),
            ("s2_hit_133", "round(avg(s2_hit_133{}), 0.01)"),
            ("s3_cmd_133", "round(avg(s3_cmd_133{}), 0.01)"),
            ("s3_use_133", "round(avg(s3_use_133{}), 0.01)"),
            ("s3_hit_133", "round(avg(s3_hit_133{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_cmd_diag",
        [
            ("s2_cmd_legal", "round(avg(s2_cmd_legal{}), 0.01)"),
            ("s3_cmd_legal", "round(avg(s3_cmd_legal{}), 0.01)"),
            ("s2_cmd_blocked_by_cd", "round(avg(s2_cmd_blocked_by_cd{}), 0.01)"),
            ("s3_cmd_blocked_by_cd", "round(avg(s3_cmd_blocked_by_cd{}), 0.01)"),
            ("s2_cmd_blocked_by_mask", "round(avg(s2_cmd_blocked_by_mask{}), 0.01)"),
            ("s3_cmd_blocked_by_mask", "round(avg(s3_cmd_blocked_by_mask{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_prior",
        [
            ("luban_s2_prior_try_rate", "round(avg(luban_s2_prior_try_rate{}), 0.01)"),
            ("luban_s2_prior_apply_rate", "round(avg(luban_s2_prior_apply_rate{}), 0.01)"),
            ("luban_s3_prior_try_rate", "round(avg(luban_s3_prior_try_rate{}), 0.01)"),
            ("luban_s3_prior_apply_rate", "round(avg(luban_s3_prior_apply_rate{}), 0.01)"),
            ("direnjie_s2_defense_prior_try_rate", "round(avg(direnjie_s2_defense_prior_try_rate{}), 0.01)"),
            ("direnjie_s2_defense_prior_apply_rate", "round(avg(direnjie_s2_defense_prior_apply_rate{}), 0.01)"),
            ("direnjie_s3_prior_try_rate", "round(avg(direnjie_s3_prior_try_rate{}), 0.01)"),
            ("direnjie_s3_prior_apply_rate", "round(avg(direnjie_s3_prior_apply_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_prior_context",
        [
            ("luban_s1_follow_attack_rate", "round(avg(luban_s1_follow_attack_rate{}), 0.01)"),
            ("luban_s2_finish_try_rate", "round(avg(luban_s2_finish_try_rate{}), 0.01)"),
            ("luban_s2_defense_try_rate", "round(avg(luban_s2_defense_try_rate{}), 0.01)"),
            ("luban_s3_push_zone_try_rate", "round(avg(luban_s3_push_zone_try_rate{}), 0.01)"),
            ("direnjie_s1_poke_try_rate", "round(avg(direnjie_s1_poke_try_rate{}), 0.01)"),
            ("direnjie_s2_defense_try_rate", "round(avg(direnjie_s2_defense_try_rate{}), 0.01)"),
            ("direnjie_s3_control_try_rate", "round(avg(direnjie_s3_control_try_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "skill_prior_safety",
        [
            ("prior_blocked_by_cd_rate", "round(avg(prior_blocked_by_cd_rate{}), 0.01)"),
            ("prior_blocked_by_mask_rate", "round(avg(prior_blocked_by_mask_rate{}), 0.01)"),
            ("prior_blocked_by_tower_risk_rate", "round(avg(prior_blocked_by_tower_risk_rate{}), 0.01)"),
            ("prior_blocked_by_low_hp_retreat_rate", "round(avg(prior_blocked_by_low_hp_retreat_rate{}), 0.01)"),
            ("skill_after_hit_follow_attack_rate", "round(avg(skill_after_hit_follow_attack_rate{}), 0.01)"),
            ("skill_interrupt_sweep_rate", "round(avg(skill_interrupt_sweep_rate{}), 0.01)"),
            ("bad_skill_under_enemy_tower_rate", "round(avg(bad_skill_under_enemy_tower_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "鲁班先验阻断",
        [
            ("luban_s2_prior_block_cd_rate", "round(avg(luban_s2_prior_block_cd_rate{}), 0.01)"),
            ("luban_s2_prior_block_mask_rate", "round(avg(luban_s2_prior_block_mask_rate{}), 0.01)"),
            ("luban_s2_prior_block_tower_risk_rate", "round(avg(luban_s2_prior_block_tower_risk_rate{}), 0.01)"),
            ("luban_s2_prior_block_retreat_rate", "round(avg(luban_s2_prior_block_retreat_rate{}), 0.01)"),
            ("luban_s3_prior_block_cd_rate", "round(avg(luban_s3_prior_block_cd_rate{}), 0.01)"),
            ("luban_s3_prior_block_mask_rate", "round(avg(luban_s3_prior_block_mask_rate{}), 0.01)"),
            ("luban_s3_prior_block_tower_risk_rate", "round(avg(luban_s3_prior_block_tower_risk_rate{}), 0.01)"),
            ("luban_s3_prior_block_retreat_rate", "round(avg(luban_s3_prior_block_retreat_rate{}), 0.01)"),
        ],
        name_en="luban_prior_block",
    )
    builder = _panel(
        builder,
        "狄仁杰先验阻断",
        [
            ("direnjie_s2_prior_block_cd_rate", "round(avg(direnjie_s2_prior_block_cd_rate{}), 0.01)"),
            ("direnjie_s2_prior_block_mask_rate", "round(avg(direnjie_s2_prior_block_mask_rate{}), 0.01)"),
            ("direnjie_s2_prior_block_tower_risk_rate", "round(avg(direnjie_s2_prior_block_tower_risk_rate{}), 0.01)"),
            ("direnjie_s2_prior_block_retreat_rate", "round(avg(direnjie_s2_prior_block_retreat_rate{}), 0.01)"),
            ("direnjie_s3_prior_block_cd_rate", "round(avg(direnjie_s3_prior_block_cd_rate{}), 0.01)"),
            ("direnjie_s3_prior_block_mask_rate", "round(avg(direnjie_s3_prior_block_mask_rate{}), 0.01)"),
            ("direnjie_s3_prior_block_tower_risk_rate", "round(avg(direnjie_s3_prior_block_tower_risk_rate{}), 0.01)"),
            ("direnjie_s3_prior_block_retreat_rate", "round(avg(direnjie_s3_prior_block_retreat_rate{}), 0.01)"),
        ],
        name_en="direnjie_prior_block",
    )
    # 诊断指标：每个英雄每个技能的 ready/legal/context 率
    builder = _panel(
        builder,
        "鲁班技能诊断",
        [
            ("luban_s2_ready_rate", "round(avg(luban_s2_ready_rate{}), 0.01)"),
            ("luban_s2_legal_rate", "round(avg(luban_s2_legal_rate{}), 0.01)"),
            ("luban_s2_context_rate", "round(avg(luban_s2_context_rate{}), 0.01)"),
            ("luban_s3_ready_rate", "round(avg(luban_s3_ready_rate{}), 0.01)"),
            ("luban_s3_legal_rate", "round(avg(luban_s3_legal_rate{}), 0.01)"),
            ("luban_s3_context_rate", "round(avg(luban_s3_context_rate{}), 0.01)"),
        ],
        name_en="luban_skill_diag",
    )
    builder = _panel(
        builder,
        "狄仁杰技能诊断",
        [
            ("direnjie_s2_ready_rate", "round(avg(direnjie_s2_ready_rate{}), 0.01)"),
            ("direnjie_s2_legal_rate", "round(avg(direnjie_s2_legal_rate{}), 0.01)"),
            ("direnjie_s2_context_rate", "round(avg(direnjie_s2_context_rate{}), 0.01)"),
            ("direnjie_s3_ready_rate", "round(avg(direnjie_s3_ready_rate{}), 0.01)"),
            ("direnjie_s3_legal_rate", "round(avg(direnjie_s3_legal_rate{}), 0.01)"),
            ("direnjie_s3_context_rate", "round(avg(direnjie_s3_context_rate{}), 0.01)"),
        ],
        name_en="direnjie_skill_diag",
    )
    builder = _panel(
        builder,
        "skill_follow",
        [
            ("reward_skill1_good_cast_trigger_rate", "round(avg(reward_skill1_good_cast_trigger_rate{}), 0.01)"),
            ("reward_skill3_good_cast_trigger_rate", "round(avg(reward_skill3_good_cast_trigger_rate{}), 0.01)"),
            ("reward_skill_after_attack_trigger_rate", "round(avg(reward_skill_after_attack_trigger_rate{}), 0.01)"),
            ("berserk_attack_follow_trigger_rate", "round(avg(reward_berserk_attack_follow_trigger_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="berserk", group_name_en="berserk")
    builder = _panel(
        builder,
        "berserk_state",
        [
            ("berserk_active_rate", "round(avg(berserk_active_rate{}), 0.01)"),
            ("berserk_attack_hero_rate", "round(avg(berserk_attack_hero_rate{}), 0.01)"),
            ("berserk_death_after_use_rate", "round(avg(berserk_death_after_use_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "berserk_combat",
        [
            ("berserk_hurt_to_hero", "round(avg(berserk_hurt_to_hero{}), 0.01)"),
            ("berserk_hurt_by_hero", "round(avg(berserk_hurt_by_hero{}), 0.01)"),
            ("berserk_hp_advantage_delta", "round(avg(berserk_hp_advantage_delta{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="combo_start", group_name_en="combo_start")
    builder = _panel(
        builder,
        "start_fix",
        [
            ("base_stuck_rate", "round(avg(base_stuck_rate{}), 0.01)"),
            ("early_forward_rate", "round(avg(early_forward_rate{}), 0.01)"),
            ("early_low_movement_rate", "round(avg(early_low_movement_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "combo_trigger",
        [
            ("reward_base_stuck_trigger_rate", "round(avg(reward_base_stuck_trigger_rate{}), 0.01)"),
            ("reward_early_forward_trigger_rate", "round(avg(reward_early_forward_trigger_rate{}), 0.01)"),
            ("reward_skill_follow_attack_trigger_rate", "round(avg(reward_skill_follow_attack_trigger_rate{}), 0.01)"),
            ("reward_skill_follow_tower_trigger_rate", "round(avg(reward_skill_follow_tower_trigger_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="tower_keep", group_name_en="tower_keep")
    builder = _panel(
        builder,
        "tower_state",
        [
            ("self_tower_hp_ratio", "round(avg(self_tower_hp_ratio{}), 0.01)"),
            ("enemy_tower_hp_ratio", "round(avg(enemy_tower_hp_ratio{}), 0.01)"),
            ("enemy_tower_hp_drop", "round(avg(enemy_tower_hp_drop{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "tower_safety",
        [
            ("enemy_tower_range_rate", "round(avg(enemy_tower_range_rate{}), 0.01)"),
            ("no_minion_tower_dive_rate", "round(avg(no_minion_tower_dive_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "tower_attack",
        [
            ("safe_attack_tower_rate", "round(avg(safe_attack_tower_rate{}), 0.01)"),
            ("attack_tower_with_minion_rate", "round(avg(attack_tower_with_minion_rate{}), 0.01)"),
            ("attack_hero_under_enemy_tower_rate", "round(avg(attack_hero_under_enemy_tower_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "tower_poke",
        [
            ("tower_poke_window_rate", "round(avg(tower_poke_window_rate{}), 0.01)"),
            ("tower_poke_hero_rate", "round(avg(tower_poke_hero_rate{}), 0.01)"),
            ("reward_tower_poke_hero_trigger_rate", "round(avg(reward_tower_poke_hero_trigger_rate{}), 0.01)"),
            ("reward_tower_poke_dive_trigger_rate", "round(avg(reward_tower_poke_dive_trigger_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "defend_tower",
        [
            ("defend_tower_window_rate", "round(avg(defend_tower_window_rate{}), 0.01)"),
            ("defend_tower_clear_rate", "round(avg(defend_tower_clear_rate{}), 0.01)"),
            ("reward_defend_tower_clear_trigger_rate", "round(avg(reward_defend_tower_clear_trigger_rate{}), 0.01)"),
            ("reward_defend_tower_ignore_trigger_rate", "round(avg(reward_defend_tower_ignore_trigger_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="push_retreat", group_name_en="push_retreat")
    builder = _panel(
        builder,
        "push_window",
        [
            ("strict_push_window_rate", "round(avg(strict_push_window_rate{}), 0.01)"),
            ("near_push_window_rate", "round(avg(near_push_window_rate{}), 0.01)"),
            ("push_window_rate", "round(avg(push_window_rate{}), 0.01)"),
            ("push_window_target_tower_rate", "round(avg(push_window_target_tower_rate{}), 0.01)"),
            ("push_window_wrong_hero_rate", "round(avg(push_window_wrong_hero_rate{}), 0.01)"),
            ("attack_tower_with_minion_rate", "round(avg(attack_tower_with_minion_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "retreat_window",
        [
            ("retreat_window_rate", "round(avg(retreat_window_rate{}), 0.01)"),
            ("retreat_to_own_tower_rate", "round(avg(retreat_to_own_tower_rate{}), 0.01)"),
            ("retreat_success_rate", "round(avg(retreat_success_rate{}), 0.01)"),
            ("bad_attack_low_hp_rate", "round(avg(bad_attack_low_hp_rate{}), 0.01)"),
            ("safe_recall_rate", "round(avg(safe_recall_rate{}), 0.01)"),
            ("low_hp_death_rate", "round(avg(low_hp_death_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="gold_source", group_name_en="gold_source")
    builder = _panel(
        builder,
        "gold_known",
        [
            ("gold_from_soldier", "round(avg(gold_from_soldier{}), 0.01)"),
            ("gold_from_hero", "round(avg(gold_from_hero{}), 0.01)"),
            ("gold_from_other", "round(avg(gold_from_other{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "gold_total",
        [
            ("gold_passive_or_unknown", "round(avg(gold_passive_or_unknown{}), 0.01)"),
            ("gold_event_total", "round(avg(gold_event_total{}), 0.01)"),
            ("gold_total_delta", "round(avg(gold_total_delta{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "kill_source",
        [
            ("soldier_last_hit_count", "round(avg(soldier_last_hit_count{}), 0.01)"),
            ("hero_kill_event_count", "round(avg(hero_kill_event_count{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="reward_d", group_name_en="reward_d")
    builder = _panel(
        builder,
        "reward_combat",
        [
            ("reward_hero_damage", "round(avg(reward_hero_damage{}), 0.01)"),
            ("reward_combat_kite", "round(avg(reward_combat_kite{}), 0.01)"),
            ("reward_combat_standstill", "round(avg(reward_combat_standstill{}), 0.01)"),
            ("reward_attack_move", "round(avg(reward_attack_move{}), 0.01)"),
            ("reward_attack_standstill", "round(avg(reward_attack_standstill{}), 0.01)"),
            ("reward_skill_hit_hero", "round(avg(reward_skill_hit_hero{}), 0.01)"),
            ("reward_skill_damage", "round(avg(reward_skill_damage{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_recovery",
        [
            ("reward_low_hp_cake", "round(avg(reward_low_hp_cake{}), 0.01)"),
            ("reward_low_hp_heal", "round(avg(reward_low_hp_heal{}), 0.01)"),
            ("reward_low_hp_idle", "round(avg(reward_low_hp_idle{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_map_pressure",
        [
            ("reward_follow_minion_after_clear", "round(avg(reward_follow_minion_after_clear{}), 0.01)"),
            ("reward_idle_after_clear", "round(avg(reward_idle_after_clear{}), 0.01)"),
            ("reward_river_monster", "round(avg(reward_river_monster{}), 0.01)"),
            ("reward_idle_when_monster", "round(avg(reward_idle_when_monster{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_tower",
        [
            ("reward_push_assist", "round(avg(reward_push_assist{}), 0.01)"),
            ("reward_attack_hero_under_tower", "round(avg(reward_attack_hero_under_tower{}), 0.01)"),
            ("reward_retreat_from_tower", "round(avg(reward_retreat_from_tower{}), 0.01)"),
            ("reward_stay_under_tower", "round(avg(reward_stay_under_tower{}), 0.01)"),
            ("reward_tower_poke_hero", "round(avg(reward_tower_poke_hero{}), 0.01)"),
            ("reward_tower_poke_dive", "round(avg(reward_tower_poke_dive{}), 0.01)"),
            ("reward_defend_tower_clear", "round(avg(reward_defend_tower_clear{}), 0.01)"),
            ("reward_defend_tower_ignore", "round(avg(reward_defend_tower_ignore{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_combo_start",
        [
            ("reward_base_stuck", "round(avg(reward_base_stuck{}), 0.01)"),
            ("reward_early_forward", "round(avg(reward_early_forward{}), 0.01)"),
            ("reward_skill_follow_attack", "round(avg(reward_skill_follow_attack{}), 0.01)"),
            ("reward_skill_follow_tower", "round(avg(reward_skill_follow_tower{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_combo_skill",
        [
            ("reward_skill1_good_cast", "round(avg(reward_skill1_good_cast{}), 0.01)"),
            ("reward_skill2_good_cast", "round(avg(reward_skill2_good_cast{}), 0.01)"),
            ("reward_skill3_good_cast", "round(avg(reward_skill3_good_cast{}), 0.01)"),
            ("reward_skill2_ready_unused", "round(avg(reward_skill2_ready_unused{}), 0.01)"),
            ("reward_skill3_ready_unused", "round(avg(reward_skill3_ready_unused{}), 0.01)"),
            ("reward_skill_after_attack", "round(avg(reward_skill_after_attack{}), 0.01)"),
            ("reward_berserk_attack_follow", "round(avg(reward_berserk_attack_follow{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_windows",
        [
            ("reward_push_window_target_tower", "round(avg(reward_push_window_target_tower{}), 0.01)"),
            ("reward_push_window_wrong_hero", "round(avg(reward_push_window_wrong_hero{}), 0.01)"),
            ("reward_retreat_to_own_tower", "round(avg(reward_retreat_to_own_tower{}), 0.01)"),
            ("reward_bad_attack_low_hp", "round(avg(reward_bad_attack_low_hp{}), 0.01)"),
            ("reward_safe_recall", "round(avg(reward_safe_recall{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "reward_base",
        [
            ("reward_gold", "round(avg(reward_gold{}), 0.01)"),
            ("reward_exp", "round(avg(reward_exp{}), 0.01)"),
            ("reward_last_hit", "round(avg(reward_last_hit{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="behavior", group_name_en="behavior")
    builder = _panel(
        builder,
        "action_main",
        [
            ("move_action_rate", "round(avg(move_action_rate{}), 0.01)"),
            ("attack_action_rate", "round(avg(attack_action_rate{}), 0.01)"),
            ("low_movement_rate", "round(avg(low_movement_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="training", group_name_en="training")
    builder = _panel(
        builder,
        "loss",
        [
            ("total_loss", "round(avg(total_loss{}), 0.01)"),
            ("value_loss", "round(avg(value_loss{}), 0.01)"),
            ("policy_loss", "round(avg(policy_loss{}), 0.01)"),
            ("entropy_loss", "round(avg(entropy_loss{}), 0.01)"),
        ],
    )
    return builder.end_group().build()
