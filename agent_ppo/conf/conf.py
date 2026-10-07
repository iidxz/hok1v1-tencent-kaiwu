#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    HERO_IDS = (112, 133)
    TRAIN_HERO_IDS = (112, 133)
    TRAIN_LINEUP_PAIRS = (
        (112, 112),
        (112, 133),
        (133, 112),
        (133, 112),
        (133, 133),
    )
    SKILL_ID_PREFIX_TO_NO = {
        "1121": 1,
        "1122": 2,
        "1123": 3,
        "1331": 1,
        "1332": 2,
        "1333": 3,
    }
    CAMP_BLUE = 1
    CAMP_RED = 2
    ACTOR_TYPE_HERO = 0
    ACTOR_TYPE_MONSTER = 1
    ACTOR_TYPE_ORGAN = 2

    BUTTON_NONE = 0
    BUTTON_NOOP = 1
    BUTTON_MOVE = 2
    BUTTON_ATTACK = 3
    BUTTON_SKILL_1 = 4
    BUTTON_SKILL_2 = 5
    BUTTON_SKILL_3 = 6
    BUTTON_HEAL = 7
    BUTTON_SUMMONER = 8
    BUTTON_RECALL = 9
    BUTTON_SKILL_4 = 10
    BUTTON_EQUIPMENT_SKILL = 11
    SKILL_BUTTONS = (BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3, BUTTON_SKILL_4)
    FIXED_SUMMONER_SKILL_ID = 80110
    TRAIN_OPPONENT_MODEL_ID = 280471
    EVAL_OPPONENT_MODEL_ID = 280471
    TRAIN_OPPONENT_MODEL_POOL = (280471, 279609, 278719)
    EVAL_OPPONENT_MODEL_POOL = (280471,)

    TARGET_NONE = 0
    TARGET_ENEMY = 1
    TARGET_SELF = 2
    TARGET_SOLDIERS = (3, 4, 5, 6)
    TARGET_TOWER = 7
    TARGET_MONSTER = 8

    TOWER_SUB_TYPES = (21,)
    CRYSTAL_SUB_TYPES = (24, 25)
    SOLDIER_SUB_TYPES = (11,)
    LANE_SOLDIER_CONFIG_IDS = (6800, 6801, 6802, 6803, 6804, 6805)

    HERO_SIGHT_FALLBACK = 7000.0
    DEFENSE_TOWER_SIGHT_FALLBACK = 8800.0
    SPRING_TOWER_SIGHT_FALLBACK = 8000.0
    MAX_FRAME_NO = 20000.0
    MAX_GOLD_DIFF = 6000.0
    MAX_EXP_DIFF = 6000.0

    UNIT_FEATURE_DIM = 24
    GLOBAL_FEATURE_DIM = 38
    SOLDIER_SLOT_COUNT = 4
    ORGAN_SLOT_COUNT = 2
    TARGET_SLOT_COUNT = 9

    REWARD_STAGE = "D"
    ENABLE_TOWER_DANGER_REWARD = True
    ENABLE_PUSH_ASSIST_REWARD = True
    ENABLE_LANE_FOLLOW_REWARD = True
    ENABLE_FINISH_REWARD = True
    ENABLE_SKILL_REWARD = True
    ENABLE_COMBO_REWARD = True
    ENABLE_SKILL_FOLLOW_REWARD = True
    ENABLE_SKILL_BAD_REWARD = True
    ENABLE_BERSERK_REWARD = True
    ENABLE_PUSH_RETREAT_REWARD = True
    ENABLE_BUFF_REWARD = False
    ENABLE_FULL_BUFF_FEATURE = False
    ENABLE_SKILL_EXPLORATION = True
    SKILL_OFFSET_CENTER = 8
    SKILL_OFFSET_CENTER_FALLBACK = 7
    SKILL_OFFSET_PRIORITIES = (8, 7, 9, 6, 10)
    LUBAN_SKILL1_PRIOR_PROB = 0.025
    LUBAN_SKILL2_FINISH_PRIOR_PROB = 0.045
    LUBAN_SKILL2_DEFENSE_PRIOR_PROB = 0.040
    LUBAN_SKILL3_PUSH_ZONE_PRIOR_PROB = 0.030
    DIRENJIE_SKILL1_POKE_PRIOR_PROB = 0.025
    DIRENJIE_SKILL2_DEFENSE_PRIOR_PROB = 0.035
    DIRENJIE_SKILL3_CONTROL_PRIOR_PROB = 0.030
    SKILL_HIT_FOLLOW_ATTACK_PROB = 0.35
    LUBAN_SWEEP_FOLLOW_ATTACK_PROB = 0.30
    USE_TARGET_ATTENTION = True
    ATTENTION_ALPHA = 0.15
    ATTENTION_LOGIT_SCALE = 1.0
    USE_RECURRENT = True

    REWARD_WEIGHT_DICT = {
        "reward_hp": 2.0,
        "reward_tower": 14.0,
        "reward_gold": 0.008,
        "reward_exp": 0.008,
        "reward_last_hit": 0.5,
        "reward_death": -1.0,
        "reward_kill": 1.0,
        "reward_forward": 0.03,
        "reward_win": 3.0,
        "reward_tower_danger": 1.0,
        "reward_push_assist": 1.0,
        "reward_miss_push_window": 1.0,
        "reward_lane_follow": 1.0,
        "reward_finish": 1.0,
        "reward_hero_damage": 1.2,
        "reward_combat_kite": 1.2,
        "reward_combat_standstill": 1.5,
        "reward_attack_move": 1.2,
        "reward_attack_standstill": 1.6,
        "reward_low_hp_cake": 1.0,
        "reward_low_hp_heal": 1.0,
        "reward_low_hp_idle": 1.0,
        "reward_follow_minion_after_clear": 1.2,
        "reward_idle_after_clear": 1.6,
        "reward_river_monster": 1.4,
        "reward_idle_when_monster": 1.6,
        "reward_attack_hero_in_range": 1.0,
        "reward_miss_hero_in_range": 1.0,
        "reward_skill_hit_hero": 1.4,
        "reward_skill_damage": 1.4,
        "reward_skill2_ready_unused": 2.0,
        "reward_skill3_ready_unused": 2.4,
        "reward_luban_buff_damage": 0.0,
        "reward_skill2_finish": 0.0,
        "reward_summoner_finish": 0.0,
        "reward_attack_hero_under_tower": 1.4,
        "reward_retreat_from_tower": 1.0,
        "reward_stay_under_tower": 1.0,
        "reward_base_stuck": 0.01,
        "reward_early_forward": 0.01,
        "reward_skill_follow_attack": 1.0,
        "reward_skill_follow_tower": 1.0,
        "reward_bad_skill2": 1.0,
        "reward_bad_skill3": 1.0,
        "reward_bad_summoner": 0.0,
        "reward_push_window_target_tower": 1.2,
        "reward_push_window_wrong_hero": 1.0,
        "reward_retreat_to_own_tower": 1.0,
        "reward_bad_attack_low_hp": 1.0,
        "reward_safe_recall": 1.0,
        "reward_skill1_good_cast": 1.2,
        "reward_skill2_good_cast": 1.8,
        "reward_skill3_good_cast": 2.4,
        "reward_skill_after_attack": 1.0,
        "reward_berserk_attack_follow": 1.0,
        "reward_tower_poke_hero": 1.8,
        "reward_tower_poke_dive": 1.6,
        "reward_defend_tower_clear": 2.0,
        "reward_defend_tower_ignore": 2.0,
    }
    TIME_SCALE_ARG = 0
    MODEL_SAVE_INTERVAL = 1800

    @classmethod
    def skill_no_from_id(cls, skill_id):
        skill_text = str(skill_id)
        for prefix, skill_no in cls.SKILL_ID_PREFIX_TO_NO.items():
            if skill_text.startswith(prefix):
                return skill_no
        return None


class DimConfig:
    UNIT_FEATURE_DIM = GameConfig.UNIT_FEATURE_DIM
    GLOBAL_FEATURE_DIM = GameConfig.GLOBAL_FEATURE_DIM
    UNIT_SLOT_COUNT = (
        1
        + 1
        + GameConfig.SOLDIER_SLOT_COUNT
        + GameConfig.SOLDIER_SLOT_COUNT
        + GameConfig.ORGAN_SLOT_COUNT
        + GameConfig.ORGAN_SLOT_COUNT
    )
    DIM_OF_FEATURE = [UNIT_SLOT_COUNT * UNIT_FEATURE_DIM + GLOBAL_FEATURE_DIM]


class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    LEGAL_ACTION_DIM = 85
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 2e-4
    TARGET_LR = 5e-5
    TARGET_STEP = 4000
    BETA_START = 0.045
    LOG_EPSILON = 1e-6
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001

    UNIT_EMBED_DIM = 64
    GROUP_EMBED_DIM = 96
    TARGET_EMBED_DIM = 64

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * LSTM_TIME_STEPS],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
