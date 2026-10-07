#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Buff constants collected from buff.md.

These ids are used only by reward and monitor code in Stage D. They are not
part of the model feature vector, so changing this file does not change the
network input dimension.
"""


COMMON_BUFF_IDS = {
    90015,
    10000,
    10010,
    10014,
    11001,
    11002,
    11010,
    90025,
    911290,
    914110,
    914210,
    914211,
}

LUBAN_BUFF_IDS = {
    112001,
    112015,
    112025,
    112035,
    112040,
    112043,
    112044,
    112045,
    112046,
    112047,
    112048,
    112100,
    112200,
    112201,
    112300,
    112301,
    112320,
    112890,
    112910,
    112990,
    112991,
    112000,
    112010,
    112020,
    112030,
    112041,
    112042,
    112190,
    112191,
    112192,
    112210,
}

LUBAN_BASE_OR_ALWAYS_BUFF_IDS = {
    112000,
    112001,
    112010,
    112020,
    112030,
    112100,
    112200,
    112300,
}

LUBAN_OUTPUT_BUFF_IDS = LUBAN_BUFF_IDS - LUBAN_BASE_OR_ALWAYS_BUFF_IDS

DIRENJIE_BUFF_IDS = {
    133000,
    133001,
    133010,
    133011,
    133020,
    133090,
    133200,
    133250,
    133260,
    133950,
    133951,
    133100,
    133300,
    133310,
    912260,
    912262,
    912263,
}

OTHER_OBSERVED_BUFF_IDS = {
    50000,
    914230,
    914232,
    131956,
    167600,
    167602,
    500009,
    801100,
    90019,
    90110,
    911260,
    911261,
    912330,
    919900,
}

BERSERK_BUFF_IDS = {
    801100,
}

RIVER_SPIRIT_CONFIG_ID = 6827

CAKE_LOCATIONS_BY_CAMP = {
    1: (-15220.0, -15120.0),
    2: (15340.0, 15100.0),
}


def collect_buff_config_ids(unit):
    buff_state = unit.get("buff_state", {}) if isinstance(unit, dict) else {}
    config_ids = set()
    for group_key in ("buff_skills", "buff_marks"):
        for item in buff_state.get(group_key, []) or []:
            if not isinstance(item, dict):
                continue
            for key in ("configId", "config_id", "buff_id", "id"):
                value = item.get(key)
                if value is None:
                    continue
                try:
                    config_ids.add(int(value))
                except (TypeError, ValueError):
                    pass
    return config_ids


def has_luban_any_buff(unit):
    return bool(collect_buff_config_ids(unit) & LUBAN_BUFF_IDS)


def has_luban_output_buff(unit):
    return bool(collect_buff_config_ids(unit) & LUBAN_OUTPUT_BUFF_IDS)


def has_luban_buff(unit):
    return has_luban_any_buff(unit)
