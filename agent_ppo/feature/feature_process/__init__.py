#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math

from agent_ppo.conf.conf import DimConfig, GameConfig
from agent_ppo.feature.buff_constants import BERSERK_BUFF_IDS, collect_buff_config_ids, has_luban_output_buff


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp

    def reset(self, camp):
        self.camp = camp

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        main_hero = self._get_hero(frame_state, self.camp)
        enemy_hero = self._get_enemy_hero(frame_state, self.camp)
        main_tower = self._get_tower(frame_state, self.camp)
        enemy_tower = self._get_enemy_tower(frame_state, self.camp)
        main_crystal = self._get_crystal(frame_state, self.camp)
        enemy_crystal = self._get_enemy_crystal(frame_state, self.camp)

        lane = self._lane_basis(main_tower or main_crystal, enemy_tower or enemy_crystal)
        hero_pos = self._position(main_hero)

        my_soldiers = self._soldiers(frame_state, self.camp)
        enemy_soldiers = self._enemy_soldiers(frame_state, self.camp)

        slots = [
            self._unit_feature(main_hero, lane, hero_pos, 1.0, self.camp),
            self._unit_feature(enemy_hero, lane, hero_pos, 1.0, self._enemy_camp()),
        ]
        slots.extend(self._slot_features(my_soldiers, GameConfig.SOLDIER_SLOT_COUNT, lane, hero_pos))
        slots.extend(self._slot_features(enemy_soldiers, GameConfig.SOLDIER_SLOT_COUNT, lane, hero_pos))
        slots.extend(self._slot_features([main_tower, main_crystal], GameConfig.ORGAN_SLOT_COUNT, lane, hero_pos))
        slots.extend(self._slot_features([enemy_tower, enemy_crystal], GameConfig.ORGAN_SLOT_COUNT, lane, hero_pos))

        feature = []
        for slot in slots:
            feature.extend(slot)
        feature.extend(
            self._global_feature(
                frame_state,
                main_hero,
                enemy_hero,
                main_tower,
                enemy_tower,
                my_soldiers,
                enemy_soldiers,
                lane,
            )
        )

        expected_dim = DimConfig.DIM_OF_FEATURE[0]
        if len(feature) != expected_dim:
            raise ValueError(f"feature dim mismatch: got {len(feature)}, expected {expected_dim}")
        return feature

    def _unit_feature(self, unit, lane, hero_pos, exists, default_camp=None):
        if not unit:
            return [0.0] * GameConfig.UNIT_FEATURE_DIM

        pos = self._position(unit)
        lane_progress, lane_offset = self._lane_coord(pos, lane)
        rel_forward, rel_lateral = self._relative_lane_coord(pos, hero_pos, lane)
        config_id = int(self._value(unit, "config_id"))
        sub_type = int(self._value(unit, "sub_type"))
        camp = int(self._value(unit, "camp") or (default_camp or 0))
        skill_usable, skill_cd, skill_cd_max = self._skill_summary(unit)

        return [
            float(exists),
            self._hp_ratio(unit),
            self._ep_ratio(unit),
            self._ratio(self._value(unit, "level"), 15.0),
            self._ratio(self._resource(unit, "exp"), 10000.0),
            self._ratio(self._resource(unit, "money"), 20000.0),
            self._clip(lane_progress, 0.0, 1.0),
            self._clip(lane_offset / 40000.0, -1.0, 1.0),
            self._clip(rel_forward / 40000.0, -1.0, 1.0),
            self._clip(rel_lateral / 40000.0, -1.0, 1.0),
            self._dist_ratio(pos, hero_pos),
            1.0 if camp == self.camp else 0.0,
            1.0 if self._value(unit, "hp") > 0 else 0.0,
            self._camp_visible(unit),
            self._ratio(self._value(unit, "attack_range"), 20000.0),
            self._ratio(self._value(unit, "sight_area"), 12000.0),
            self._ratio(self._value_any(unit, ("mov_spd", "move_speed")), 15000.0),
            self._ratio(self._value_any(unit, ("atk_spd", "attack_speed")), 20000.0),
            skill_usable,
            skill_cd,
            skill_cd_max,
            1.0 if config_id == 112 else 0.0,
            1.0 if config_id == 133 else 0.0,
            self._ratio(sub_type, 100.0),
        ]

    def _slot_features(self, units, slot_count, lane, hero_pos):
        features = []
        ordered_units = sorted([unit for unit in units if unit], key=lambda unit: self._sort_key(unit, hero_pos))
        for index in range(slot_count):
            unit = ordered_units[index] if index < len(ordered_units) else None
            features.append(self._unit_feature(unit, lane, hero_pos, 1.0 if unit else 0.0))
        return features

    def _global_feature(
        self,
        frame_state,
        main_hero,
        enemy_hero,
        main_tower,
        enemy_tower,
        my_soldiers,
        enemy_soldiers,
        lane,
    ):
        hero_pos = self._position(main_hero)
        enemy_pos = self._position(enemy_hero)
        hero_progress, hero_offset = self._lane_coord(hero_pos, lane)
        main_config = int(self._value(main_hero, "config_id"))
        enemy_config = int(self._value(enemy_hero, "config_id"))
        self_hp = self._hp_ratio(main_hero)
        enemy_hp = self._hp_ratio(enemy_hero)
        self_money = self._resource(main_hero, "money")
        enemy_money = self._resource(enemy_hero, "money")
        self_exp = self._resource(main_hero, "exp")
        enemy_exp = self._resource(enemy_hero, "exp")
        self_tower_hp = self._hp_ratio(main_tower)
        enemy_tower_hp = self._hp_ratio(enemy_tower)
        has_ally_under_tower = self._has_ally_minion_under_enemy_tower(my_soldiers, enemy_tower)
        enemy_close = self._dist_ratio(hero_pos, enemy_pos) < 0.18 and enemy_hp > 0.25
        safe_to_push = bool(has_ally_under_tower and self_hp >= 0.30 and not enemy_close)
        skill_state_features = self._skill_state_features(main_hero)

        return [
            self._ratio(self._value(frame_state, "frame_no"), GameConfig.MAX_FRAME_NO),
            self._ratio(self._value(frame_state, "frame_no"), GameConfig.MAX_FRAME_NO),
            1.0 if self_hp < 0.30 else 0.0,
            1.0 if enemy_hp < 0.30 else 0.0,
            self._advantage01(self_hp - enemy_hp),
            self._min_max(self_money - enemy_money, -GameConfig.MAX_GOLD_DIFF, GameConfig.MAX_GOLD_DIFF),
            self._min_max(self_exp - enemy_exp, -GameConfig.MAX_EXP_DIFF, GameConfig.MAX_EXP_DIFF),
            self._advantage01(self_tower_hp - enemy_tower_hp),
            1.0 if any(self._hp_ratio(soldier) < 0.20 for soldier in enemy_soldiers) else 0.0,
            1.0 if safe_to_push else 0.0,
            1.0 if has_ally_under_tower else 0.0,
            1.0 if enemy_close else 0.0,
            1.0 if enemy_hero else 0.0,
            1.0 if my_soldiers or enemy_soldiers else 0.0,
            self._ratio(len(my_soldiers), GameConfig.SOLDIER_SLOT_COUNT),
            self._ratio(len(enemy_soldiers), GameConfig.SOLDIER_SLOT_COUNT),
            self._clip(hero_progress, 0.0, 1.0),
            self._clip(hero_offset / 40000.0, -1.0, 1.0),
            1.0 if main_config == 112 else 0.0,
            1.0 if main_config == 133 else 0.0,
            1.0 if enemy_config == 112 else 0.0,
            1.0 if enemy_config == 133 else 0.0,
            1.0 if main_config == 112 and enemy_config == 112 else 0.0,
            1.0 if main_config == 112 and enemy_config == 133 else 0.0,
            1.0 if main_config == 133 and enemy_config == 112 else 0.0,
            1.0 if main_config == 133 and enemy_config == 133 else 0.0,
            self._ratio(self._value(enemy_tower, "attack_range"), 20000.0),
            1.0,
            skill_state_features["skill1_usable"],
            skill_state_features["skill1_cd_ratio"],
            skill_state_features["skill2_usable"],
            skill_state_features["skill2_cd_ratio"],
            skill_state_features["skill3_usable"],
            skill_state_features["skill3_cd_ratio"],
            skill_state_features["summoner_usable"],
            skill_state_features["summoner_cd_ratio"],
            1.0 if self._has_berserk_buff(main_hero) else 0.0,
            1.0 if has_luban_output_buff(main_hero) else 0.0,
        ]

    def _lane_basis(self, start_unit, end_unit):
        start = self._position(start_unit)
        end = self._position(end_unit)
        vx, vz = end[0] - start[0], end[1] - start[1]
        length = max((vx * vx + vz * vz) ** 0.5, 1.0)
        return {"start": start, "end": end, "ux": vx / length, "uz": vz / length, "length": length}

    def _lane_coord(self, pos, lane):
        px, pz = pos[0] - lane["start"][0], pos[1] - lane["start"][1]
        forward = px * lane["ux"] + pz * lane["uz"]
        lateral = px * (-lane["uz"]) + pz * lane["ux"]
        return forward / lane["length"], lateral

    def _relative_lane_coord(self, pos, origin, lane):
        px, pz = pos[0] - origin[0], pos[1] - origin[1]
        forward = px * lane["ux"] + pz * lane["uz"]
        lateral = px * (-lane["uz"]) + pz * lane["ux"]
        return forward, lateral

    def _has_ally_minion_under_enemy_tower(self, soldiers, enemy_tower):
        if not enemy_tower:
            return False
        tower_pos = self._position(enemy_tower)
        tower_range = self._value(enemy_tower, "attack_range") or 9000.0
        return any(math.dist(self._position(soldier), tower_pos) <= tower_range for soldier in soldiers)

    def _skill_summary(self, unit):
        skill_state = unit.get("skill_state") if isinstance(unit, dict) else None
        if not skill_state:
            return 0.0, 0.0, 0.0
        slots = (
            skill_state.get("slot_states")
            or skill_state.get("skill_slot")
            or skill_state.get("skill_slot_state")
            or skill_state.get("slots")
            or []
        )
        usable, cooldown, cooldown_max, count = 0.0, 0.0, 0.0, 0
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            usable = max(usable, 1.0 if slot.get("usable", False) else 0.0)
            cooldown += float(slot.get("cooldown", 0.0) or 0.0)
            cooldown_max += float(slot.get("cooldown_max", 0.0) or 0.0)
            count += 1
        if count <= 0:
            return usable, 0.0, 0.0
        return usable, self._ratio(cooldown / count, 60000.0), self._ratio(cooldown_max / count, 60000.0)

    def _skill_state_features(self, unit):
        features = {
            "skill1_usable": 0.0,
            "skill1_cd_ratio": 1.0,
            "skill2_usable": 0.0,
            "skill2_cd_ratio": 1.0,
            "skill3_usable": 0.0,
            "skill3_cd_ratio": 1.0,
            "summoner_usable": 0.0,
            "summoner_cd_ratio": 1.0,
        }
        skill_state = unit.get("skill_state") if isinstance(unit, dict) else None
        if not skill_state:
            return features
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
            key_prefix = self._slot_feature_prefix(slot, index)
            if key_prefix is None:
                continue
            usable = self._slot_usable(slot)
            cooldown = self._value_any(slot, ("cooldown", "cd", "coolDown", "cooldownMs"))
            cooldown_max = self._value_any(slot, ("cooldown_max", "maxCooldown", "max_cooldown", "cooldownMax"))
            if cooldown_max <= 0.0:
                cd_ratio = self._ratio(cooldown, 60000.0)
            else:
                cd_ratio = self._clip(cooldown / cooldown_max, 0.0, 1.0)
            features[f"{key_prefix}_usable"] = usable
            features[f"{key_prefix}_cd_ratio"] = cd_ratio
        return features

    def _slot_feature_prefix(self, slot, index):
        slot_type = int(self._value_any(slot, ("slotType", "slot_type")))
        skill_id = int(self._value_any(slot, ("skillID", "skill_id", "configId", "config_id")))
        if slot_type in (1, 2, 3):
            return f"skill{slot_type}"
        if slot_type == GameConfig.BUTTON_SUMMONER or skill_id == GameConfig.FIXED_SUMMONER_SKILL_ID:
            return "summoner"
        skill_no = GameConfig.skill_no_from_id(skill_id)
        if skill_no is not None:
            return f"skill{skill_no}"
        if index in (0, 1, 2):
            return f"skill{index + 1}"
        return None

    def _slot_usable(self, slot):
        if "usable" in slot:
            return 1.0 if slot.get("usable") else 0.0
        cooldown = self._value_any(slot, ("cooldown", "cd", "coolDown", "cooldownMs"))
        return 1.0 if cooldown <= 0.0 else 0.0

    def _has_berserk_buff(self, unit):
        return bool(collect_buff_config_ids(unit) & BERSERK_BUFF_IDS)

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
            if npc.get("camp") == camp and self._is_tower(npc):
                return npc
        return None

    def _get_enemy_tower(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") != camp and self._is_tower(npc):
                return npc
        return None

    def _get_crystal(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") == camp and self._is_crystal(npc):
                return npc
        return None

    def _get_enemy_crystal(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") != camp and self._is_crystal(npc):
                return npc
        return None

    def _soldiers(self, frame_state, camp):
        return [
            npc
            for npc in frame_state.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") == camp
            and not self._is_organ(npc)
            and self._value(npc, "hp") > 0
        ]

    def _enemy_soldiers(self, frame_state, camp):
        return [
            npc
            for npc in frame_state.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") != camp
            and not self._is_organ(npc)
            and self._value(npc, "hp") > 0
        ]

    def _sort_key(self, unit, hero_pos):
        return (math.dist(self._position(unit), hero_pos), self._unit_id(unit))

    def _is_tower(self, unit):
        return int(self._value(unit, "sub_type")) in GameConfig.TOWER_SUB_TYPES

    def _is_crystal(self, unit):
        return int(self._value(unit, "sub_type")) in GameConfig.CRYSTAL_SUB_TYPES

    def _is_organ(self, unit):
        return self._is_tower(unit) or self._is_crystal(unit)

    def _enemy_camp(self):
        if self.camp == GameConfig.CAMP_BLUE:
            return GameConfig.CAMP_RED
        if self.camp == GameConfig.CAMP_RED:
            return GameConfig.CAMP_BLUE
        return 0

    def _position(self, unit):
        if not unit:
            return 0.0, 0.0
        location = unit.get("location", {})
        return float(location.get("x", 0.0) or 0.0), float(location.get("z", 0.0) or 0.0)

    def _hp_ratio(self, unit):
        max_hp = self._value(unit, "max_hp")
        if max_hp <= 0:
            return 0.0
        return self._clip(self._value(unit, "hp") / max_hp, 0.0, 1.0)

    def _ep_ratio(self, unit):
        max_ep = self._value(unit, "max_ep")
        if max_ep <= 0:
            return 0.0
        return self._clip(self._value(unit, "ep") / max_ep, 0.0, 1.0)

    def _camp_visible(self, unit):
        if not isinstance(unit, dict):
            return 0.0
        visible = unit.get("camp_visible", [])
        if not isinstance(visible, (list, tuple)):
            return 0.0
        index = self.camp - 1
        if index < 0 or index >= len(visible):
            return 0.0
        return 1.0 if visible[index] else 0.0

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
        if isinstance(unit, dict):
            return float(unit.get(key, 0.0) or 0.0)
        return 0.0

    def _unit_id(self, unit):
        for key in ("runtime_id", "actor_runtime_id", "obj_id"):
            value = unit.get(key)
            if value is not None:
                return int(value)
        pos = self._position(unit)
        return int(pos[0] * 10 + pos[1])

    def _dist_ratio(self, a, b):
        return self._ratio(math.dist(a, b), 80000.0)

    def _advantage01(self, value):
        return self._clip((value + 1.0) / 2.0, 0.0, 1.0)

    def _min_max(self, value, min_value, max_value):
        if max_value <= min_value:
            return 0.0
        return self._clip((float(value) - min_value) / (max_value - min_value), 0.0, 1.0)

    def _ratio(self, value, denom):
        if denom <= 0:
            return 0.0
        return self._clip(float(value) / float(denom), -1.0, 1.0)

    def _clip(self, value, min_value, max_value):
        return max(min_value, min(max_value, float(value)))
