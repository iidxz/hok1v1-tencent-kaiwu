#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict
from typing import List

from agent_ppo.conf.conf import Config, DimConfig, GameConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = DimConfig.DIM_OF_FEATURE[0]
        self.unit_feature_dim = GameConfig.UNIT_FEATURE_DIM
        self.global_feature_dim = GameConfig.GLOBAL_FEATURE_DIM
        self.unit_slot_count = DimConfig.UNIT_SLOT_COUNT
        self.unit_flat_dim = self.unit_slot_count * self.unit_feature_dim
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)

        group_dim = Config.GROUP_EMBED_DIM
        self.group_slices = {
            "hero_frd": (0, 1),
            "hero_emy": (1, 2),
            "soldier_frd": (2, 6),
            "soldier_emy": (6, 10),
            "organ_frd": (10, 12),
            "organ_emy": (12, 14),
        }
        self.unit_mlps = ModuleDict(
            {
                name: MLP([self.unit_feature_dim, Config.UNIT_EMBED_DIM, group_dim], f"{name}_unit_mlp")
                for name in self.group_slices
            }
        )

        encoder_input_dim = group_dim * len(self.group_slices) + self.global_feature_dim
        self.public_mlp = MLP([encoder_input_dim, 512, self.lstm_unit_size], "public_mlp", non_linearity_last=True)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [self.lstm_unit_size, 256, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.value_mlp = MLP([self.lstm_unit_size, 256, 1], "hero_value_mlp")

        self.target_unit_mlp = MLP(
            [self.unit_feature_dim, Config.UNIT_EMBED_DIM, self.target_embed_dim],
            "target_unit_mlp",
            non_linearity_last=True,
        )
        self.target_query_mlp = MLP(
            [self.lstm_unit_size, 128, self.target_embed_dim],
            "target_query_mlp",
            non_linearity_last=True,
        )
        self.target_bias_mlp = MLP([self.lstm_unit_size, 128, self.label_size_list[-1]], "target_bias_mlp")
        self.target_head = nn.Linear(self.lstm_unit_size, self.label_size_list[-1])

        self.target_fusion_mlp = MLP(
            [self.lstm_unit_size + self.target_embed_dim, 256, self.label_size_list[-1]],
            "target_fusion_mlp",
        )

        self.last_attention_stats = self._zero_attention_stats()

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        batch_size = max(int(lstm_hidden_init.shape[0]), 1)
        seq_len = max(int(feature_vec.shape[0] // batch_size), 1)

        encoded = self._encode_features(feature_vec)
        encoded_seq = encoded.reshape(batch_size, seq_len, self.lstm_unit_size)

        lstm_cell = lstm_cell_init.reshape(batch_size, self.lstm_unit_size).unsqueeze(0)
        lstm_hidden = lstm_hidden_init.reshape(batch_size, self.lstm_unit_size).unsqueeze(0)
        lstm_out, (self.lstm_hidden_output, self.lstm_cell_output) = self.lstm(
            encoded_seq, (lstm_hidden, lstm_cell)
        )
        public_result = lstm_out.reshape(-1, self.lstm_unit_size)

        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            result_list.append(self.label_mlp[f"hero_label{label_index}_mlp"](public_result))
        result_list.append(self._target_logits(feature_vec, public_result))
        result_list.append(self.value_mlp(public_result))

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        return result_list

    def _encode_features(self, feature_vec):
        units, global_feature = self._split_feature(feature_vec)
        group_outputs = []
        for name, (start, end) in self.group_slices.items():
            group_units = units[:, start:end, :]
            group_embed = self.unit_mlps[name](group_units.reshape(-1, self.unit_feature_dim))
            group_embed = group_embed.reshape(group_units.shape[0], end - start, -1)
            group_outputs.append(self._masked_max_pool(group_embed, group_units[:, :, 0]))
        return self.public_mlp(torch.cat(group_outputs + [global_feature], dim=1))

    def _split_feature(self, feature_vec):
        units_flat = feature_vec[:, : self.unit_flat_dim]
        global_feature = feature_vec[:, self.unit_flat_dim : self.unit_flat_dim + self.global_feature_dim]
        units = units_flat.reshape(-1, self.unit_slot_count, self.unit_feature_dim)
        return units, global_feature

    def _masked_max_pool(self, values, mask):
        mask = mask.unsqueeze(-1)
        masked_values = values.masked_fill(mask <= 0.0, -1.0e9)
        pooled = torch.max(masked_values, dim=1).values
        return torch.where(torch.isfinite(pooled) & (pooled > -1.0e8), pooled, torch.zeros_like(pooled))

    def _target_logits(self, feature_vec, public_result):
        target_head_logits = self.target_head(public_result)

        if not GameConfig.USE_TARGET_ATTENTION:
            self.last_attention_stats = self._zero_attention_stats()
            return target_head_logits

        attention_logits, target_mask, target_context = self._target_attention_logits(
            feature_vec,
            public_result,
            return_mask=True,
            return_context=True,
        )

        fusion_input = torch.cat([public_result, target_context], dim=1)
        fusion_logits = self.target_fusion_mlp(fusion_input)

        target_logits = (
            (1.0 - GameConfig.ATTENTION_ALPHA) * target_head_logits
            + GameConfig.ATTENTION_ALPHA * fusion_logits
        )

        self.last_attention_stats = self._attention_stats(
            target_head_logits,
            target_logits,
            attention_logits,
            target_mask,
        )
        return target_logits

    def _target_attention_logits(self, feature_vec, public_result, return_mask=False, return_context=False):
        units, _ = self._split_feature(feature_vec)
        zero = torch.zeros_like(units[:, 0:1, :])

        target_units = torch.cat(
            [
                zero,
                units[:, 1:2, :],
                units[:, 0:1, :],
                units[:, 6:10, :],
                units[:, 12:13, :],
                zero,
            ],
            dim=1,
        )

        target_mask = torch.cat(
            [
                torch.ones_like(units[:, 0:1, 0]),
                units[:, 1:2, 0],
                units[:, 0:1, 0],
                units[:, 6:10, 0],
                units[:, 12:13, 0],
                torch.zeros_like(units[:, 0:1, 0]),
            ],
            dim=1,
        )

        target_embed = self.target_unit_mlp(target_units.reshape(-1, self.unit_feature_dim))
        target_embed = target_embed.reshape(-1, GameConfig.TARGET_SLOT_COUNT, self.target_embed_dim)

        query = self.target_query_mlp(public_result).unsqueeze(1)
        dot_logits = torch.sum(target_embed * query, dim=2) / np.sqrt(float(self.target_embed_dim))
        attention_logits = dot_logits + self.target_bias_mlp(public_result)

        valid_logits = attention_logits.masked_fill(target_mask <= 0.0, -1.0e9)
        attn_weights = torch.softmax(valid_logits, dim=1)

        target_context = torch.sum(
            target_embed * attn_weights.unsqueeze(-1),
            dim=1,
        )

        if return_mask and return_context:
            return attention_logits, target_mask, target_context
        if return_mask:
            return attention_logits, target_mask
        if return_context:
            return attention_logits, target_context
        return attention_logits

    def _attention_stats(self, target_head_logits, target_logits, attention_logits, target_mask):
        valid_logits = attention_logits.masked_fill(target_mask <= 0.0, -1.0e9)
        probs = torch.softmax(valid_logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=1)
        max_prob = probs.max(dim=1).values
        changed = (torch.argmax(target_head_logits, dim=1) != torch.argmax(target_logits, dim=1)).float()
        return {
            "attention_entropy": float(entropy.detach().mean().cpu().item()),
            "attention_max_prob": float(max_prob.detach().mean().cpu().item()),
            "attention_target_hero_prob": float(probs[:, 1].detach().mean().cpu().item()),
            "attention_target_soldier_prob": float(probs[:, 3:7].detach().sum(dim=1).mean().cpu().item()),
            "attention_target_tower_prob": float(probs[:, 7].detach().mean().cpu().item()),
            "target_changed_by_attention_rate": float(changed.detach().mean().cpu().item()),
        }

    def _zero_attention_stats(self):
        return {
            "attention_entropy": 0.0,
            "attention_max_prob": 0.0,
            "attention_target_hero_prob": 0.0,
            "attention_target_soldier_prob": 0.0,
            "attention_target_tower_prob": 0.0,
            "target_changed_by_attention_rate": 0.0,
        }

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=value_result.device)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                boundary = torch.pow(torch.tensor(10.0, device=value_result.device), torch.tensor(20.0, device=value_result.device))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary
                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )
                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )
                label_probability = label_exp_logits / label_exp_logits.sum(1, keepdim=True)
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                ratio = torch.exp(policy_log_p - old_policy_log_p).clamp(0.0, 3.0)
                surr1 = ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                denom = torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train),
                    torch.tensor(1.0, device=value_result.device),
                )
                self.policy_cost = self.policy_cost - torch.sum(
                    torch.minimum(surr1, surr2) * weight_list[task_index].float() * frame_is_train
                ) / denom

        entropy_loss_list = []
        current_entropy_loss_index = 0
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                denom = torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train),
                    torch.tensor(1.0, device=value_result.device),
                )
                temp_entropy_loss = -torch.sum(temp_entropy_loss * weight_list[task_index].float() * frame_is_train) / denom
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=value_result.device))

        self.entropy_cost = torch.stack(entropy_loss_list).sum()
        self.entropy_cost_list = entropy_loss_list
        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
