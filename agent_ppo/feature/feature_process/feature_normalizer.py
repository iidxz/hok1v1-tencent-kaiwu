#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class FeatureNormalizer:
    def __init__(self):
        pass

    def one_hot(self, value, value_list, compare_method):
        # one hot encoding
        # one hot 编码
        if compare_method != "eq":
            raise ValueError("Unsupported compare method: " + compare_method)
        return [1 if value == v else 0 for v in value_list]

    def min_max(self, value, min_value, max_value):
        # Normalize the value based on min and max values
        # 根据最大值和最小值归一化
        if value <= min_value:
            return 0
        elif value >= max_value:
            return 1
        else:
            return (value - min_value) / (max_value - min_value)

    def parse_config(self, config_list):
        config_dict = {}
        for config_str in config_list:
            parts = config_str.split(":")
            feature_name = parts[0]
            method = parts[1]
            if method == "one_hot":
                values = list(map(int, parts[2:-1]))
                compare_method = parts[-1]
                config_dict[feature_name] = (self.one_hot, values, compare_method)
            elif method == "min_max":
                min_value = int(parts[2])
                max_value = int(parts[3])
                config_dict[feature_name] = (self.min_max, min_value, max_value)
            else:
                raise ValueError("Unsupported method: " + method)
        return config_dict
