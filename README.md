# HOK 1v1 PPO 智能体

这是一个运行在腾讯开悟强化学习环境中的 1v1 对战智能体项目。当前主实现位于 `agent_ppo`，目标是在英雄对抗、兵线运营、防御塔攻防和资源争夺之间学习稳定的长期策略。

项目使用 PPO actor-critic、结构化单位特征、LSTM 时序建模、目标注意力、合法动作掩码和分阶段奖励塑形。当前训练英雄为 ID `112`（鲁班七号）和 ID `133`（狄仁杰），并针对两名英雄设计了不同的技能探索先验和技能奖励。


环境接口、数据协议、动作定义、平台配置和完整开发流程详见[腾讯开悟 HOK 1v1 官方开发指南](https://tencentarena.com/p/competition/doc?path=https%3A%2F%2Ftencentarena.com%2Fdocs%2Fp-competition-hok1v1%2F61.1.4%2F&expId=11425&stageId=444)。本 README 重点说明本仓库在官方环境之上实现的特征工程、策略网络、奖励设计和英雄训练策略；若环境接口或平台规则发生变化，请以官方开发指南为准。

## 1. 项目目标

智能体需要在不完全观测的 1v1 环境中完成以下决策：

- 什么时候补刀、清线或跟随兵线推进；
- 什么时候攻击英雄，什么时候优先攻击防御塔；
- 低血量、塔下危险或处于劣势时如何撤退；
- 如何根据英雄、距离、冷却和目标选择技能；
- 如何利用召唤师技能、河道资源和恢复点；
- 如何对抗不同英雄组合与不同历史版本的对手模型。

当前实现不是单纯依靠稀疏胜负奖励，而是把“活下来、建立经济优势、形成安全推塔窗口并最终拆塔”拆成多个可观测的学习信号。

## 2. 核心能力

- PPO clipped objective、GAE 和状态价值估计；
- 16 步序列、512 维单层 LSTM；
- 14 个单位槽位和 38 维全局特征，共 374 维输入；
- 英雄、小兵和建筑分组编码与 masked max pooling；
- 六分支层次化动作空间；
- 面向敌方英雄、小兵和防御塔的目标注意力；
- legal action 与 sub-action mask 双重动作约束；
- 两英雄混合阵容训练和历史对手池轮换；
- 英雄专属技能探索、技能后摇衔接和错误施法抑制；
- 细粒度奖励、行为诊断和分英雄/分对局监控。

## 3. 代码结构

```text
.
├── agent_ppo/
│   ├── agent.py                    # 推理、动作采样、技能探索和模型读写
│   ├── algorithm/algorithm.py      # 反向传播、梯度裁剪和学习率更新
│   ├── conf/
│   │   ├── conf.py                 # 网络、PPO、奖励、英雄和探索参数
│   │   ├── train_env_conf.toml     # 阵容、对手池、评估和监控配置
│   │   └── monitor_builder.py      # 训练监控面板
│   ├── feature/
│   │   ├── definition.py           # Frame、GAE 和 LSTM 样本拼接
│   │   ├── reward_process.py       # 奖励状态提取与奖励计算
│   │   ├── buff_constants.py       # BUFF、恢复点和河道单位常量
│   │   └── feature_process/        # 结构化特征生成
│   ├── model/model.py              # 分组编码、LSTM、策略头和价值头
│   └── workflow/train_workflow.py  # 对局、采样、评估、对手与监控循环
├── agent_diy/                      # 最小自定义算法模板
├── conf/                           # 应用注册与 replay buffer 配置
├── kaiwu.json                      # 平台可加载的历史对手模型 ID
└── train_test.py                   # 平台训练冒烟测试入口
```

## 4. 从环境状态到 374 维特征

### 4.1 单位槽位

每个单位使用 24 维特征。当前固定 14 个槽位：

| 分组 | 槽位数 | 内容 |
| --- | ---: | --- |
| `hero_frd` | 1 | 己方英雄 |
| `hero_emy` | 1 | 敌方英雄 |
| `soldier_frd` | 4 | 距离己方英雄最近的 4 个己方小兵 |
| `soldier_emy` | 4 | 距离己方英雄最近的 4 个敌方小兵 |
| `organ_frd` | 2 | 己方防御塔与水晶 |
| `organ_emy` | 2 | 敌方防御塔与水晶 |

因此单位部分为：

```text
14 slots × 24 features = 336 dimensions
336 unit dimensions + 38 global dimensions = 374 dimensions
```

代码中的维度定义：

```python
class GameConfig:
    UNIT_FEATURE_DIM = 24
    GLOBAL_FEATURE_DIM = 38
    SOLDIER_SLOT_COUNT = 4
    ORGAN_SLOT_COUNT = 2

class DimConfig:
    UNIT_SLOT_COUNT = 1 + 1 + 4 + 4 + 2 + 2
    DIM_OF_FEATURE = [UNIT_SLOT_COUNT * UNIT_FEATURE_DIM + GLOBAL_FEATURE_DIM]
```

### 4.2 单位特征

`FeatureProcess._unit_feature()` 为英雄、小兵和建筑使用统一结构，主要包含：

- 是否存在、是否存活、是否可见；
- HP、EP、等级、经验、经济；
- 沿兵线方向的绝对进度与横向偏移；
- 相对己方英雄的前后、左右位置和距离；
- 阵营标记；
- 攻击范围、视野、移动速度和攻击速度；
- 技能是否可用、当前冷却比例和最大冷却；
- 英雄 ID 与单位子类型。

不存在的槽位填充为全零，并由第一个 `exists` 特征作为 pooling mask。

```python
slots = [
    self._unit_feature(main_hero, lane, hero_pos, 1.0, self.camp),
    self._unit_feature(enemy_hero, lane, hero_pos, 1.0, self._enemy_camp()),
]
slots.extend(self._slot_features(my_soldiers, 4, lane, hero_pos))
slots.extend(self._slot_features(enemy_soldiers, 4, lane, hero_pos))
slots.extend(self._slot_features([main_tower, main_crystal], 2, lane, hero_pos))
slots.extend(self._slot_features([enemy_tower, enemy_crystal], 2, lane, hero_pos))
```

### 4.3 全局特征

38 维全局特征用于描述单个单位槽位难以表达的局势：

- 对局帧进度；
- 双方低血量状态；
- HP、经济、经验和防御塔血量优势；
- 是否存在可补刀小兵；
- 是否有己方兵线进入敌塔；
- 是否为安全推塔窗口；
- 敌方是否靠近、是否可见；
- 双方兵线数量和己方兵线进度；
- 112/133 英雄及四种对局组合编码；
- 敌塔攻击范围；
- 1、2、3 技能与召唤师技能的可用性和冷却比例；
- 狂暴状态和鲁班输出 BUFF 状态。

位置特征使用己方塔到敌方塔的兵线方向建立局部坐标系，使红蓝方在模型视角中保持一致。

## 5. 策略网络

### 5.1 分组编码

不同类别的单位使用独立 MLP：

```python
self.group_slices = {
    "hero_frd": (0, 1),
    "hero_emy": (1, 2),
    "soldier_frd": (2, 6),
    "soldier_emy": (6, 10),
    "organ_frd": (10, 12),
    "organ_emy": (12, 14),
}

self.unit_mlps = ModuleDict({
    name: MLP([24, 64, 96], f"{name}_unit_mlp")
    for name in self.group_slices
})
```

同一组存在多个单位时使用 masked max pooling。这样既保留固定大小的网络输入，又不会让补零槽位影响聚合结果。

### 5.2 公共干线与 LSTM

六个分组各输出 96 维表示，与 38 维全局特征拼接：

```text
6 × 96 + 38 = 614
614 -> 512 -> 512 -> LSTM(512)
```

训练样本按连续 16 帧组成序列。推理时保留并滚动更新 `lstm_hidden` 与 `lstm_cell`，用于表示近期交战、技能命中、撤退和推进状态。

### 5.3 动作空间

策略输出六个离散分支：

| 动作头 | 维度 | 含义 |
| --- | ---: | --- |
| button | 12 | 无动作、移动、普攻、技能、回城等 |
| move X | 16 | 移动方向 X |
| move Z | 16 | 移动方向 Z |
| skill X | 16 | 技能方向 X |
| skill Z | 16 | 技能方向 Z |
| target | 9 | 空目标、英雄、小兵、防御塔和野怪 |

目标合法性依赖 button，因此环境提供的目标 mask 在内部按 `12 × 9` 展开。采样时先屏蔽非法动作，再从合法概率中采样；评估时选择合法动作中的最大概率项。

### 5.4 目标注意力

目标头除了普通线性 head，还从候选目标单位中计算注意力上下文。候选顺序为：

```text
None, Enemy, Self, Enemy Soldier × 4, Enemy Tower, Monster
```

当前 `Monster` 没有对应单位 embedding，mask 为 0；其他不存在或不可见目标也由 mask 排除。

最终目标 logits 为基础目标头和注意力融合头的加权结果：

```python
target_logits = (
    (1.0 - GameConfig.ATTENTION_ALPHA) * target_head_logits
    + GameConfig.ATTENTION_ALPHA * fusion_logits
)
```

当前 `ATTENTION_ALPHA = 0.15`。训练监控会记录注意力熵、最大概率、英雄/小兵/塔注意力概率，以及注意力是否改变最终目标选择。

## 6. PPO 训练

### 6.1 GAE

`FrameCollector` 从轨迹末端反向计算 TD 残差和 GAE：

```python
delta = reward + gamma * next_value - value
gae = delta + gamma * lamda * gae
advantage = gae
value_target = gae + value
```

当前参数：

| 参数 | 当前值 |
| --- | ---: |
| `gamma` | 0.995 |
| `lambda` | 0.95 |
| LSTM 序列长度 | 16 |

### 6.2 PPO loss

每个动作头独立计算概率比率，并使用同一个 advantage：

```python
ratio = exp(new_log_prob - old_log_prob).clamp(0.0, 3.0)
surr1 = ratio * advantage
surr2 = ratio.clamp(1.0 - 0.2, 1.0 + 0.2) * advantage
policy_loss = -mean(min(surr1, surr2))
```

总损失由 value loss、policy loss 和 entropy cost 构成：

```python
loss = value_cost + policy_cost + beta * entropy_cost
```

当前优化参数：

| 参数 | 当前值 |
| --- | ---: |
| 初始学习率 | `2e-4` |
| 目标学习率 | `5e-5` |
| 线性衰减步数 | `4000` |
| PPO clip | `0.2` |
| 初始 entropy beta | `0.045` |
| 梯度范数上限 | `0.5` |
| 优化器 | Adam，`betas=(0.9, 0.999)` |

只有环境标记为可训练、且对应 sub-action 有效的动作头才参与策略损失。

## 7. 奖励设计

### 7.1 总体形式

`GameRewardManager` 先从前后两帧提取局势差分和当前行为，再计算多个原始奖励项：

```text
R_t = clip(sum(weight_i × reward_item_i), -100, 100)
```

核心代码：

```python
reward_items = self._compute_reward_items(self.prev_stats, cur_stats)
reward_sum = 0.0
for name, value in reward_items.items():
    reward_sum += value * self.reward_weights.get(name, 0.0)
reward_items["reward_sum"] = self._clip(reward_sum, -100.0, 100.0)
```

这种设计把“检测条件”和“最终强度”分开：`reward_process.py` 决定某行为的原始奖励，`REWARD_WEIGHT_DICT` 决定它对总回报的权重。

### 7.2 基础差分奖励

基础奖励直接反映局势变化：

```python
enemy_hp_drop = prev["enemy_hp_ratio"] - cur["enemy_hp_ratio"]
self_hp_drop = prev["self_hp_ratio"] - cur["self_hp_ratio"]

reward_hp = clip(enemy_hp_drop - self_hp_drop, -1, 1)
reward_tower = clip(enemy_tower_drop - self_tower_drop, -1, 1)
reward_gold = clip(self_gold_delta - enemy_gold_delta, -200, 200)
reward_exp = clip(self_exp_delta - enemy_exp_delta, -200, 200)
```

主要基础权重：

| 奖励项 | 权重 | 设计目的 |
| --- | ---: | --- |
| `reward_hp` | 2.0 | 鼓励有利换血 |
| `reward_tower` | 14.0 | 把拆塔设为最重要的连续目标 |
| `reward_gold` | 0.008 | 建立经济优势 |
| `reward_exp` | 0.008 | 建立等级与经验优势 |
| `reward_last_hit` | 0.5 | 鼓励有效补刀 |
| `reward_kill` | 1.0 | 击杀奖励 |
| `reward_death` | -1.0 | 死亡惩罚 |
| `reward_forward` | 0.03 | 小幅鼓励沿兵线推进 |
| `reward_win` | 3.0 | 推掉敌塔的终局奖励 |

补刀不是根据经济变化猜测，而是读取 `frame_action.dead_action`，确认死亡单位是小兵且 killer 的 runtime ID 为己方英雄，并使用事件集合去重。

### 7.3 推塔与兵线奖励

安全推塔窗口要求：

- 敌塔仍存活；
- 己方 HP 高于阈值；
- 己方兵线进塔，或自己已在可攻击塔的距离内；
- 当前不属于无兵线越塔危险。

主要行为：

| 行为 | 原始奖励 | 权重 | 最终作用 |
| --- | ---: | ---: | --- |
| 安全窗口攻击防御塔 | `+0.04` | 1.0 | 主动选择塔目标 |
| 安全窗口实际打掉塔血 | `+0.04` | 1.0 | 奖励真实推进效果 |
| 强推窗口普攻塔 | `+0.08` | 1.2 | 强化优先推塔 |
| 强推窗口错误追击非残血英雄 | `-0.08` | 1.0 | 避免放弃拆塔机会 |
| 清线后跟随己方兵线前进 | `+0.03` | 1.2 | 减少清线后发呆 |
| 清线后停滞 | `-0.035` | 1.6 | 提高地图节奏 |
| 己塔下清理敌兵 | `+0.06` | 2.0 | 强化防守意识 |
| 己塔受压时忽略兵线 | `-0.06` | 2.0 | 避免错误追人/打野 |

无己方兵线进入敌塔攻击范围时，进入敌塔会得到 `-0.04` 原始奖励；如果还继续攻击英雄，额外得到 `-0.20`。受到塔伤后向外移动会得到小额撤退奖励，继续停留则继续受罚。

### 7.4 对线与走 A

奖励管理器显式比较前后两帧位置：

- 交战中移动距离不小于 300，奖励 `combat_kite`；
- 敌人较近且移动不足 220，惩罚 `combat_standstill`；
- 攻击有效目标同时移动不小于 180，奖励 `attack_move`；
- 非推塔攻击时原地站立，惩罚 `attack_standstill`；
- 敌人在安全攻击范围内且不是推塔窗口时，普攻目标选敌方英雄得到奖励，选错目标受到惩罚。

这样做的目的不是用规则替代策略，而是让 PPO 在早期更快区分“持续输出”和“站桩挨打”。

### 7.5 低血量与撤退

当己方 HP 低于 35% 时：

- 向恢复点移动得到小额奖励；
- 实际回血、治疗或回城得到奖励；
- 低血量继续原地停留或无意义攻击受到惩罚。

`retreat_window` 在以下情况成立：

```text
self_hp < 30%
or (hp_advantage < -25% and enemy_distance <= 12000)
```

撤退窗口中向己塔方向移动得到 `+0.03`；在敌方并非残血且自己不在己塔保护下时继续进攻，得到 `-0.02`；满足安全回城条件时执行回城得到 `+0.03`。

### 7.6 技能通用奖励

技能奖励同时参考动作、技能统计和敌方 HP 变化，避免只根据按键判断是否命中：

```python
skill_used = action_button in SKILL_BUTTONS or used_total_increased
skill_hit = hit_total_increased or (skill_used and enemy_hp_drop > 0.001)
```

通用技能奖励包括：

- 技能命中英雄；
- 技能造成有效伤害；
- 技能命中后衔接普攻；
- 技能后利用安全窗口攻击防御塔；
- 狂暴生效期间继续攻击英雄；
- 合理使用 1、2、3 技能；
- 错误目标、错误距离、塔下乱放技能和无场景空放惩罚。

### 7.7 英雄专属技能奖励

#### 鲁班七号（112）

- 1 技能：目标为敌方英雄或小兵、且不在危险塔区时视为合理施法；
- 2 技能：用于敌方残血收割、敌人贴脸自保或狂暴后追击；
- 3 技能：用于清兵、压塔、推塔窗口或区域压制；
- 3 技能安全打小兵或塔不会因为“没有命中英雄”被错误惩罚；
- 低血量越塔、无兵线进入塔区、无有效目标乱放会受到惩罚。

合理 2 技能施法原始奖励为 `+0.04`，合理 3 技能施法为 `+0.07`；对应权重分别为 1.8 和 2.4。

#### 狄仁杰（133）

- 1 技能：承担消耗和清线；
- 2 技能：被定义为防守技能，目标优先为自己或空目标；
- 2 技能在低血量、撤退、刚受到英雄伤害或敌人贴脸时使用不会受罚；
- 2 技能朝敌人释放会被视为错误用法；
- 3 技能用于 4500 到 12500 距离的中距离控制，目标必须是敌方英雄；
- 敌塔危险区、错误目标和过远空放会受到惩罚。

### 7.8 为什么同时使用正奖励和负奖励

只奖励“命中”会导致模型频繁乱放；只惩罚“空放”又会让模型不敢探索技能。因此当前设计使用三层约束：

1. legal action mask 保证动作在环境规则上可执行；
2. 技能探索先验提供少量高价值动作样本；
3. 奖励同时区分合理施法、无效施法和危险施法。

## 8. 英雄训练策略

### 8.1 阵容覆盖

当前训练阵容直接定义在 `GameConfig.TRAIN_LINEUP_PAIRS`：

```python
TRAIN_LINEUP_PAIRS = (
    (112, 112),
    (112, 133),
    (133, 112),
    (133, 112),
    (133, 133),
)
```

训练工作流按顺序循环这些阵容。由于 `(133, 112)` 出现两次，当前实际采样比例约为：

| 己方 vs 敌方 | 比例 |
| --- | ---: |
| 112 vs 112 | 20% |
| 112 vs 133 | 20% |
| 133 vs 112 | 40% |
| 133 vs 133 | 20% |

这意味着当前配置有意或事实上加强了狄仁杰对鲁班的训练。如果希望四种对局完全均衡，应删除重复的 `(133, 112)`。

`auto_switch_monitor_side = true` 允许最新模型在蓝红双方之间切换，减少固定出生方向带来的策略偏差。

### 8.2 历史对手池

当前配置：

```toml
opponent_agent = "280471"
train_opponent_pool = ["280471", "279609", "278719"]

eval_interval = 10
eval_opponent_type = "280471"
eval_opponent_pool = ["280471"]
```

训练对手并非随机抽取，而是由 `OpponentSchedule` 顺序循环：

```python
opponent = pool[index % len(pool)]
index += 1
```

对局中 monitor side 加载 `latest` 模型；另一侧加载指定历史模型，并关闭该侧采样。这样 replay buffer 中主要保存最新策略产生的 on-policy 数据，而不是把固定历史对手的数据混入训练样本。

每 10 局插入一次评估对局，评估时使用确定性动作 `exploit()`，不采集训练样本，当前固定对手为 `280471`，便于保持评估曲线可比较。

### 8.3 固定召唤师技能

当前两名英雄都固定选择技能 ID `80110`，即狂暴：

```python
FIXED_SUMMONER_SKILL_ID = 80110

def init_config(self, config_data):
    return {
        hero_id: FIXED_SUMMONER_SKILL_ID
        for hero_id in config_data.get("my_heroes", [])
    }
```

奖励和特征会检测狂暴 BUFF，并鼓励在狂暴生效期间继续攻击敌方英雄。

### 8.4 鲁班技能探索先验

PPO 原始动作完成采样后，训练阶段会以较低概率注入符合场景的技能动作。当前概率来自 `GameConfig`：

| 场景 | 概率 | 条件摘要 |
| --- | ---: | --- |
| 1 技能消耗 | 2.5% | 敌人可见、中近距离、HP > 35%、无塔险 |
| 2 技能收割 | 4.5% | 敌人残血且较远/撤退，我方有血量或连招优势 |
| 2 技能防守 | 4.0% | 敌人贴脸，己方 HP < 55% 或刚受到伤害 |
| 3 技能压制 | 3.0% | 推塔窗口、敌兵聚集、兵线进塔、敌人在塔边或塔低血量 |

先验动作仍必须通过以下检查：

- 技能冷却比例不高于 2%；
- button、方向和 target 均通过合法动作 mask；
- 非防守技能在低血量撤退窗口不强制执行；
- 非防守 2/3 技能在 HP 不高于 45% 时被阻止；
- 无兵线塔下危险时被阻止。

鲁班 3 技能优先选择敌方小兵和防御塔；只有敌方残血且没有塔险时，才把敌方英雄加入候选目标。

### 8.5 狄仁杰技能探索先验

| 场景 | 概率 | 条件摘要 |
| --- | ---: | --- |
| 1 技能消耗 | 2.5% | 敌人可见、中近距离、HP > 35%、无塔险 |
| 2 技能防守 | 3.5% | HP < 55%、刚受伤、撤退窗口或敌人贴脸 |
| 3 技能控制 | 3.0% | 敌人可见、距离 4500~12500、HP > 45%、无塔险 |

狄仁杰 2 技能优先选择 `Self` 或 `None`；3 技能只优先敌方英雄。

### 8.6 探索兜底与技能衔接

如果没有场景先验候选，代码最多以 5% 概率从 ready 且 legal 的 2/3 技能中选择一个进行 probe。每帧最多 probe 一次，并继续受低血量和塔险条件约束。

技能命中后还会打开短期衔接窗口：

- 任意技能命中后，最多 10 帧内以 35% 概率衔接对英雄普攻；
- 鲁班 1 技能命中后，最多 10 帧内以 30% 概率优先衔接英雄、塔或小兵普攻；
- 所有衔接动作仍需通过合法动作和塔险检查。

这种方式只在训练探索阶段修改随机动作。评估阶段的 `exploit()` 完全由模型最大概率输出决定，因此评估结果反映模型是否已经学会这些行为，而不是依赖强制规则取胜。

## 9. 训练工作流

一次训练对局的主要流程：

```text
选择阵容
  -> 选择训练/评估对手
  -> 注入召唤师技能
  -> env.reset
  -> 加载 latest 与历史对手模型
  -> 特征处理与动作采样
  -> env.step
  -> 奖励计算与 Frame 保存
  -> 终局计算 GAE
  -> 每 16 帧拼接 LSTM 样本
  -> 发送 replay buffer
  -> learner 执行 PPO 更新
```

训练模式使用 `predict()` 进行随机采样；评估模式使用 `exploit()` 选择最大概率动作。评估数据不进入 replay buffer。

模型每隔约 1800 秒主动保存一次；平台侧同时依据 `dump_model_freq = 50` 管理模型输出。replay buffer 当前配置为：

| 参数 | 当前值 |
| --- | ---: |
| 容量 | 5000 |
| batch size | 1024 |
| sampler | Uniform |
| remover | FIFO |
| rate limiter | SampleToInsertRatio |
| 每条样本最大复用比 | 8 |

## 10. 监控指标

`monitor_builder.py` 将监控分为多个面板：

- 总训练胜率、评估胜率；
- 四种英雄对局的独立胜率和敌塔掉血；
- 各历史对手模型的胜率；
- K/D、英雄伤害、双方 HP 和血量优势；
- 1/2/3 技能使用率、命中率、冷却和合法率；
- 鲁班与狄仁杰分别统计的技能命令、执行和命中；
- 技能先验 try/apply 比例；
- 被冷却、mask、塔险和低血量撤退阻断的比例；
- 注意力分布和目标选择变化；
- 每个奖励子项的均值。

判断策略是否真正变好时，不应只看总 reward。建议至少联合观察：

```text
eval_win_rate
enemy_tower_hp_drop
kill / death
reward_tower / reward_win
hero_target_rate / tower_target_rate
skill_use_rate / skill_hit_hero_rate
prior_apply_rate / prior_block_rate
```

例如总 reward 上升但 `enemy_tower_hp_drop` 不变，可能只是模型学会刷局部塑形奖励；技能使用率上升但命中率下降，可能是探索概率或错误施法惩罚失衡。

## 11. 运行与配置

本项目依赖腾讯开悟客户端或比赛平台提供的运行时、环境模块和 `kaiwudrl` 包，不能作为普通独立 Python 项目直接启动。

在对应平台工作区执行冒烟测试：

```bash
python train_test.py
```

`train_test.py` 当前使用：

```python
algorithm_name = "ppo"
```

冒烟测试会临时使用较小的 replay buffer 和 batch size，并在约 1000 帧后提前结束。正式训练应从平台训练任务入口启动。

常用修改位置：

| 目标 | 文件 |
| --- | --- |
| 调整 PPO、网络、奖励权重和开关 | `agent_ppo/conf/conf.py` |
| 调整对手池、评估频率和默认阵容 | `agent_ppo/conf/train_env_conf.toml` |
| 调整奖励判定 | `agent_ppo/feature/reward_process.py` |
| 调整英雄技能探索 | `agent_ppo/agent.py` |
| 调整特征 | `agent_ppo/feature/feature_process/__init__.py` |
| 调整训练对局循环 | `agent_ppo/workflow/train_workflow.py` |
| 调整监控面板 | `agent_ppo/conf/monitor_builder.py` |
| 调整历史模型白名单 | `kaiwu.json` |

## 12. 修改代码时的同步约束

- 修改特征数量时，需要同步 `GameConfig`、`DimConfig`、模型切分和样本 shape；
- 修改动作头维度时，需要同步 legal action 切分、target mask 和 SampleData；
- 修改英雄 ID 时，需要同步技能 ID 映射、英雄特征编码和英雄专属奖励；
- 修改对手池时，模型 ID 必须在当前平台账号或比赛环境中可加载；
- 调大奖励前，应先检查原始 reward item 的取值范围，避免权重叠加后某一项支配全部学习；
- 修改技能探索时，应同时观察 try、apply、ready、legal、context 和 blocked 指标；
- 训练阵容列表中的重复项会直接改变采样权重。

## 13. 当前实现边界

- 没有在仓库中提供可离线运行的游戏环境；
- 没有附带训练好的 checkpoint；
- `agent_diy` 只是最小模板，不具备 `agent_ppo` 的完整能力；
- 目前固定使用狂暴，没有根据对局动态选择召唤师技能；
- 对手池是顺序轮换，不是基于 Elo 或胜率的自适应课程；
- 技能先验使用固定概率和阈值，仍需通过消融实验判断最佳参数；
- README 不声明尚未通过评估结果证明的胜率或泛化性能。
