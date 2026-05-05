# Packing 实现分析（基于 wjq 脚手架）

## 文档目的

这份文档用于梳理 `LlamaFactory-wjq` 中 packing 相关脚手架的实现思路，重点参考：

- `src/llamafactory/v1/core/utils/batching.py`
- `src/llamafactory/v1/config/arg_parser.py`
- `src/llamafactory/v1/core/base_trainer.py`
- `src/llamafactory/v1/utils/objects.py`

这里把 `normal` 视为基线，再对比 `padding_free`、`dynamic normal`、`dynamic padding_free` 四种模式。

## 总体分层

`wjq` 的 batching 设计可以概括为四层：

1. `StatefulDistributedSampler`
   负责按 DP rank 切分数据，并支持状态恢复。
2. `StatefulDataLoader`
   负责把原始 sample 渲染成 `ModelInput`，并提供可恢复的数据流。
3. `StatefulBuffer`
   负责在“取数”和“组 batch”之间做缓冲，既存样本数，也存 token 总量。
4. `BatchGenerator`
   负责根据 batching strategy 从 buffer 中产出一步训练要消费的 `list[micro_batch]`。

这个分层有两个明显优点：

- 采样、tokenize、batch 组织彼此解耦，后续加新策略只需要改 `BatchGenerator` 和相关 helper。
- dataloader 与 buffer 都可持久化，后续做断点续训和 stateful 恢复比较自然。

## 四种模式总览

| 模式 | dataloader 取数粒度 | 单步 batch 语义 | epoch 语义 | `wjq` 当前状态 |
| --- | --- | --- | --- | --- |
| `normal` | 固定 `micro_batch_size * num_micro_batch` | 固定样本数，pad/truncate 后训练 | 保持传统定义 | 已实现 |
| `padding_free` | 固定 `micro_batch_size * num_micro_batch` | 每个 micro batch 固定样本数，直接拼接成一个无 padding 序列 | 基本保持传统定义 | 已实现 |
| `dynamic_batching` | 单条样本 | 每个 micro batch 按 token budget 控制，样本数可变 | 明显弱化，更适合 `max_steps` | 已实现 |
| `dynamic_padding_free` | 单条样本 | 每个 micro batch 按 token budget 做 packing，样本数可变 | 明显弱化，更适合 `max_steps` | 预留未实现 |

下面分别展开。

## 1. normal：固定 batch size 的传统语义

`normal` 是最标准的训练模式：

- `StatefulDataLoader` 直接以 `micro_batch_size * num_micro_batch` 为 `batch_size` 取数。
- `BatchGenerator._fill_buffer()` 只做一件事：保证 buffer 里至少有一整步需要的样本数。
- `default_collate_fn()` 再把这一整步切成若干个 micro batch，每个 micro batch 单独 `pad_and_truncate()`。

这个模式的关键特征是：

- batch size 的定义是传统定义，也就是“每一步消费多少条样本”。
- `len(dataloader)` 与 epoch 长度有直观对应关系。
- `num_train_epochs * len(train_batch_generator)` 仍然有很强的语义稳定性。

如果只看训练器侧，`normal` 模式与传统 HF Trainer 的理解几乎一致。

## 2. padding_free：固定样本数 pack，但不做动态 token budget

`wjq` 对 `padding_free` 的处理不是“动态装箱”，而是“固定样本数拼接”。

### 2.1 取数语义

- 上游 dataloader 仍然按固定样本数取数。
- 一个训练 step 仍然由固定数量的样本构成。
- 不同的是，进入每个 micro batch 后，不再对样本做 padding，而是调用 `_pack_padding_free_samples()` 直接拼接。

### 2.2 拼接逻辑

`_pack_padding_free_samples()` 做了几件核心事情：

1. 对每个 sample 按 `cutoff_len` 截断。
2. 忽略原始 sample 里的 `attention_mask` 和 `position_ids`，统一重新生成。
3. 将多条样本的 `input_ids`、`labels`、`loss_weights` 直接串接。
4. 每条后续样本的第一个 token：
   - `labels[0] = IGNORE_INDEX`
   - `loss_weights[0] = 0.0`
5. 每条样本的 `position_ids` 从 0 重新开始。
6. 整个 pack 的 `attention_mask` 置为全 1。

### 2.3 这意味着什么

这一实现已经抓住了 `padding_free` 的三个核心点：

- 没有 padding token，浪费更少。
- 通过重置 `position_ids` 给 FlashAttention varlen 路径提供样本边界信号。
- 通过 mask 掉后续样本首 token，避免跨样本 label 泄漏。

同时它也刻意没有做两件事：

- 不在一个 step 内重新决定“每个 micro batch 该放多少条样本”。
- 不做大 buffer 上的 greedy / best-fit。

所以它本质上仍然是“固定样本数训练”，只是把 micro batch 内部从 padded 变成了 packed。

### 2.4 与 `normal` 的关系

相对 `normal`，`padding_free` 改的是“张量组织方式”，不是“step 语义”：

- batch size 还是传统定义。
- epoch 长度还是传统定义。
- scheduler 仍可基于 `len(dataloader)` 理解。

这也是为什么 `wjq` 里 `padding_free` 的 `_length` 直接等于 `len(self._data_provider)`。

## 3. dynamic normal：按 token budget 控制 batch，而不是按样本数控制

`dynamic_batching` 是 `wjq` 当前已实现的动态模式。

### 3.1 预算定义

其 budget 定义为：

`micro_batch_size * cutoff_len`

这里的 `micro_batch_size` 不再严格表示“每个 micro batch 固定有多少条样本”，而更像是：

- 一个名义上的参考样本数；
- 用来定义 token budget 的尺度参数。

### 3.2 组 batch 方式

`_get_dynamic_micro_batch_sizes()` 会从当前 buffer 头部顺序扫描样本，并根据：

- 当前已选样本数
- 当前最大样本长度
- `max_len * sample_count <= budget`

来决定一个 micro batch 最终能装多少条样本。

随后：

- `BatchGenerator._fill_buffer()` 会持续往 buffer 塞单条样本；
- 直到能够凑出 `num_micro_batch` 个满足 budget 的 micro batch；
- `dynamic_batching_collate_fn()` 再把这些可变大小的 micro batch 逐个 `pad_and_truncate()`。

### 3.3 为什么 dataloader 改成 `batch_size=1`

动态模式下，`wjq` 把 dataloader 的取数粒度改成单条样本，这一点很关键。

因为只有单条取数，batching 层才能自己决定：

- 何时停下来形成一个 micro batch；
- 何时继续读样本扩张 buffer；
- 一个 step 最终由多少条样本组成。

如果上游仍按固定 batch size 取数，就会把动态 batching 的决策权锁死在 dataloader 这一层。

### 3.4 epoch 语义为什么会弱化

一旦 step 的核心约束从“样本数”变成“token budget”，就会出现：

- 不同步之间的样本数不同；
- 不同 rank 之间，每一步实际消费的样本数也可能不同；
- `len(dataloader)` 只能是近似值，而不是严格语义。

`wjq` 已经开始承认这一点：

- `dynamic_batching` 的 `_length` 只是 `ceil(len(data_provider) / (micro_batch_size * num_micro_batch))`，本质上是近似长度。
- `arg_parser.py` 强制动态模式必须配置 `max_steps`。
- `BaseTrainer` 也优先以 `max_steps` 作为训练总步数。

所以 `dynamic normal` 的正确理解应该是：

- 更适合 fixed steps 训练；
- epoch 只剩“外层 shuffle / 日志归档”的弱语义。

## 4. dynamic padding_free：脚手架预留了位置，但真正实现还没落地

`wjq` 目前在 `BatchGenerator.__init__()` 中直接对 `dynamic_padding_free` 抛了 `NotImplementedError`。但从现有结构看，最终落点已经很清晰。

### 4.1 它应该继承哪些已有设计

从 `dynamic_batching` 继承：

- dataloader 以单条样本取数；
- 本地 buffer 驱动 batch 生成；
- 训练以 `max_steps` 为主；
- 数据流支持 iterator 重新拉起和状态恢复。

从 `padding_free` 继承：

- pack 后无 padding；
- 每条样本重置 `position_ids`；
- 样本边界需要处理 label / loss_weight mask；
- 需要依赖 FlashAttention varlen 路径，因此 `arg_parser` 已强制 `flash_attn == fa2`。

### 4.2 最终模式的正确语义

结合你的目标描述，`dynamic_padding_free` 不应该做全量排序，而应该是：

1. 先用 `StatefulDistributedSampler` 把原始数据切到各个 DP rank。
2. 每个 rank 只维护自己的本地 processed-sample buffer。
3. buffer 大小受控，例如最多读入 `buffer_size` 条已处理样本。
4. 在这个局部 buffer 内做 greedy / best-fit，生成满足 token budget 的 pack。
5. 每一步继续产出 `num_micro_batch` 个 packed micro batch。
6. 如果某个 rank 样本不够，需要：
   - 重新打开本 rank shard 的 iterator 补数，或
   - 丢弃尾部残样本，
   但无论哪种方式，都必须保证所有 rank 最终执行相同步数。

### 4.3 为什么它必须依赖 fixed `max_steps`

因为 dynamic + local greedy 之后：

- 每个 rank 的 pack 结果不同；
- 每个 rank 的“一个 epoch 能产出多少 step”不再严格相同；
- 如果按“谁先耗尽谁停”处理，某些 rank 会提前退出，而其他 rank 还在做反向和 all-reduce，最终就会死锁。

所以这个模式下，训练主语义必须从 epoch 切到 fixed steps。

## 5. `wjq` 脚手架里已经为最终实现准备好的关键条件

我认为 `wjq` 这套脚手架最重要的价值，不在于它已经把所有 packing 都写完了，而在于它已经把“正确的承载形态”搭出来了：

### 5.1 状态可恢复

`BatchGenerator.state_dict()` / `load_state_dict()` 已经把：

- dataloader 状态
- buffer 状态

纳入恢复路径。`checkpoint.py` 还会按 rank 单独保存 dataloader 状态。这对动态 packing 尤其重要。

### 5.2 训练器已兼容 `max_steps` 优先

`BaseTrainer` 已经把：

- `max_steps` 优先于 `num_train_epochs`
- 到达步数直接停止

作为一等语义。动态模式可以直接复用。

### 5.3 参数约束已经开始收口

`arg_parser.py` 已经增加了两个非常关键的限制：

1. `padding_free` / `dynamic_padding_free` 必须用 `flash_attn: fa2`
2. `dynamic_*` 必须显式设置 `max_steps`

这说明脚手架已经不是纯想法，而是在主动把最终实现需要的使用边界收紧。

## 结论

如果用一句话总结 `wjq` 的实现路线：

- `normal` 负责保持传统训练语义；
- `padding_free` 先解决“固定样本数下的无 padding pack”；
- `dynamic normal` 再解决“按 token budget 组 batch”；
- `dynamic padding_free` 最后把“局部 buffer 上的 greedy packing”接到动态语义上，并彻底转向 fixed-step 训练。

这条路线是合理的，因为它把问题拆成了：

1. 先解决张量组织问题；
2. 再解决 token budget 问题；
3. 最后解决多卡下的局部 greedy 与 step 对齐问题。

从工程上看，这比一开始就直接做完整的 dynamic padding-free 更稳。
