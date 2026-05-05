# Packing 最终实现方案

## 文档目的

这份文档把最终目标统一整理为一套可落地的实现思路。目标不是“继续扩写一版实验代码”，而是把 `packing` 正式纳入 LlamaFactory 的训练主路径中，并保证：

- 单卡可用；
- 多卡不会因 step 数不一致而死锁；
- 断点续训可恢复；
- 四种 batching 语义清晰。

建议以 `LlamaFactory-wjq` 的 batching 主框架为骨架，再吸收 `LlamaFactory-cxy` 中局部 buffer + greedy packing 的思路。

## 1. 四种模式的最终定义

### 1.1 `normal`

定义：

- 固定 `batch size`
- 普通 `padding / truncate`
- `micro_batch_size`、`global_batch_size`、`epoch` 都保持传统语义

实现原则：

- dataloader 固定按 `micro_batch_size * num_micro_batch` 取数
- 直接走 `default_collate_fn()`
- `len(dataloader)` 和 epoch 长度等价

### 1.2 `padding_free`

定义：

- 固定 `micro_batch_size` 条样本做 pack
- 不带 dynamic buffer
- 不改变 step / epoch 语义

实现原则：

- dataloader 仍按固定样本数取数
- 每个 micro batch 内把固定样本数的样本直接拼接
- 必须处理：
  - `position_ids` 重置
  - FA / varlen attention 的样本边界
  - 样本边界首 token 的 `labels` / `loss_weights` mask

这实际上是“固定样本数 + 无 padding”的模式，不是动态打包。

### 1.3 `dynamic normal`

定义：

- batch size 不再表示固定样本数
- 每个 micro batch 按 token budget 控制
- 更适合 `max_steps` 驱动训练

实现原则：

- dataloader 按单条样本取数
- buffer 连续积累样本
- 用 `budget = micro_batch_size * cutoff_len` 控制每个 micro batch
- 允许每个 micro batch 内样本数变化
- 必须要求用户显式设置 `max_steps`

这时 epoch 只保留很弱的语义，不再适合作为严格训练单位。

### 1.4 `dynamic padding_free`

定义：

- 先用 `StatefulDistributedSampler` 把数据按 DP rank 分 shard
- 每个 rank 只维护自己的本地 processed-sample buffer
- 在本地 buffer 内做 greedy / best-fit packing
- 每个 packed micro batch 按 token budget 控制，而不是按固定样本数控制
- 训练必须以 fixed `max_steps` / `train_steps` 对齐

这是最终最关键的模式，也是最容易在多卡下出问题的模式。

## 2. 推荐的总体架构

最终实现建议继续放在 `src/llamafactory/v1/core/utils/batching.py` 主文件内，而不是再把主流程拆散到 plugin 里。原因有三点：

1. `wjq` 的主框架已经把 dataloader、buffer、state_dict、trainer 对齐了。
2. batching 是训练主路径，不适合再引入一层接口尚不稳定的 plugin 分发。
3. `dynamic_padding_free` 和 `dynamic_batching` 共用大量状态机逻辑，放在一起更清晰。

推荐分层如下：

1. 通用 helper
   - `_get_dynamic_micro_batch_sizes()`
   - `_pack_padding_free_samples()`
   - `_select_best_fit_pack_indices()`
2. `BatchGenerator`
   - 负责初始化 dataloader / sampler / buffer
   - 负责按策略填充 buffer
   - 负责产出一步训练的多个 micro batch
3. `BaseTrainer`
   - 继续复用 `max_steps` 优先逻辑
   - 动态模式下只以 fixed steps 为准，不依赖 epoch 结束来停训

## 3. 关键实现要点

## 3.1 dataloader 取数粒度

建议如下：

- `normal` / `padding_free`
  - `batch_size = micro_batch_size * num_micro_batch`
- `dynamic_batching` / `dynamic_padding_free`
  - `batch_size = 1`

原因很简单：

- 固定模式不需要下游自己决定取数边界；
- 动态模式必须把“何时形成一个 micro batch”的控制权留在 batching 层。

## 3.2 `padding_free` 的 pack helper

建议保留 `wjq` 现有 `_pack_padding_free_samples()` 的骨架，并明确以下约束：

1. 所有样本先按 `cutoff_len` 截断。
2. 每条样本的 `position_ids` 从 0 重新开始。
3. 拼接后整条序列的 `attention_mask` 为全 1。
4. 对第二条及之后的样本：
   - 首 token `labels = IGNORE_INDEX`
   - 首 token `loss_weights = 0.0`

如果当前模型 attention 路径还需要显式的 varlen 边界信息，可以进一步在 batch 中补充：

- `seq_lens`
- `cu_seqlens`

如果现有 FlashAttention 变长路径仅靠 `position_ids` 即可隔离样本边界，则这部分可以先不暴露到统一 `BatchInput`。

## 3.3 `dynamic normal` 的组 batch 逻辑

推荐直接沿用 `wjq` 当前思路：

1. buffer 内保持顺序样本流；
2. 用 `budget = micro_batch_size * cutoff_len`；
3. 顺序扫描 buffer，找出能够组成 `num_micro_batch` 个动态 micro batch 的最小前缀；
4. 每个动态 micro batch 再单独 pad。

这条路径的优点是：

- 实现简单；
- 与上游数据顺序一致；
- 恢复状态容易；
- 不引入新的局部排序成本。

## 3.4 `dynamic padding_free` 的核心算法

这是最终实现的重点，建议按下面方式做。

### 本地 buffer 语义

- buffer 中存的是已经 `renderer.process_samples()` 过的 `ModelInput`
- 每个 rank 只维护自己的 buffer
- 设置一个局部上限 `buffer_size`
- 当 buffer 低于某个阈值时，再继续从 dataloader 取单条样本补入

### 组 pack 的预算

每个 packed micro batch 的 budget 建议仍定义为：

`micro_batch_size * cutoff_len`

它不表示固定样本数，而是表示“名义上相当于多少条长度接近 `cutoff_len` 的样本”。

### 选样算法

建议复用 `cxy` 的思路，但重新整理为更严格的 helper：

1. 从 buffer 中读取每条样本的有效长度 `min(len(input_ids), cutoff_len)`。
2. 在当前 buffer 内按长度维护索引。
3. 用 greedy / best-fit 选出一组样本，使总 token 尽量逼近 budget，但不超过 budget。
4. 选中的样本交给 `_pack_padding_free_samples()` 做真正拼接。
5. 从 buffer 中删除这些样本。

这里不建议做全量数据排序。只在本地小 buffer 内贪心即可。

### 一步训练如何产生多个 micro batch

一个训练 step 仍然应该返回 `num_micro_batch` 个 micro batch。

因此 `dynamic_padding_free` 的 `generate_batch()` 需要：

1. 重复 `num_micro_batch` 次选样；
2. 每次从当前 buffer 里找一组 best-fit 样本；
3. 生成一个 packed micro batch；
4. 拼成 `list[BatchInput]` 返回给训练器。

这样训练器侧完全不需要感知差异，仍然按现有 micro-batch accumulation 逻辑工作。

## 4. 多卡训练的关键约束

## 4.1 先分 shard，再本地 greedy

最终方案必须先用 `StatefulDistributedSampler` 做 DP shard，再让每个 rank 在自己的 shard 上独立维护 buffer。

不要尝试：

- 全局共享一个大 buffer
- 全局统一排序后再广播

这两种方案都不符合 stateful dataloader 和多卡工程现实。

## 4.2 所有 rank 必须执行相同步数

动态 packing 下，真正的危险不在“每步样本不同”，而在“每个 rank 总步数不同”。

只要有某个 rank 提前退出训练循环，而其他 rank 还在：

- forward
- backward
- all-reduce

就会存在死锁风险。

因此必须明确规定：

- `dynamic_batching`
- `dynamic_padding_free`

都以 fixed `max_steps` 为主训练语义。

## 4.3 rank 数据不足时的处理策略

当某个 rank 的本地 shard 不足以继续凑出下一步时，允许两种策略：

### 策略 A：重新打开本 rank shard 的 iterator 补数

优点：

- 容易保证始终能跑满 `max_steps`
- 不会因为尾部残样本导致训练提前停掉

代价：

- 某些样本会被重复使用
- 不同 rank 的局部“伪 epoch”边界会错开

这是我更推荐的默认策略。

### 策略 B：drop last

优点：

- 不重复使用样本

代价：

- 必须非常小心地确保所有 rank 还能跑到同样的总步数
- 否则仍然可能因为某些 rank 提前退出而死锁

如果采用 drop-last，训练总步数最好直接由外部 `max_steps` 限死，且生成器不能在某个 rank 本地简单 `StopIteration`。

## 4.4 epoch 语义只保留弱约束

在动态模式下，epoch 应该只承担：

- 调 sampler seed
- 做日志展示
- 做 checkpoint 元数据记录

不要再把“一个 epoch 恰好完整覆盖所有样本一次”当成强语义。

这在 `dynamic_padding_free` 下本来就不成立。

## 5. 配置和参数约束

建议把以下限制固化到 `config/arg_parser.py`：

1. `padding_free` / `dynamic_padding_free` 必须要求 `flash_attn: fa2`
2. `padding_free` / `dynamic_padding_free` 暂不支持 sequence parallel
3. `dynamic_batching` / `dynamic_padding_free` 必须显式设置 `max_steps`

这三条里，前两条解决运行正确性，第三条解决多卡步数对齐。

如果后续希望把本地 buffer 大小暴露给用户，可以考虑新增参数，例如：

- `packing_buffer_size`
- `packing_refill_ratio`

如果暂时不想扩 CLI，也可以先在 `BatchGenerator` 内部用常量实现。

## 6. 状态恢复与断点续训

最终方案必须继续沿用 `wjq` 的 stateful 恢复思路，并确保保存以下状态：

1. dataloader / sampler 状态
2. 当前本地 buffer 中剩余的样本
3. buffer 当前 token 总量
4. 当前全局 step
5. 当前 epoch 或外层 shuffle 计数
6. 如果采用“本地 shard 反复 reopen”策略，还要能恢复 reopen 后的迭代位置

只恢复 sampler 而不恢复 buffer 是不够的，因为动态 packing 的决定发生在 buffer 层。

## 7. 推荐的落地顺序

建议按下面顺序实现，而不是一次性大改：

1. 先把 `wjq` 的 `BatchGenerator` 同步到 `cxy`
   - 包括 `dynamic_batching`
   - 包括 `arg_parser` 的限制
   - 包括 `state_dict` / `load_state_dict`
2. 让 `padding_free` 固定样本数版本先完全稳定
   - 边界 mask
   - `position_ids`
   - FA2 约束
3. 再实现 `dynamic_padding_free` 的本地 buffer + greedy 选样
4. 用 `max_steps` 跑通单卡和多卡，优先验证“不会死锁”
5. 最后再调优 packing 策略
   - greedy
   - best-fit
   - buffer size
   - refill 水位

这个顺序的好处是：

- 先把框架承载面搭稳；
- 再把最复杂的动态 packing 接进去；
- 最后才做算法层优化。

## 8. 最终结论

最终路线应该明确成一句话：

以 `wjq` 的 stateful batching 框架为主线，以 `cxy` 的局部 buffer + greedy packing 为算法补充，最终落地四种清晰分层的 batching 模式。

其中：

- `normal` 解决传统训练语义；
- `padding_free` 解决固定样本数下的 pack；
- `dynamic normal` 解决 token budget batching；
- `dynamic padding_free` 解决多卡本地 greedy packing，并强制转向 fixed-step 训练。

这也是当前最稳、最容易在 LlamaFactory 里真正跑通的实现方向。
