# Packing 方案复盘（基于 cxy 现有实现）

## 文档目的

这份文档用于复盘 `LlamaFactory-cxy` 里已有的 packing 思路，重点参考：

- `src/llamafactory/v1/core/utils/batching.py`
- `src/llamafactory/v1/plugins/trainer_plugins/batching.py`
- `src/llamafactory/v1/config/arg_parser.py`

结论先说：这套实现不能直接作为最终多卡方案使用，但里面有几处思路非常有价值，尤其适合并入后续的 `dynamic_padding_free` 实现。

## 1. 当前实现的基本思路

`cxy` 的实现方向和 `wjq` 最大的不同，是更早地把注意力放在“局部 buffer 内做 packing”这件事上。

整体思路是：

1. 训练主入口仍然通过 `BatchGenerator` 驱动。
2. `normal` 模式走固定 batch size 的普通逻辑。
3. 其他模式准备下沉到 `plugins/trainer_plugins/batching.py`。
4. `padding_free` 方向上引入了：
   - 小 buffer
   - 按样本长度排序
   - 贪心 knapsack / best-fit 风格选样
   - 直接拼接成 packed sequence

这说明你当时已经把问题从“简单 concat”推进到了“局部搜索式 packing”。

## 2. 这套实现的闪光点

虽然最终路径要改到 `wjq` 思路，但下面这些点值得明确保留。

### 2.1 不是做全量排序，而是做局部 buffer packing

`padding_free_fill_buffer()` 的核心想法不是全数据集排序，而是：

- 只维护一个局部 buffer；
- buffer 不够时再补样本；
- 在这个局部窗口内做长度排序和 packing。

这是非常好的工程取舍，因为它同时兼顾了：

- packing 效率；
- 实现复杂度；
- 流式 / stateful dataloader 兼容性；
- 多卡下的本地独立决策。

这和你现在给出的最终目标方向是一致的。

### 2.2 提前把“选样算法”独立出来了

`search_for_fit()` + `greedy_knapsack()` 这一层很有价值，因为它把“怎么从候选样本里拼出一个 pack”抽象成了独立问题。

这比把 packing 逻辑直接写死在 collate 里更好，后面可以很自然地替换成：

- greedy first-fit
- best-fit
- 长度分桶后再 best-fit

而不需要重写整个 batching 流程。

### 2.3 有 buffer 水位控制意识

`BUFFER_MULTIPLIER` 与“低于一半再 refill”的做法，本质上是在做 buffer 水位控制。

这有两个现实好处：

- 避免每一步都频繁访问 dataloader；
- 避免 buffer 无限膨胀，占用过多 host 内存。

这个想法很适合直接保留到最终实现。

### 2.4 packing 发生在 tokenized sample 层，而不是原始文本层

`cxy` 的 packing 目标对象已经是 `ModelInput`，也就是 tokenize 之后的样本，而不是原始文本。

这很重要，因为真正决定 pack 质量的是 token 长度，不是文本字符长度。这个抽象层次是对的。

### 2.5 已经明确关注了超过 cutoff 的样本

`padding_free_fill_buffer()` 里会跳过长度超过 `cutoff_len` 的样本。

虽然最终策略未必一定是“直接跳过”，但你已经把“过长样本如何进入 packing 流”当成显式问题处理了，这也是对的。

## 3. 为什么当前实现无法直接作为最终方案

下面这些问题解释了为什么现有实现有借鉴价值，但不能直接落地。

### 3.1 plugin 接口没有完全接上

`BatchGenerator._fill_buffer()` 调用的是：

`BatchingPlugin(self.batching_strategy).fill_buffer(self._buffer, self._batch_info)`

但 `plugins/trainer_plugins/batching.py` 里的 `fill_buffer()` 签名要求额外传入 `data_iter`。这意味着当前代码路径并没有真正接通，运行时会出接口不匹配问题。

也就是说，这套实现更接近“设计草稿 + 局部验证”，还不是完整可执行链路。

### 3.2 dataloader 取数粒度仍然是固定大 batch

`cxy` 的 `StatefulDataLoader` 对所有策略都使用：

`batch_size = micro_batch_size * num_micro_batch`

这会直接限制动态 packing 的可塑性。原因是：

- 上游一次已经拿了一大组样本；
- 下游 buffer 只能在这些大块结果上二次处理；
- 很难精细控制“缺多少就补多少”。

而 `wjq` 在 `dynamic_batching` 里改成单条取数，恰恰解决了这个问题。

### 3.3 `padding_free` 当前只返回一个 micro batch

`padding_free_generate_batch()` 最后直接 `return [micro_batch]`，它没有真正处理：

- `num_micro_batch > 1`
- 一个训练 step 内多个 packed micro batch 的生成

这会导致它和训练器期望的“每一步返回若干个 micro batch”语义不完全一致。

### 3.4 样本边界 label mask 没处理完整

当前拼接逻辑里：

- `input_ids` 直接拼；
- `labels` 直接拼；
- `loss_weights` 直接拼；

但没有像 `wjq` 那样把后续样本首 token 做边界 mask。这会带来一个直接问题：

- pack 后第 N 条样本的首 token 可能会把前一条样本最后一个 token 当成上下文来计算 loss；
- 这不是我们想要的跨样本训练信号。

所以从训练正确性角度，边界 mask 是必须补的。

### 3.5 `position_ids` 处理还不够和 FA/varlen 语义对齐

当前实现会为每个样本重新从 0 生成 `position_ids`，这一点方向是对的。

但它还没有把这件事和以下约束系统性连起来：

- 必须依赖 `flash_attn: fa2`
- sequence parallel 下暂不支持
- 训练配置层需要明确拦截非法组合

`cxy` 的 `arg_parser.py` 目前还没有这些参数检查，因此运行边界没有收紧。

### 3.6 长度估算是经验常数，不适合作为最终调度依据

`padding_free_compute_length()` 使用固定经验值：

`estimated_batches_per_pack = 4`

这只能算非常粗的占位逻辑。对于：

- 学习率调度
- 日志中的 epoch 显示
- `save_epochs` 之类依赖长度的功能

都不够稳定。

### 3.7 多卡下最核心的问题：各 rank 产出的 step 数不一致

这是当前方案不能直接用于最终实现的根本原因。

在每个 rank 本地独立做 greedy packing 时：

- 每个 rank 看到的样本长度分布不同；
- 每个 rank buffer 内可组合出的 pack 数量不同；
- 某些 rank 可能先把本地样本耗尽；
- 另一些 rank 还能继续往后跑。

一旦有 rank 提前结束而其他 rank 还在做反向和梯度同步，就会出现死锁风险。

这正是你现在想从 `cxy` 路线切到 `wjq` 路线的核心原因。

## 4. `cxy` 实现里最值得继承到最终方案的部分

如果把最终目标定为“基于 `wjq` 主框架实现 `dynamic_padding_free`”，那么 `cxy` 里最建议保留的是下面这些思路。

### 4.1 局部 buffer 而不是全量排序

最终方案应该继续坚持：

- 每个 rank 只维护本地 buffer；
- buffer 尺寸有上限；
- 只在局部窗口内做 greedy / best-fit。

这是可扩展、可恢复、可多卡运行的现实工程方案。

### 4.2 packing 选择算法可插拔

`search_for_fit()` / `greedy_knapsack()` 说明“选哪几条样本组成一个 pack”可以被独立成 helper。

后续建议保留这个抽象，便于继续优化 packing 质量。

### 4.3 buffer 水位控制

“低于阈值再 refill”的设计值得留下。最终实现里完全可以继续采用：

- `buffer_size`
- `refill_threshold`

两个参数控制局部搜索窗口。

### 4.4 以 tokenized 长度作为唯一决策依据

packing 决策应该继续建立在 `len(sample["input_ids"])` 上，而不是别的代理指标上。

这一点 `cxy` 的方向是对的，不需要推翻。

## 5. 结论

`cxy` 方案的价值不在于“它已经是一套可直接上线的 packing 实现”，而在于它提前抓住了最终难点：

- packing 不能只做简单 concat；
- 需要局部 buffer；
- 需要贪心 / best-fit；
- 需要按 token 长度决策；
- 多卡下每个 rank 会天然独立产生不同 pack。

它的问题主要出在两层：

1. 框架承载形态还不够稳，接口没有完全接通；
2. 多卡训练的 step 对齐问题没有被彻底处理。

因此最合理的路线不是继续沿着 `cxy` 当前实现硬修，而是：

- 用 `wjq` 的 batching 主框架承载最终方案；
- 把 `cxy` 的局部 buffer + greedy packing 思路迁移进去。

这样两边的优点才能真正合并起来。
