from bisect import bisect_right
from collections.abc import Callable
from math import ceil

import torch
from torch.utils.data import default_collate

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import pad_and_truncate
from ...utils.objects import StatefulBuffer
from ...utils.plugin import BasePlugin
from ...utils.types import BatchInfo, BatchInput, DataLoader, ModelInput


# cxy 分支保留 plugin 形式，但策略语义与 wjq 方案保持一致：
# 1. padding_free：固定样本数 pack，不引入动态 token budget。
# 2. dynamic_batching：按 token budget 形成普通 padded micro batch。
# 3. dynamic_padding_free：各 rank 只在本地有限 buffer 内做 greedy pack，
#    并通过上层 BatchGenerator 的 restart 语义继续向 fixed max_steps 推进。
DYNAMIC_PADDING_FREE_BUFFER_MULTIPLIER = 8


def _get_effective_sample_len(sample: ModelInput, cutoff_len: int) -> int:
    return min(len(sample["input_ids"]), cutoff_len)


def _get_dynamic_token_budget(batch_info: BatchInfo) -> int:
    return batch_info["micro_batch_size"] * batch_info["cutoff_len"]


def _get_dynamic_padding_free_buffer_limit(batch_info: BatchInfo) -> int:
    return max(
        batch_info["micro_batch_size"] * batch_info["num_micro_batch"] * DYNAMIC_PADDING_FREE_BUFFER_MULTIPLIER,
        batch_info["num_micro_batch"],
    )


def _get_dynamic_micro_batch_sizes(samples: list[ModelInput], batch_info: BatchInfo) -> list[int]:
    budget = _get_dynamic_token_budget(batch_info)
    cutoff_len = batch_info["cutoff_len"]
    sizes = []
    index = 0
    while index < len(samples) and len(sizes) < batch_info["num_micro_batch"]:
        max_sample_len = 0
        used = 0
        while index + used < len(samples):
            # dynamic normal 关注的是“pad 之后的矩形张量面积”，
            # 因此这里不是简单累加 token，而是用 max_len * sample_count 估算。
            sample_len = _get_effective_sample_len(samples[index + used], cutoff_len)
            if used > 0 and max(max_sample_len, sample_len) * (used + 1) > budget:
                break

            max_sample_len = max(max_sample_len, sample_len)
            used += 1
            if max_sample_len * used >= budget:
                break

        if used == 0:
            break

        sizes.append(used)
        index += used

    return sizes


def _pack_padding_free_samples(samples: list[ModelInput], cutoff_len: int) -> BatchInput | None:
    packed: dict[str, list[int] | list[float]] = {}
    position_ids: list[int] = []

    for sample_index, sample in enumerate(samples):
        sample_len = _get_effective_sample_len(sample, cutoff_len)
        if sample_len <= 0:
            continue

        for key, value in sample.items():
            if key in ("attention_mask", "position_ids") or isinstance(value, str):
                continue

            if key not in packed:
                packed[key] = []

            sliced_value = list(value[:sample_len])
            if sample_index > 0 and sliced_value:
                # pack 之后相邻样本在物理上紧挨着，不做边界 mask 会制造错误的跨样本监督。
                if key == "labels":
                    sliced_value[0] = IGNORE_INDEX
                elif key == "loss_weights":
                    sliced_value[0] = 0.0

            packed[key].extend(sliced_value)

        # 每条样本都从 0 重新开始计 position_ids，给 varlen attention 提供边界信号。
        position_ids.extend(range(sample_len))

    if not position_ids:
        return None

    packed["position_ids"] = position_ids
    packed["attention_mask"] = [1] * len(position_ids)
    return {key: torch.tensor(value).unsqueeze(0) for key, value in packed.items()}


def _select_best_fit_sample_indices(
    samples: list[ModelInput], cutoff_len: int, budget: int
) -> list[int]:
    lengths_with_indices = [
        (_get_effective_sample_len(sample, cutoff_len), index)
        for index, sample in enumerate(samples)
        if _get_effective_sample_len(sample, cutoff_len) > 0
    ]
    lengths_with_indices.sort(key=lambda item: item[0])

    lengths = [item[0] for item in lengths_with_indices]
    indices = [item[1] for item in lengths_with_indices]
    selected_indices: list[int] = []
    remaining_budget = budget

    while lengths:
        # 使用局部 best-fit greedy：每次挑“当前还能放进去的最长样本”。
        # 它不是数学最优装箱，但和 stateful dataloader / 本地小 buffer 非常兼容。
        fit_index = bisect_right(lengths, remaining_budget) - 1
        if fit_index < 0:
            break

        remaining_budget -= lengths.pop(fit_index)
        selected_indices.append(indices.pop(fit_index))

    return selected_indices


def _plan_dynamic_padding_free_micro_batches(
    samples: list[ModelInput], batch_info: BatchInfo
) -> list[list[int]]:
    working_samples = list(samples)
    working_to_original = list(range(len(samples)))
    planned_indices: list[list[int]] = []
    cutoff_len = batch_info["cutoff_len"]
    budget = _get_dynamic_token_budget(batch_info)

    for _ in range(batch_info["num_micro_batch"]):
        local_indices = _select_best_fit_sample_indices(working_samples, cutoff_len, budget)
        if not local_indices:
            break

        planned_indices.append([working_to_original[index] for index in local_indices])

        # 这里只改工作副本，不改真实 buffer。先规划完整个 step，确认能凑够
        # num_micro_batch 个 pack，再由 generate_batch 一次性把真实样本弹出。
        for index in sorted(local_indices, reverse=True):
            del working_samples[index]
            del working_to_original[index]

    return planned_indices


def _default_collate(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    cutoff_len = batch_info["cutoff_len"]
    batch_size = micro_batch_size * num_micro_batch
    if len(buffer) < batch_size:
        return None

    samples = buffer.get(batch_size)
    batch = []
    for index in range(num_micro_batch):
        micro_batch = samples[index * micro_batch_size : (index + 1) * micro_batch_size]
        batch.append(default_collate(pad_and_truncate(micro_batch, cutoff_len)))

    return batch


def _padding_free_collate(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    cutoff_len = batch_info["cutoff_len"]
    batch_size = micro_batch_size * num_micro_batch
    if len(buffer) < batch_size:
        return None

    samples = buffer.get(batch_size)
    batch = []
    for index in range(num_micro_batch):
        micro_batch = samples[index * micro_batch_size : (index + 1) * micro_batch_size]
        packed_micro_batch = _pack_padding_free_samples(micro_batch, cutoff_len)
        if packed_micro_batch is None:
            return None

        batch.append(packed_micro_batch)

    return batch


def _dynamic_batching_collate(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    micro_batch_sizes = _get_dynamic_micro_batch_sizes(buffer.samples, batch_info)
    if len(micro_batch_sizes) < batch_info["num_micro_batch"]:
        return None

    batch = []
    cutoff_len = batch_info["cutoff_len"]
    for num_samples in micro_batch_sizes:
        samples = buffer.get(num_samples)
        batch.append(default_collate(pad_and_truncate(samples, cutoff_len)))

    return batch


def _dynamic_padding_free_collate(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    planned_indices = _plan_dynamic_padding_free_micro_batches(buffer.samples, batch_info)
    if len(planned_indices) < batch_info["num_micro_batch"]:
        return None

    batch = []
    for indices in planned_indices:
        packed_micro_batch = _pack_padding_free_samples(
            buffer.pop_indices(indices),
            batch_info["cutoff_len"],
        )
        if packed_micro_batch is None:
            return None

        batch.append(packed_micro_batch)

    return batch


class BatchingPlugin(BasePlugin):
    def compute_length(self, data_provider: DataLoader) -> int:
        """Compute the length of the batch generator."""
        return self["compute_length"](data_provider)

    def fill_buffer(
        self,
        buffer: StatefulBuffer,
        batch_info: BatchInfo,
        next_samples: Callable[[bool], list[ModelInput] | None],
    ) -> None:
        """Fill the buffer with data."""
        return self["fill_buffer"](buffer, batch_info, next_samples)

    def generate_batch(self, buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        """Generate a batch from the buffer."""
        return self["generate_batch"](buffer, batch_info)


@BatchingPlugin("padding_free").register("compute_length")
def padding_free_compute_length(data_provider: DataLoader) -> int:
    return len(data_provider)


@BatchingPlugin("padding_free").register("fill_buffer")
def padding_free_fill_buffer(
    buffer: StatefulBuffer,
    batch_info: BatchInfo,
    next_samples: Callable[[bool], list[ModelInput] | None],
) -> None:
    target_size = batch_info["micro_batch_size"] * batch_info["num_micro_batch"]
    while len(buffer) < target_size:
        samples = next_samples(False)
        if samples is None:
            break

        buffer.put(samples)


@BatchingPlugin("padding_free").register("generate_batch")
def padding_free_generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    return _padding_free_collate(buffer, batch_info)


@BatchingPlugin("dynamic_batching").register("compute_length")
def dynamic_batching_compute_length(data_provider: DataLoader) -> int:
    # 这里只能给一个近似长度。动态模式真正的训练总步数应该由 max_steps 控制。
    return ceil(len(data_provider) / max(getattr(data_provider, "batch_size", 1), 1))


@BatchingPlugin("dynamic_batching").register("fill_buffer")
def dynamic_batching_fill_buffer(
    buffer: StatefulBuffer,
    batch_info: BatchInfo,
    next_samples: Callable[[bool], list[ModelInput] | None],
) -> None:
    while len(_get_dynamic_micro_batch_sizes(buffer.samples, batch_info)) < batch_info["num_micro_batch"]:
        samples = next_samples(True)
        if samples is None:
            break

        buffer.put(samples)


@BatchingPlugin("dynamic_batching").register("generate_batch")
def dynamic_batching_generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    return _dynamic_batching_collate(buffer, batch_info)


@BatchingPlugin("dynamic_padding_free").register("compute_length")
def dynamic_padding_free_compute_length(data_provider: DataLoader) -> int:
    # dynamic padding-free 同样无法从 dataloader 长度精确推导步数，
    # 这里只提供给日志和调度器一个近似值。
    return ceil(len(data_provider) / max(getattr(data_provider, "batch_size", 1), 1))


@BatchingPlugin("dynamic_padding_free").register("fill_buffer")
def dynamic_padding_free_fill_buffer(
    buffer: StatefulBuffer,
    batch_info: BatchInfo,
    next_samples: Callable[[bool], list[ModelInput] | None],
) -> None:
    buffer_limit = _get_dynamic_padding_free_buffer_limit(batch_info)
    while len(_plan_dynamic_padding_free_micro_batches(buffer.samples, batch_info)) < batch_info["num_micro_batch"]:
        # dynamic padding-free 的关键是“局部小窗口 + greedy”。
        # 到达窗口上限后就停止 refill，避免把局部策略退化成全量排序。
        if len(buffer) >= buffer_limit:
            break

        samples = next_samples(True)
        if samples is None:
            break

        buffer.put(samples)


@BatchingPlugin("dynamic_padding_free").register("generate_batch")
def dynamic_padding_free_generate_batch(buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
    return _dynamic_padding_free_collate(buffer, batch_info)
