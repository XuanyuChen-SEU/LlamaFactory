import bisect
from typing import Iterator, List
from torch.utils.data import default_collate
import torch

from ...utils.constants import IGNORE_INDEX
from ...utils.objects import StatefulBuffer
from ...utils.plugin import BasePlugin
from ...utils.types import BatchInfo, BatchInput, DataLoader, ModelInput

def search_for_fit(numbers: List[int], capacity: int) -> int:
    """使用二分查找找到最接近但不超过容量的索引。

    Args:
        numbers: 已排序的数字列表（从小到大）。
        capacity: 容量限制。

    Returns:
        最接近但不超过容量的索引。如果没有合适的数字，返回 -1。
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(samples: List[BatchInput], cutoff_len: int) -> List[int]:
    """使用贪心算法选择样本进行打包，返回索引列表。

    Args:
        samples: 待选择的样本列表，每个样本是 ModelInput 字典。
        cutoff_len: 打包批次批次的最大序列长度。

    Returns:
        能够一起打包的选中样本的索引列表。
    """
    lengths = [len(s["input_ids"]) for s in samples]
    indices = list(range(len(samples)))

    selected_indices = []
    remaining_capacity = cutoff_len

    while lengths:
        idx = search_for_fit(lengths, remaining_capacity)
        if idx == -1:
            break

        selected_len = lengths.pop(idx)
        selected_idx = indices.pop(idx)

        selected_indices.append(selected_idx)
        remaining_capacity -= selected_len

    return selected_indices


class BatchingPlugin(BasePlugin):
    def compute_length(self, data_provider: DataLoader) -> int:
        """Compute the length of the batch generator.

        The approximate length is used to calculate the lr schedule.
        """
        raise NotImplementedError()

    def fill_buffer(self, buffer: StatefulBuffer, batch_info: BatchInfo, data_iter: Iterator[List[ModelInput]]) -> None:
        """Fill the buffer with data."""
        return self['fill_buffer'](buffer, batch_info, data_iter)

    def generate_batch(self, buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        """Generate a batch from the buffer."""
        return self['generate_batch'](buffer, batch_info)

@BatchingPlugin("padding_free").register("compute_length")
def padding_free_compute_length(data_provider: DataLoader) -> int:
    """估算打包后的批次数量。

    Args:
        data_provider: 包含数据集的数据加载器。

    Returns:
        打包后的估算批次数量。
    """
    original_length = len(data_provider.dataset)

    if original_length == -1:
        return -1

    estimated_batches_per_pack = 4
    estimated_length = original_length // estimated_batches_per_pack

    return max(1, estimated_length)


@BatchingPlugin("padding_free").register("fill_buffer")
def padding_free_fill_buffer(
    buffer: StatefulBuffer,
    batch_info: BatchInfo,
    data_iter: Iterator[list[ModelInput]]
) -> None:
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]

    BUFFER_MULTIPLIER = 10
    target_buffer_size = micro_batch_size * num_micro_batch * BUFFER_MULTIPLIER

    if len(buffer) < target_buffer_size // 2:
        while len(buffer) < target_buffer_size:
            try:
                samples = next(data_iter)
            except StopIteration:
                break

            for sample in samples:
                if len(sample["input_ids"]) > batch_info["cutoff_len"]:
                    continue

                buffer.put([sample])
        buffer._buffer.sort(key=lambda x: len(x["input_ids"]))


@BatchingPlugin("padding_free").register("generate_batch")
def padding_free_generate_batch(
    buffer: StatefulBuffer,
    batch_info: BatchInfo
) -> List[BatchInput] | None:
    if len(buffer) == 0:
        return None

    cutoff_len = batch_info["cutoff_len"]
    samples = buffer._buffer

    selected_indices = greedy_knapsack(samples, cutoff_len)

    if not selected_indices:
        return None

    selected_samples = [samples[i] for i in selected_indices]

    selected_set = set(selected_indices)
    new_buffer = [samples[i] for i in range(len(samples)) if i not in selected_set]

    buffer._buffer = new_buffer
    buffer._buffer_size = sum(len(sample["input_ids"]) for sample in new_buffer)

    packed_input_ids = []
    packed_attention_mask = []
    packed_position_ids = []
    packed_labels = []
    packed_loss_weights = []

    for sample in selected_samples:
        sample_len = len(sample["input_ids"])

        packed_input_ids.extend(sample["input_ids"])
        packed_attention_mask.extend([1] * sample_len)
        packed_position_ids.extend(list(range(sample_len)))
        packed_labels.extend(sample["labels"])
        packed_loss_weights.extend(sample["loss_weights"])

    micro_batch = {
        "input_ids": torch.tensor(packed_input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(packed_attention_mask, dtype=torch.bool).unsqueeze(0),
        "position_ids": torch.tensor(packed_position_ids, dtype=torch.long).unsqueeze(0),
        "labels": torch.tensor(packed_labels, dtype=torch.long).unsqueeze(0),
        "loss_weights": torch.tensor(packed_loss_weights, dtype=torch.float).unsqueeze(0),
    }

    return [micro_batch]
