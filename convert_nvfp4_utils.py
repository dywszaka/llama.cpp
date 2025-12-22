import torch
import numpy as np
import gguf
import logging
from convert_hf_to_gguf import ModelBase

logger = logging.getLogger("hf-to-gguf-nvfp4")

def build_nvfp4_weight_block_data(weight: np.ndarray, weight_scale: np.ndarray) -> np.ndarray:
    """
    构建 NVFP4 权重块数据
    格式: 每 9 字节一个块 = 1 个 weight_scale (F8, 1字节) + 16 个 weight 值 (8 个 U8, 8字节)

    Args:
        weight: uint8 数组，每个 uint8 包含 2 个 FP4 值
        weight_scale: float8 缩放因子数组

    Returns:
        合并后的 NVFP4 块数据
    """
    if weight.dtype != np.uint8:
        raise ValueError(f"Weight data must be of type uint8 for NVFP4 format, got {weight.dtype}")

    # weight_scale 应该是 float8_e4m3fn 类型，但 numpy 中表示可能不同
    # 检查维度: weight 应该是 weight_scale 的 8 倍 (因为每个 scale 对应 8 个 uint8)
    num_blocks = weight_scale.size
    expected_weight_size = num_blocks * 8

    if weight.size != expected_weight_size:
        raise ValueError(f"Weight size {weight.size} does not match expected size {expected_weight_size} for {num_blocks} blocks")

    # 重塑为块结构
    weight_blocks = weight.reshape(num_blocks, 8)  # (num_blocks, 8 uint8)
    weight_scale_flat = weight_scale.flatten()     # (num_blocks,)

    # 完全向量化操作，避免 Python 循环
    # logger.info(f"  Building NVFP4 blocks: {num_blocks} blocks...")

    # 预分配输出数组
    nvfp4_block_data = np.empty(num_blocks * 9, dtype=np.uint8)

    # 使用 NumPy 的高级索引一次性完成所有赋值
    # 创建索引数组
    block_indices = np.arange(num_blocks)

    # scale 的位置: 0, 9, 18, 27, ... (每个块的第一个字节)
    scale_positions = block_indices * 9
    nvfp4_block_data[scale_positions] = weight_scale_flat

    # weight 的位置: 每个块占 8 字节，从偏移 1 开始
    # 使用 reshape 和切片一次性赋值
    weight_view = nvfp4_block_data.reshape(num_blocks, 9)
    weight_view[:, 1:9] = weight_blocks

    # logger.info(f"  Building NVFP4 blocks: Complete")

    return nvfp4_block_data


def prepare_tensors_for_nvfp4(instance: ModelBase):
    self = instance
    max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

    all_modified_tensors = []
    # xxxx.weight(new_name) -> xxxx.weight_scale(data_torch)
    weight_scale_tensors_map = {}

    for name, data_torch in self.get_tensors():
        # we don't need these
        if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue

        old_dtype = data_torch.dtype

        # 对于 NVFP4 模型，保留特定的数据类型
        # BF16 -> F32, 保留 F32, U8, F8_E4M3
        if data_torch.dtype == torch.bfloat16:
            # 将 BF16 转换为 F32
            data_torch = data_torch.to(torch.float32)
        elif data_torch.dtype not in (torch.float16, torch.float32, torch.uint8, torch.float8_e4m3fn):
            # 其他不支持的类型转换为 float32
            logger.warning(f"Tensor {name} has unsupported dtype {data_torch.dtype}, converting to float32")
            data_torch = data_torch.to(torch.float32)

        # use the first number-like part of the tensor name as the block id
        bid = None
        for part in name.split("."):
            if part.isdecimal():
                bid = int(part)
                break

        for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
            if new_name.endswith(".weight_scale"):
                weight_scale_tensors_map[new_name] = data_torch
            else:
                all_modified_tensors.append((new_name, data_torch, old_dtype))

    for new_name, data_torch, old_dtype in all_modified_tensors:

        data = data_torch.numpy()

        # 处理 0 维张量（标量）：input_scale, weight_scale_2, k_scale, v_scale 等
        if len(data.shape) == 0:
            if new_name.endswith(("input_scale", "weight_scale_2", "k_scale", "v_scale")):
                # 这些是 tensor-wise 的标量值，转换为 shape [1]
                data = data.reshape(1)
                logger.debug(f"Reshaped 0-dim tensor {new_name} to shape [1]")
            else:
                # 其他 0 维张量也转换为 [1]
                data = data.reshape(1)

        # 处理 NVFP4 权重: 合并 weight 和 weight_scale
        if new_name.endswith(".weight"):
            weight_scale_name = new_name.rsplit(".", 1)[0] + ".weight_scale"
            if weight_scale_name in weight_scale_tensors_map:
                weight_scale_torch = weight_scale_tensors_map[weight_scale_name]
                # Float8 需要先转换为可以用 numpy 表示的类型（保留原始字节）
                # 使用 view 将 float8 数据视为 uint8 以保留原始字节
                weight_scale_data = weight_scale_torch.view(torch.uint8).numpy()
                # 构建 NVFP4 块数据: 每块 = 1 个 weight_scale (F8) + 8 个 weight (U8)
                original_shape = data.shape
                data = build_nvfp4_weight_block_data(data.flatten(), weight_scale_data.flatten())
                # 重新计算形状: 原始最后一维扩展为包含 scale 的块数据
                # 原始形状的最后一维元素数 / 8 = 块数, 每块 9 字节
                new_last_dim = (original_shape[-1] // 8) * 9
                if len(original_shape) > 1:
                    shape = (*original_shape[:-1], new_last_dim)
                else:
                    shape = (new_last_dim,)
                data = data.reshape(shape)

        dtype = data.dtype
        data_qtype = gguf.GGMLQuantizationType.NVFP4
        if dtype == np.float32:
            data_qtype = gguf.GGMLQuantizationType.F32
        elif dtype == np.float16:
            data_qtype = gguf.GGMLQuantizationType.F16
        elif dtype == np.uint8:
            # 如果是 uint8，检查是否是 NVFP4 格式（已合并的数据）
            # NVFP4 格式的数据最后一维应该是 9 的倍数
            if data.shape[-1] % 9 == 0:
                data_qtype = gguf.GGMLQuantizationType.NVFP4
            else:
                logger.warning(f"uint8 tensor {new_name} shape {data.shape} is not NVFP4 compatible")
                data_qtype = gguf.GGMLQuantizationType.I8

        shape = data.shape
        # reverse shape to make it similar to the internal ggml dimension order
        shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

        # n_dims is implicit in the shape
        logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)
