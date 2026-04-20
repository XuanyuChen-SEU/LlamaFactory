#!/bin/bash

# 加载环境
source /home/cann/900b060/cann-9.0.0/set_env.sh
source /home/cann/900b060/nnal/atb/set_env.sh

# 可选：指定 NPU 卡
# export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
# export ASCEND_RT_VISIBLE_DEVICES="8,9,10,11"
export ASCEND_RT_VISIBLE_DEVICES="0"
# ===================== 日志配置 =====================
LOG_DIR="./logs"

# 【只有文件夹不存在时才创建】
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "已创建日志文件夹: $LOG_DIR"
fi

# 带时间戳的日志文件名
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M).log"
# ====================================================

echo "========================================"
echo "训练开始"
echo "日志文件: $LOG_FILE"
echo "========================================"

# 运行训练，保存所有输出和报错
USE_V1=1 llamafactory-cli sft examples/v1/ulysses_cp.yaml 2>&1 | tee "$LOG_FILE"

echo "训练完成！日志已保存到: $LOG_FILE"
