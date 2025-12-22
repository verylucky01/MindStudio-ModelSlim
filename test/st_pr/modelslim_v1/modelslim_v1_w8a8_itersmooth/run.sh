#!/usr/bin/env bash

# 无论在哪个目录执行脚本，SCRIPT_DIR 始终指向当前 .sh 文件所在的绝对目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output_modelslim_v1_w8a8_itersmooth"
# 直接拼接「脚本目录 + 配置文件名」，确保无论在哪执行都能找到配置文件
CONFIG_FILE="${SCRIPT_DIR}/dense-w8a8-itersmooth-v1.yaml"

msmodelslim quant \
--model_path ${MODEL_RESOURCE_PATH}/Qwen3-14B \
--save_path ${OUTPUT_DIR} \
--device npu \
--model_type Qwen3-14B \
--config_path ${CONFIG_FILE} \
--trust_remote_code True

# 若需清理环境 此处不能直接做操作，不然会误以为执行成功
