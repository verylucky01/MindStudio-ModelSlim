#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

# 用例名（用于打屏和指定输出路径）
CASE_NAME=llm_ptq_w4a16_pergroup_hqq

# 无论在哪个目录执行脚本，SCRIPT_DIR 始终指向当前 .sh 文件所在的绝对目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

pip install -r ${SCRIPT_DIR}/requirements.txt

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

python run.py

if [ $? -eq 0 ]; then
  echo ${CASE_NAME}: Success
else
  echo ${CASE_NAME}: Failed
  run_ok=$ret_failed
fi

exit $run_ok
