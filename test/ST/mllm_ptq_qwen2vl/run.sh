#!/usr/bin/env/bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

source /opt/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate multimodal_testcase

pip install qwen_vl_utils
pip install transformers==4.46.0

rm -rf $PROJECT_PATH/output/mllm_ptq_qwen2vl
python run.py

if [ $? -eq 0 ]
then
    echo mllm_ptq_qwen2vl: Success
else
    echo mllm_ptq_qwen2vl: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

# 清理output
rm -rf $PROJECT_PATH/output/mllm_ptq_qwen2vl

exit $run_ok