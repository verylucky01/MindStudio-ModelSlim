#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

# hunyuan-large模型存在环境依赖
pip install transformers==4.48.2

rm -rf $PROJECT_PATH/output/llm_ptq_hooks_hunyuan
python run.py

if [ $? -eq 0 ]
then
    echo llm_ptq_hooks_hunyuan: Success
else
    echo llm_ptq_hooks_hunyuan: Failed
    run_ok=$ret_failed
fi

rm -rf $PROJECT_PATH/output/llm_ptq_hooks_hunyuan

exit $run_ok