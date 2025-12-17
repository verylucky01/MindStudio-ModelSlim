#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

rm -rf $PROJECT_PATH/output/llm_ptq_w4a8_per_token
python run.py

if [ $? -eq 0]
then
    echo "llm_ptq_w4a8_per_token test: Success"
else
    echo "llm_ptq_w4a8_per_token test: Failed"
    run_ok=$ret_failed
fi

# 清理 output
rm -rf $PROJECT_PATH/output/llm_ptq_w4a8_per_token

exit $run_ok