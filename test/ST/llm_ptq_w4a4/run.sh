#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

rm -rf $PROJECT_PATH/output/llm_ptq_w4a4
python run.py

if [ $? -eq 0 ]
then
    echo llm_ptq_w4a4: Success
else
    echo llm_ptq_w4a4: Failed
    run_ok=$ret_failed
fi

rm -rf $PROJECT_PATH/output/llm_ptq_w4a4

exit $run_ok