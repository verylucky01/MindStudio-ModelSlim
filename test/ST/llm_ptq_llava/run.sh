#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

rm -rf $PROJECT_PATH/output/llm_ptq_llava
python run.py

if [ $? -eq 0 ]
then
    echo llm_ptq_llava: Success
else
    echo llm_ptq_llava: Failed
    run_ok=$ret_failed
fi

rm -rf $PROJECT_PATH/output/llm_ptq_llava

exit $run_ok