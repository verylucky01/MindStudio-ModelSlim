#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

rm -rf $PROJECT_PATH/output/ptq-tools/*
ASCEND_RT_VISIBLE_DEVICES=0
python run.py

if [ $? -eq 0 ]
then
    echo quant_sd3: Success
else
    echo quant_sd3: Failed
    run_ok=$ret_failed
fi

rm -rf $PROJECT_PATH/output/ptq-tools/*

exit $run_ok