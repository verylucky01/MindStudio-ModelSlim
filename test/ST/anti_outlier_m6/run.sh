#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

python run.py

if [ $? -eq 0 ]
then
    echo anti_outlier_m6: Success
else
    echo anti_outlier_m6: Failed
    run_ok=$ret_failed
fi


exit $run_ok