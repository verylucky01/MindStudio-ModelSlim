#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

python run.py

if [ $? -eq 0 ]
then
    echo layer_select: Success
else
    echo layer_select: Failed
    run_ok=$ret_failed
fi


exit $run_ok