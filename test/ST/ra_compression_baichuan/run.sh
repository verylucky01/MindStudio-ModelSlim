#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

source /home/ptq-test/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate llm_ptq_ra

rm -rf $PROJECT_PATH/output/ra_compression_baichuan/*

python run.py
if [ $? -eq 0 ]
then
    echo ra_compression_baichuan: Success
else
    echo ra_compression_baichuan: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

exit $run_ok