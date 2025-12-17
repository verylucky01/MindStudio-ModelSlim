#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

source /home/ptq-test/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate modelslim_py310

rm -rf $PROJECT_PATH/output/llm_ptq_multi_model
python run.py
if [ $? -eq 0 ]
then
    echo llm_ptq_multi_model: Success
else
    echo llm_ptq_multi_model: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

exit $run_ok