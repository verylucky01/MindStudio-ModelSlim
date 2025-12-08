#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
# ==============================================================================
set -e
script=$(readlink -f "$0")
route=$(dirname "$script")
rootdir=$(dirname "$route")
kia_dir=$(dirname "$rootdir")/automl_kia

if [ -e $kia_dir ]; then
    mkdir ${rootdir}/msmodelslim/pytorch/quant/ptq_tools/ptq_kia
    mkdir ${rootdir}/msmodelslim/onnx/squant_ptq/onnx_ptq_kia

    cp ${kia_dir}/quant_funcs.* ${rootdir}/msmodelslim/pytorch/quant/ptq_tools/ptq_kia -rf
    cp ${kia_dir}/weight_transform.* ${rootdir}/msmodelslim/pytorch/quant/ptq_tools/ptq_kia -rf

    cp ${kia_dir}/quant_funcs_onnx.* ${rootdir}/msmodelslim/onnx/squant_ptq/onnx_ptq_kia -rf
    cp ${kia_dir}/weight_transform_onnx.* ${rootdir}/msmodelslim/onnx/squant_ptq/onnx_ptq_kia -rf
fi

export PYTHONPATH="${rootdir}":$PYTHONPATH
export DEVICE_ID=0
echo "PYTHONPATH is ${PYTHONPATH}"

rm -rf ${route}/.coverage ${route}/report
mkdir -p ${route}/report
chmod o= ${route}/resources -R  # Others no permission
chmod g-w ${route}/resources -R  # Group not writable

ret=0
code_dir=${rootdir}/msmodelslim,${rootdir}/ascend_utils
cp ${rootdir}/lab_calib     ${rootdir}/msmodelslim/ -rf
cp ${rootdir}/lab_practice  ${rootdir}/msmodelslim/ -rf
cp ${rootdir}/config        ${rootdir}/msmodelslim/ -rf
# Final output results need a `final.xml`, but merging multi xml requires tools like  `junitparser`. Don't know how...
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/pytorch --junitxml="${route}/report/final.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/mindspore --junitxml="${route}/report/final_mindspore.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/common --junitxml="${route}/report/final_common.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/onnx --junitxml="${route}/report/final_onnx.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/msmodelslim --junitxml="${route}/report/final_msmodelslim.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/quant --junitxml="${route}/report/final_quant.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/utils --junitxml="${route}/report/final_utils.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/analyze --junitxml="${route}/report/final_analyze.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/anti --junitxml="${route}/report/final_anti.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/model --junitxml="${route}/report/final_model.xml" || ret=1
python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/smoke --junitxml="${route}/report/final_smoke.xml" || ret=1

python3 -m coverage combine
python3 -m coverage xml -o ${route}/report/coverage.xml
cat ${route}/report/coverage.xml | grep line-rate | grep coverage

exit ${ret}