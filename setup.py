#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import logging
import os
from configparser import ConfigParser

from setuptools import setup, find_packages

config = ConfigParser()
config.read('./config/config.ini')

abs_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(abs_path, "requirements.txt")) as f:
    required = f.read().splitlines()

model_adapter_plugins = []
entry_section = config["ModelAdapterEntryPoints"]

for group, models in config.items("ModelAdapter"):
    model_list = [m.strip() for m in models.split(",")]
    if group in entry_section:
        entry_point = entry_section[group]
    else:
        logging.warning(f"ModelAdapter group '{group}' has no entry point defined in ModelAdapterEntryPoints")
        continue

    for model in model_list:
        model_adapter_plugins.append(f"{model}={entry_point}")

setup(
    name='msmodelslim',
    version='8.2.0',
    description='msModelSlim, MindStudio ModelSlim Tools',
    long_description_content_type='text/markdown',
    url=config.get('URL', 'repository_url'),
    packages=find_packages(exclude=['precision_tool', 'security', ]) + ['msmodelslim.config', 'msmodelslim.lab_calib',
                                                                        'msmodelslim.lab_practice'],
    package_dir={
        'msmodelslim': 'msmodelslim',
        'msmodelslim.config': 'config',
        'msmodelslim.lab_calib': 'lab_calib',
        'msmodelslim.lab_practice': 'lab_practice',
    },
    package_data={
        '': [
            'LICENSE',
            'data.json',
            'README.md',
            '*.txt',
            '*.bat',
            '*.sh',
            '*.cpp',
            '*.h',
            '*.py',
            '*.so',
        ],
        'msmodelslim.config': ['*'],
        'msmodelslim.lab_calib': ['**'],
        'msmodelslim.lab_practice': ['**'],
    },
    data_files=[('', ['requirements.txt'])],
    license='Mulan PSL v2',
    keywords='msmodelslim',
    python_requires='>=3.7',
    install_requires=required,
    entry_points={
        'console_scripts': [
            'msmodelslim=msmodelslim.cli.__main__:main'
        ],
        "msmodelslim.quant_service.plugins": [
            "modelslim_v0=msmodelslim.core.quant_service.modelslim_v0.quant_service:ModelslimV0QuantService",
            "modelslim_v1=msmodelslim.core.quant_service.modelslim_v1:ModelslimV1QuantService",
            "multimodal_sd_modelslim_v1="
            "msmodelslim.core.quant_service.multimodal_sd_v1:MultimodalSDModelslimV1QuantService",
            "multimodal_vlm_modelslim_v1="
            "msmodelslim.core.quant_service.multimodal_vlm_v1:MultimodalVLMModelslimV1QuantService",
        ],
        "msmodelslim.model_adapter.plugins": model_adapter_plugins,
        "msmodelslim.strategy_config.plugins": [
            "standing_high=msmodelslim.core.tune_strategy.standing_high.strategy:StandingHighStrategyConfig",
        ],
        "msmodelslim.strategy.plugins": [
            "standing_high=msmodelslim.core.tune_strategy.standing_high.strategy:StandingHighStrategy",
        ],
        "msmodelslim.evaluate_config.plugins": [
            "service_oriented=msmodelslim.infra.service_oriented_evaluate_service:ServiceOrientedEvaluateServiceConfig",
        ],
        "msmodelslim.precheck_config.plugins": [
            "garbled_text=msmodelslim.infra.evaluation.precheck.garbled_text_rule:GarbledTextPrecheckConfig",
            "expected_answer=msmodelslim.infra.evaluation.precheck.expected_answer_rule:ExpectedAnswerPrecheckConfig",
        ],
        "msmodelslim.precheck_rule.plugins": [
            "garbled_text=msmodelslim.infra.evaluation.precheck.garbled_text_rule:GarbledTextRule",
            "expected_answer=msmodelslim.infra.evaluation.precheck.expected_answer_rule:ExpectedAnswerRule",
        ],
    },
)
