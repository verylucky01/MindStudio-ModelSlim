# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
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
    license='Apache-2.0',
    keywords='msmodelslim',
    python_requires='>=3.7',
    install_requires=required,
    entry_points={
        'console_scripts': [
            'msmodelslim=msmodelslim.cli.__main__:main'
        ],
        "msmodelslim.quant_service.plugins": [
            "modelslim_v0=msmodelslim.app.quant_service.modelslim_v0.quant_service:ModelslimV0QuantService",
            "modelslim_v1=msmodelslim.app.quant_service.modelslim_v1:ModelslimV1QuantService",
            "multimodal_sd_modelslim_v1="
            "msmodelslim.app.quant_service.multimodal_sd_v1:MultimodalSDModelslimV1QuantService",
            "multimodal_vlm_modelslim_v1="
            "msmodelslim.app.quant_service.multimodal_vlm_v1:MultimodalVLMModelslimV1QuantService",
        ],
        "msmodelslim.model_adapter.plugins": model_adapter_plugins,
    },
)
