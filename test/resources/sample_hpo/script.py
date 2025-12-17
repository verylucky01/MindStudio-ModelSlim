# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import os
import stat
import json
import argparse
import yaml


def dump_objective(objective_key, objective_value):
    json_path = './{}.json'.format(objective_key)
    if os.path.exists(json_path):
        os.remove(json_path)
    with os.fdopen(os.open(json_path, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fout:
        json.dump({objective_key: objective_value}, fout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--config_file", type=str)
    opt = parser.parse_args()
    with open(opt.config_file) as f:
        config_yml = yaml.safe_load(f)
    batch_size = config_yml.get("batch_size")
    dump_objective('lr', opt.lr)
    dump_objective('batch_size', batch_size)
    dump_objective('accuracy', 0.8)
    dump_objective('latency', 10)


if __name__ == '__main__':
    main()