#!/usr/bin/env bash

ulimit -n 32768
rm -rf $PROJECT_PATH/output/weight_compression/*
echo "yes" | python run.py #功能运行脚本