#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false
rm -rf $PROJECT_PATH/output/llm_ptq_ZS_3
python run.py