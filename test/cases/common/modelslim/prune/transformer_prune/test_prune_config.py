# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import pytest 

from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig


class TestPruneConfig(object):
    def test_prune_config_given_valid_when_any_then_pass(self):
        prune_config = PruneConfig()
        prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block'])
        prune_config.add_blocks_params('test_name', {0: 1})
        prune_config.get("prune_blocks_params")

    def test_prune_config_given_invalid_when_any_then_error(self):
        prune_config = PruneConfig()
        with pytest.raises(ValueError):
            prune_config.set_steps(None)
        with pytest.raises(ValueError):
            prune_config.set_steps(["fake_step"])
            PruneConfig.check_steps_list(prune_config,
                                         ['prune_blocks', 'prune_bert_intra_block', 'prune_vit_intra_block'])
        with pytest.raises(TypeError):
            prune_config.add_blocks_params(1, {0: 1})
        with pytest.raises(TypeError):
            prune_config.add_blocks_params('test_name', "{0: 1}")
        with pytest.raises(TypeError):
            prune_config.add_blocks_params('test_name', {1.1, 2})
        with pytest.raises(TypeError):
            prune_config.add_blocks_params('test_name', {"1.1", 2})
        with pytest.raises(ValueError):
            prune_config.get("fake_name")
