# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import logging

from mindspore.common.parameter import Parameter
from ascend_utils.common.prune.transformer_prune.prune_utils_base import PruneUtilsBase


class PruneUtilsMs(PruneUtilsBase):
    def prune_bert_intra_block_ms(self, model, state_dict, model_config):
        logging.info('Attention, prune_bert_intra_block is used for "separate" qkv weight')
        model_state_dict = model.parameters_dict()
        self.prune_bert_intra_block(model_state_dict, state_dict, True, parameter=Parameter)
        return state_dict


prune_utils_ms = PruneUtilsMs()
PRUNE_STATE_DICT_FUNCS_MS = {
                             'prune_blocks': prune_utils_ms.prune_blocks,
                             'prune_bert_intra_block': prune_utils_ms.prune_bert_intra_block_ms
}
