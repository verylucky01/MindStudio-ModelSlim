import unittest
import torch
from unittest.mock import MagicMock
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
    Quantizer,
    LinearQuantizer,
    Conv2dQuantizer,
    LinearNf4Quantizer,
    _layer_wise_activation_calib,
    _layer_wise_weight_only_calib,
    layer_wise_calib
)

class TestLayerWiseCalib(unittest.TestCase):
    def setUp(self):
        self.quant_model = MagicMock()
        self.all_tensors = {'conv1': [torch.randn(1, 1, 1, 1)]}
        self.device = 'cpu'

    def test_layer_wise_calib_no_module(self):
        self.quant_model.named_modules.return_value = []
        layer_wise_calib(self.quant_model, self.all_tensors, self.device)

    def test_layer_wise_calib_module_bit_le_8(self):
        module = MagicMock()
        module.quant_input.bit = 8
        self.quant_model.named_modules.return_value = [('conv1', module)]
        layer_wise_calib(self.quant_model, self.all_tensors, self.device)

    def test_layer_wise_calib_module_bit_gt_8(self):
        module = MagicMock()
        module.quant_input.bit = 9
        self.quant_model.named_modules.return_value = [('conv1', module)]
        layer_wise_calib(self.quant_model, self.all_tensors, self.device)

    def test_layer_wise_activation_calib_no_tensor(self):
        module = MagicMock()
        module.quant_input.bit = 8
        self.quant_model.named_modules.return_value = [('conv1', module)]
        with self.assertRaises(ValueError):
            _layer_wise_activation_calib(self.all_tensors, module, 'conv2', self.device)

    def test_layer_wise_activation_calib_hessian_true(self):
        module = MagicMock()
        module.quant_input.bit = 8
        module.quant_weight.w_hessian = True
        self.quant_model.named_modules.return_value = [('conv1', module)]
        _layer_wise_activation_calib(self.all_tensors, module, 'conv1', self.device)

    def test_layer_wise_activation_calib_hessian_false(self):
        module = MagicMock()
        module.quant_input.bit = 8
        module.quant_weight.w_hessian = False
        self.quant_model.named_modules.return_value = [('conv1', module)]
        _layer_wise_activation_calib(self.all_tensors, module, 'conv1', self.device)

    def test_layer_wise_weight_only_calib_no_tensor(self):
        module = MagicMock()
        module.quant_input.bit = 9
        self.quant_model.named_modules.return_value = [('conv1', module)]
        with self.assertRaises(ValueError):
            _layer_wise_weight_only_calib(self.all_tensors, module, 'conv2', self.device)

    def test_layer_wise_weight_only_calib_hessian_true(self):
        module = MagicMock()
        module.quant_input.bit = 9
        module.quant_weight.w_hessian = True
        self.quant_model.named_modules.return_value = [('conv1', module)]
        _layer_wise_weight_only_calib(self.all_tensors, module, 'conv1', self.device)

    def test_layer_wise_weight_only_calib_hessian_false(self):
        module = MagicMock()
        module.quant_input.bit = 9
        module.quant_weight.w_hessian = False
        self.quant_model.named_modules.return_value = [('conv1', module)]
        _layer_wise_weight_only_calib(self.all_tensors, module, 'conv1', self.device)

if __name__ == '__main__':
    unittest.main()