import os
import torch
from modelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig

input_model = f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/unet/unet.onnx"
disable_names = []
config = QuantConfig(disable_names=disable_names,
                     quant_mode=0,
                     amp_num=0,
                     use_onnx=False,
                     disable_first_layer=False,
                     disable_last_layer=False,
                    #  quant_param_ops=['Conv'],
                     quant_param_ops=[],
                     atc_input_shape="latent_model_input:1,4,64,64;t:1;encoder_hidden_states:1,77,1024",
                     num_input=3,
                     #num_input=True
                     )

latent_model_input = torch.ones(1,4,64,64).to(torch.float32)
t = torch.tensor([1]).to(torch.int64)
encoder_hidden_size = torch.ones(1,77,1024).to(torch.float32)
calib_data = [[latent_model_input,t,encoder_hidden_size]]

calib = OnnxCalibrator(input_model, config, calib_data=calib_data)
calib.run()
calib.export_quant_onnx(f"{os.environ['PROJECT_PATH']}/output/onnx_squant/unet_quant.onnx", use_external=True)
