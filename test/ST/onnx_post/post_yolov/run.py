import os
from modelslim.onnx.post_training_quant import QuantConfig, run_quantize
from modelslim import set_logger_level 
set_logger_level("info")


quant_config = QuantConfig(is_dynamic_shape=True, input_shape=[[1,3,640,640]])
input_model_path = f"{os.environ['PROJECT_PATH']}/resource/onnx_post/yolov5m.onnx"
output_model_path = f"{os.environ['PROJECT_PATH']}/output/onnx_post/yolov5m_quant.onnx"
run_quantize(input_model_path,output_model_path,quant_config)