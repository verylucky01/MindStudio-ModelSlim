import os
import torch
from modelslim.common.prune.transformer_prune.prune_model import PruneConfig
from modelslim.common.prune.transformer_prune.prune_model import prune_model_weight
from modelslim import set_logger_level


class TorchOriModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.fc2(output)
        return output


weight_file_path = f"{os.environ['PROJECT_PATH']}/resource/prune/torch_model_weights.pth"
torch_ori_model = TorchOriModel()
torch.save(torch_ori_model.state_dict(), weight_file_path)
set_logger_level("info") #根据实际情况配置
config = PruneConfig()
config.set_steps(['prune_blocks', 'prune_bert_intra_block'])
config.add_blocks_params(r'fc(\d+)', {1: 2})
prune_model_weight(TorchOriModel(), config, weight_file_path) #model根据实际情况配置待剪枝模型实例，weight_file_path根据实际情况配置原模型的权重文件