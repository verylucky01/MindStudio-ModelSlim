import torch
import torchvision
from modelslim.pytorch.prune.prune_torch import PruneTorch

model = torchvision.models.vgg16(pretrained=False)
model.eval()

desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune(0.8)

eval_func_l2 = lambda chn_weight: torch.norm(chn_weight).item() / chn_weight.nelement()
desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).set_importance_evaluation_function(eval_func_l2).prune(0.8)

desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).set_node_reserved_ratio(0.5).prune(0.8)

left_params, desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).analysis()

left_params, desc = PruneTorch(model, torch.ones([1, 3, 224, 224])).analysis()
PruneTorch(model, torch.ones([1, 3, 224, 224])).prune_by_desc(desc)
