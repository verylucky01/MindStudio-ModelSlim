# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import logging

from torch.nn.modules import Module

from ascend_utils.pytorch.knowledge_distill.distill_losses_manager_torch import DistillLossesManager


class DistillDualModelsTorch(Module):
    def __init__(self, config, student_model, teacher_model):
        super(DistillDualModelsTorch, self).__init__()

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.hard_label_loss_weight = \
            config.hard_label_loss_weight if hasattr(config, 'hard_label_loss_weight') else None
        self.distill_loss = None

        self.distill_losses_manager = DistillLossesManager(config, self.student_model, self.teacher_model)

        logging.info("DistillDualModels inited.")

    def forward(self, *data):
        """
        Calculate distillation loss
        data: tuple type
        """
        t_outputs = self.distill_losses_manager.forward_teacher(self.teacher_model, data)
        s_outputs = self.distill_losses_manager.forward_student(self.student_model, data)
        t_outputs_tmp = self.set_indexable(t_outputs)
        s_outputs_tmp = self.set_indexable(s_outputs)

        self.distill_loss, _ = self.distill_losses_manager.compute_loss_pt(s_outputs_tmp, t_outputs_tmp)
        return self.distill_loss, s_outputs, t_outputs

    def set_indexable(self, data):
        if isinstance(data, (list, tuple)):
            return data
        else:
            return [data]

    def get_total_loss(self, hard_label_loss):
        if self.hard_label_loss_weight is None:
            total_loss = self.distill_loss
        else:
            total_loss = self.distill_loss * (
                        1 - self.hard_label_loss_weight) + hard_label_loss * self.hard_label_loss_weight
        return total_loss

    def set_train_state(self, is_teacher_train=False):
        if not is_teacher_train:
            self.teacher_model.eval()
            if not self.is_model_in_cpu(self.teacher_model):
                self.teacher_model.half()
        else:
            self.teacher_model.train()
        self.student_model.train()

    def get_student_model(self):
        return self.student_model

    def is_model_in_cpu(self, model):
        state = model.state_dict()
        for value in state.values():
            if value.device.type == "cpu":
                return True
        return False
