# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from mindspore.nn.cell import Cell

from ascend_utils.mindspore.knowledge_distill.distill_dual_models_ms import DistillDualModelsMs
from ascend_utils.common.security.mindspore import check_mindspore_cell
from ascend_utils.mindspore.knowledge_distill.distill_losses_func_ms import DISTILL_LOSS_FUNC_MS
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim import logger


def get_distill_model_ms(teacher: Cell, student: Cell, config: KnowledgeDistillConfig):
    """
    Build a model for knowledge distillation that contains teacher, student, and loss functions.
    And you can get fine-tuned student model from this model after training.

    Args:
        teacher(nn.Cell): teacher model.
        student(nn.Cell): student model.
        config(KnowledgeDistillConfig): Configuration for knowledge distillation.

    Returns:
        a model contains teacher and student.
    """
    logger.info("================ Start build distill model ===============")

    check_mindspore_cell(teacher)
    check_mindspore_cell(student)
    KnowledgeDistillConfig.check_config(config, DISTILL_LOSS_FUNC_MS, Cell)
    soft_label_shapes = config.get_soft_label_shape()
    KnowledgeDistillConfig.generate_loss_instance(config, DISTILL_LOSS_FUNC_MS, Cell)

    distill_dual_models = DistillDualModelsMs(config, student, teacher, soft_label_shapes)
    distill_dual_models.set_train_state(config.train_teacher)

    logger.info("================ Finish build distill model ===============")
    return distill_dual_models
