#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
"""
统一的 pytest 配置文件，包含所有 core 测试目录的通用配置。
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from testing_utils.mock import mock_kia_library, mock_security_library, mock_init_config


# ========== 基础 Mock 配置 ==========
# 这些必须在模块级别执行，确保在任何导入之前运行
mock_init_config()
mock_kia_library()
mock_security_library()


# ========== 额外的 Mock 配置 ==========
def _mock_check_dirpath_before_read(path):
    """Mock function for check_dirpath_before_read that bypasses validation"""
    return path


# Mock check_dirpath_before_read which is called by get_valid_read_path
# but not included in mock_security_library()
if 'msmodelslim.utils.security.path' not in sys.modules:
    sys.modules['msmodelslim.utils.security.path'] = MagicMock()
sys.modules['msmodelslim.utils.security.path'].check_dirpath_before_read = _mock_check_dirpath_before_read

# Mock optional third-party dependency wcmatch to avoid ModuleNotFoundError in tests
if 'wcmatch' not in sys.modules:
    sys.modules['wcmatch'] = MagicMock()


# ========== Pytest Fixtures ==========
@pytest.fixture
def mock_torch():
    """Mock torch库，确保不会误判NPU可用"""
    with patch('torch') as mock_torch:
        mock_torch.device.return_value = Mock()
        mock_torch.manual_seed.return_value = None
        # 创建 npu mock 对象，但确保 is_available() 返回 False
        mock_npu = Mock()
        mock_npu.manual_seed.return_value = None
        mock_npu.manual_seed_all.return_value = None
        mock_npu.Stream.return_value = Mock()
        mock_npu.set_compile_mode.return_value = None
        mock_npu.is_available.return_value = False  # 关键：明确返回 False，避免误判
        mock_torch.npu = mock_npu
        yield mock_torch

