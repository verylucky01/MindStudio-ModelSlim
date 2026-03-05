#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Unit tests for SubgraphFusionFactory and NonFusionSubgraphFusion (commit f2075e1).
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import NonFusionSubgraph
from msmodelslim.processor.anti_outlier.common.subgraph_fusion import (
    SubgraphFusionFactory,
    NonFusionSubgraphFusion,
)


class TestNonFusionSubgraphFusion(unittest.TestCase):
    """Tests for NonFusionSubgraphFusion.apply_fusion."""

    def test_apply_fusion_scales_linears(self):
        """apply_fusion scales each linear weight by scales (view 1, -1) per input channel."""
        linear1 = nn.Linear(4, 3)
        linear2 = nn.Linear(4, 3)
        torch.nn.init.ones_(linear1.weight)
        torch.nn.init.ones_(linear2.weight)
        subgraph = NonFusionSubgraph(linears=[linear1, linear2])
        # scales (1, in_features) -> view(1, -1), weight (out_features, in_features) * (1, in_features)
        scales = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        strategy = NonFusionSubgraphFusion()
        strategy.apply_fusion(subgraph, {"scales": scales}, None)
        torch.testing.assert_close(linear1.weight, torch.ones_like(linear1.weight) * 2.0)
        torch.testing.assert_close(linear2.weight, torch.ones_like(linear2.weight) * 2.0)

    def test_apply_fusion_no_shift(self):
        """apply_fusion ignores shifts (not supported for NonFusion)."""
        linear = nn.Linear(2, 2)
        torch.nn.init.ones_(linear.weight)
        subgraph = NonFusionSubgraph(linears=[linear])
        scales = torch.tensor([[1.0, 1.0]])
        strategy = NonFusionSubgraphFusion()
        strategy.apply_fusion(subgraph, {"scales": scales}, {"norm_shift": torch.zeros(2)})
        # No shift applied to linear; weight only scaled
        torch.testing.assert_close(linear.weight, torch.ones_like(linear.weight))


class TestSubgraphFusionFactoryNonFusion(unittest.TestCase):
    """Tests for SubgraphFusionFactory.get_fuser with NonFusionSubgraph."""

    def test_get_fuser_returns_non_fusion_fuser(self):
        """get_fuser(NonFusionSubgraph) returns NonFusionSubgraphFusion."""
        linear = nn.Linear(2, 2)
        subgraph = NonFusionSubgraph(linears=[linear])
        fuser = SubgraphFusionFactory.get_fuser(subgraph)
        self.assertIsInstance(fuser, NonFusionSubgraphFusion)

    def test_apply_fusion_to_subgraph_non_fusion(self):
        """apply_fusion_to_subgraph with NonFusionSubgraph runs without error."""
        linear = nn.Linear(2, 2)
        torch.nn.init.ones_(linear.weight)
        subgraph = NonFusionSubgraph(linears=[linear])
        scales = torch.tensor([[2.0, 2.0]])
        SubgraphFusionFactory.apply_fusion_to_subgraph(
            subgraph, {"scales": scales}, None
        )
        torch.testing.assert_close(linear.weight, torch.ones_like(linear.weight) * 2.0)


if __name__ == "__main__":
    unittest.main()
