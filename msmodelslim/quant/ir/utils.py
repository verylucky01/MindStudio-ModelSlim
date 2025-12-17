#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
from msmodelslim.utils.exception import SchemaValidateError


def reshape_to_blocks(a, axes, block_size):
    if axes is None:
        raise SchemaValidateError(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size <= 0:
        raise SchemaValidateError("block_size <= 0 in _reshape_to_blocks")
    # Fix axes to be positive and sort them
    axes = [(x + len(a.shape) if x < 0 else x) for x in axes]
    if not all(x >= 0 for x in axes):
        raise SchemaValidateError("All elements must be greater than or equal to 0")
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i, axis in enumerate(axes):
        new_axis = axis + i  # Shift axes due to added dimensions
        a = torch.unsqueeze(a, dim=new_axis + 1)

    # Pad to block_size
    orig_shape = a.size()
    pad = [0, 0] * len(orig_shape)

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        a = torch.nn.functional.pad(a, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                if not (shape[axis] % reshape_block_size == 0):
                    raise SchemaValidateError(
                        f"Dim {axis} size ({shape[axis]} not divisible by block size ({reshape_block_size})."
                    )
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = a.size()
    reshape = _reshape(list(padded_shape), block_size)

    a = a.view(reshape)
    return a, axes, orig_shape, padded_shape


def undo_reshape_to_blocks(a, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    a = a.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        a = a[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        a = torch.squeeze(a, dim=axis + 1)
    return a
