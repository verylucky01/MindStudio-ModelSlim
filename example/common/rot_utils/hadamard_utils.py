# Copyright (c) 2024 QuaRot Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from QuaRot:
# https://github.com/spcl/QuaRot.git
# The original implementation is used for Hadamard transformation in model rotation.


import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from example.common.security.path import txt_safe_load

# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py


def load_hadamard_matrix_from_txt(matrix_name, csv_dir=None):
    if csv_dir is None:
        csv_dir = os.path.dirname(__file__)
    
    csv_path = os.path.join(csv_dir, f"{matrix_name}.txt")
    
    csv_data = txt_safe_load(csv_path, check_user_stat=True)
    
    # 转换为浮点数矩阵
    matrix_data = []
    for row in csv_data:
        matrix_data.append([float(x) for x in row])
    
    # 转换为torch.FloatTensor
    return torch.FloatTensor(matrix_data)



def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def get_hadk(n, transpose=False):
    hadk, k = None, None
    if n % 172 == 0:  # llama-2-7b up
        if not is_pow2(n // 172):
            raise ValueError(f"n//172 ({n//172}) must be a power of 2")
        k = 172
        hadk = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        if not is_pow2(n // 156):
            raise ValueError(f"n//156 ({n//156}) must be a power of 2")
        k = 156
        hadk = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate 
        if not is_pow2(n // 140):
            raise ValueError(f"n//140 ({n//140}) must be a power of 2")
        k = 140
        hadk = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate 
        if not is_pow2(n // 108):
            raise ValueError(f"n//108 ({n//108}) must be a power of 2")
        k = 108
        hadk = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        if not is_pow2(n // 60):
            raise ValueError(f"n//60 ({n//60}) must be a power of 2")
        k = 60
        hadk = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        if not is_pow2(n // 52):
            raise ValueError(f"n//52 ({n//52}) must be a power of 2")
        k = 52
        hadk = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        if not is_pow2(n // 36):
            raise ValueError(f"n//36 ({n//36}) must be a power of 2")
        k = 36
        hadk = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:
        if not is_pow2(n // 28):
            raise ValueError(f"n//28 ({n//28}) must be a power of 2")
        k = 28
        hadk = get_had28().T if transpose else get_had28()
    elif n % 40 == 0:
        if not is_pow2(n // 40):
            raise ValueError(f"n//40 ({n//40}) must be a power of 2")
        k = 40
        hadk = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        if not is_pow2(n // 20):
            raise ValueError(f"n//20 ({n//20}) must be a power of 2")
        k = 20
        hadk = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        if not is_pow2(n // 12):
            raise ValueError(f"n//12 ({n//12}) must be a power of 2")
        k = 12
        hadk = get_had12().T if transpose else get_had12()
    else:
        if not is_pow2(n):
            raise ValueError(f"n ({n}) must be a power of 2")
        k = 1

    return hadk, k


def matmul_hadu(x, transpose=False):
    n = x.shape[-1]
    hadk, k = get_hadk(n, transpose)
    inp = x.clone().view(-1, n, 1)
    output = inp.clone()
    while inp.shape[1] > k:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        output = output.view(inp.shape)
        output[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        output[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        output = output.view(inp.shape[0], inp.shape[1], -1)
        (inp, output) = (output, inp)
    del output

    if k > 1:
        # Use bcast instead
        inp = hadk.view(1, k, k).to(inp) @ inp

    return inp.view(x.shape) / torch.tensor(n).sqrt()


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    q = q * 2 - 1
    q = torch.diag(q)
    return matmul_hadu(q).to(device)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    q = torch.eye(size)
    return matmul_hadu(q).to(device)


# hadamard matrices for had12, had36.pal2, had52,will, 
# # had60.pal, had108.pal, had140.pal, had156.will, had172.will:
# http://www.neilsloane.com/hadamard/index.html
def get_had12():
    return load_hadamard_matrix_from_txt("had12")


def get_had40():
    return load_hadamard_matrix_from_txt("had40")


def get_had20():
    return load_hadamard_matrix_from_txt("had20")


def get_had28():
    return load_hadamard_matrix_from_txt("had28")


def get_had36():
    return load_hadamard_matrix_from_txt("had36")


def get_had60():
    return load_hadamard_matrix_from_txt("had60")


def get_had52():
    return load_hadamard_matrix_from_txt("had52")


def get_had108():
    return load_hadamard_matrix_from_txt("had108")


def get_had140():
    return load_hadamard_matrix_from_txt("had140")


def get_had156():
    return load_hadamard_matrix_from_txt("had156")


def get_had172():
    return load_hadamard_matrix_from_txt("had172")
