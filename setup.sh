#!/bin/bash

set -euxo pipefail

# install llvm
git submodule update --init --recursive

# llvm build subshell
(
    cd submodules/llvm-project
    mkdir -p build
    cd build

    # configure llvm build
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
        -DLLVM_CCACHE_BUILD=ON

    # build llvm
    cmake --build . --target check-mlir

)

# export llvm bin
export PATH=$PWD/submodules/llvm-project/build/bin:$PATH

# python
uv python install 3.11
uv venv
uv sync
