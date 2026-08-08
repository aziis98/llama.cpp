#!/bin/bash

# Source the oneAPI environment variables
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

echo "Testing SYCL build with llama-cli..."
./build-sycl-full/bin/llama-cli -hf stamsam/maple-preview-gguf:tq2_0 -p "how are you?" -n 20 --temp 0
