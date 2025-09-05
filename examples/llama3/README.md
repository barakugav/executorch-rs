# llama example

The example demonstrates how to use the `executorch` crate to run the `llama` model.

To run the example, follow these steps (note that some steps should run from the Cpp `executorch` repository and some from `executorch-rs`):

- Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up Cpp ExecuTorch. For installation run
    ```bash
    cd executorch
    source install_requirements.sh
    source examples/models/llama/install_requirements.sh
    ```

- Download `consolidated.00.pth` and `params.json` from [Llama website](https://www.llama.com/llama-downloads/) or [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B).

- Download the vocablurary:
    ```bash
    cd executorch-rs/examples/llama3
    python download_vocab.py > vocab.json
    ```

- Export the model and generate `.pte` file.
    ```bash
    cd executorch

    # Set these paths to point to the downloaded files
    LLAMA_CHECKPOINT=path/to/checkpoint.pth
    LLAMA_PARAMS=path/to/params.json

    python -m examples.models.llama.export_llama \
        --model "llama3_2" \
        --checkpoint "${LLAMA_CHECKPOINT:?}" \
        --params "${LLAMA_PARAMS:?}" \
        -X \
        -d bf16 \
        --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
        --output_name="llama3_2.pte"
    ```

- Build executorch with optimized CPU performance as follows.
    ```bash
    cd executorch

    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -Bcmake-out .

    cmake --build cmake-out -j16 --target install --config Release
    ```

- Run the model:
    ```bash
    cd executorch-rs/examples/llama3
    EXECUTORCH_RS_EXECUTORCH_LIB_DIR=/path/to/executorch/cmake-out
    cargo run -- \
        --model llama3_2.pte \
        --tokenizer vocab.json \
        --prompt "hello world" \
        --length 20
    ```

The llama model can be exported with many options, such as quantization, different data types (f32, bf16), different backends, kv caching, etc.
This example use a specific set of options, as specified above.
Different options require different export commands and modifications to the code and build script which you can play around with.
See the [llama README](https://github.com/pytorch/executorch/blob/v0.7.0/examples/models/llama/README.md) at the Cpp executorch repository for more details.
