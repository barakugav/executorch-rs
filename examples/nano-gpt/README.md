# nano-gpt example

The example demonstrates how to use the `executorch` crate to run the `nano-gpt` model.

To run the example, follow these steps (note that some steps should run from the Cpp `executorch` repository and some from `executorch-rs`):

- Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up Cpp ExecuTorch. For installation run
    ```bash
    cd executorch
    ./install_requirements.sh
    ```

- Export the model and generate `.pte` file.
    ```bash
    cd executorch-rs/examples/nano-gpt
    python export_model.py
    ```


- Build executorch as follows.
    ```bash
    cd executorch

    cmake -DPYTHON_EXECUTABLE=python \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -Bcmake-out .

    cmake --build cmake-out -j16 --target install --config Release
    ```

- Run the model:
    ```bash
    cd executorch-rs/examples/nano-gpt
    EXECUTORCH_RS_EXECUTORCH_LIB_DIR=/path/to/executorch/cmake-out
    cargo run -- \
        --model nanogpt.pte \
        --tokenizer vocab.json \
        --prompt "Hello world" \
        --length 20
    ```
Note that the tokenizer is very limited and can not handle all prompts.