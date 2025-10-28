
# ETDump Example

Refer to https://pytorch.org/executorch/main/etdump.html for an overview of ETDump.

Compile the Cpp library with `EXECUTORCH_BUILD_DEVTOOLS=ON` and `EXECUTORCH_ENABLE_EVENT_TRACER=ON`, with addition to the flags required for the `hello_world` example.

This example links to some kernel static libs and expect the env variable `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` to be set, pointing to an output cmake directory of the cpp library. In addition, it links to the `flatcc_d` lib, which is outside the cmake directory, in `{CMAKE_DIR}/third-party/flatcc_ep/lib/`. It does so by assuming the cmake dir is in the `{CPP_EXECUTORCH_DIR}` directory. Change it if necessary.
