fn main() {
    link_executorch_kernels();
    build_cpp_runtime_stubs();
}

fn link_executorch_kernels() {
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");

    // The C++ library is built with `EXECUTORCH_SELECT_OPS_LIST=aten::add.out`, which generates the
    // `executorch_selected_kernels` library that registers only the selected operators.
    // The registration is done by static initializers, so the library must be linked as a whole.
    println!("cargo::rustc-link-search=native={libs_dir}");
    println!("cargo::rustc-link-lib=static:+whole-archive=executorch_selected_kernels");
    // The kernels library is NOT linked as a whole archive, only the kernels of the registered operators are linked.
    // Some kernels (for example `rand` and `randn`) use the C++ standard library, which is not linked in this example.
    println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
    println!("cargo::rustc-link-lib=static=portable_kernels");
}

fn build_cpp_runtime_stubs() {
    println!("cargo::rerun-if-changed=src/cpp_runtime_stubs.cpp");
    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .std("c++17")
        // Don't link a C++ standard library
        .cpp_link_stdlib(None)
        .file("src/cpp_runtime_stubs.cpp")
        .link_lib_modifier("+whole-archive");
    if builder.get_compiler().is_like_msvc() {
        builder.flag("/EHs-c-").flag("/GR-");
    } else {
        builder.flag("-fno-exceptions").flag("-fno-rtti");
    }
    builder.compile("cpp_runtime_stubs");
}
