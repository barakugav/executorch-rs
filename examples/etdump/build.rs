fn main() {
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");

    println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

    // ** Assume libs_dir is in the Cpp executroch! See README.md
    println!("cargo::rustc-link-search=native={libs_dir}/../third-party/flatcc/lib/");
    println!("cargo::rustc-link-lib=static:+whole-archive=flatcc_d");
}
