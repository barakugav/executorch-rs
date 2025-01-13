fn main() {
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");

    println!("cargo::rustc-link-search={}/kernels/portable/", libs_dir);
}
