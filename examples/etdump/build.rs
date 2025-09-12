fn main() {
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");

    println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

    let flatc_dir = format!("{libs_dir}/third-party/flatcc_external_project/lib");
    let dir_files = std::fs::read_dir(&flatc_dir)
        .unwrap_or_else(|_| panic!("failed to read flatcc lib dir: {flatc_dir}"))
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_file())
        .map(|entry| entry.file_name().to_str().unwrap().to_string())
        .collect::<Vec<_>>();
    let flatc_lib_name = if dir_files.iter().any(|path| path == "libflatcc.a") {
        "flatcc"
    } else if dir_files.iter().any(|path| path == "libflatcc_d.a") {
        "flatcc_d"
    } else {
        "flatcc"
    };
    println!("cargo::rustc-link-search=native={flatc_dir}/");
    println!("cargo::rustc-link-lib=static:+whole-archive={flatc_lib_name}");
}
