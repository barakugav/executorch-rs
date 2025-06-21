fn main() {
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");

    println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
    println!("cargo::rustc-link-lib=static=portable_kernels");
    // println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

    println!("cargo::rustc-link-search=native={libs_dir}/configurations/");
    println!("cargo::rustc-link-lib=static:+whole-archive=optimized_native_cpu_ops_lib");

    println!("cargo::rustc-link-search=native={libs_dir}/kernels/optimized/");
    println!("cargo::rustc-link-lib=static:+whole-archive=optimized_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=eigen_blas");
    println!("cargo::rustc-link-lib=static:+whole-archive=cpublas");

    println!("cargo::rustc-link-search=native={libs_dir}/kernels/quantized/");
    println!("cargo::rustc-link-lib=static:+whole-archive=quantized_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=quantized_ops_lib");

    println!("cargo::rustc-link-search=native={libs_dir}/extension/llm/custom_ops/");
    println!("cargo::rustc-link-lib=static:+whole-archive=custom_ops");

    println!("cargo::rustc-link-search=native={libs_dir}/extension/threadpool/");
    println!("cargo::rustc-link-lib=static:+whole-archive=extension_threadpool");

    // xnnpack
    println!("cargo::rustc-link-search=native={libs_dir}/backends/xnnpack/");
    println!("cargo::rustc-link-search=native={libs_dir}/backends/xnnpack/third-party/XNNPACK/");
    println!("cargo::rustc-link-search=native={libs_dir}/backends/xnnpack/third-party/cpuinfo/");
    println!(
        "cargo::rustc-link-search=native={libs_dir}/backends/xnnpack/third-party/pthreadpool/"
    );
    println!("cargo::rustc-link-lib=static:+whole-archive=xnnpack_backend");
    println!("cargo::rustc-link-lib=static:+whole-archive=XNNPACK");
    println!("cargo::rustc-link-lib=static:+whole-archive=microkernels-prod");
    println!("cargo::rustc-link-lib=static:+whole-archive=cpuinfo");
    println!("cargo::rustc-link-lib=static:+whole-archive=pthreadpool");

    // println!("cargo::rustc-link-lib=static:+whole-archive=custom_ops");
}
