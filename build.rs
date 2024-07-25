use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    let executorch_dir = download_executorch();
    build_executorch(&executorch_dir);
    build_c_extension(&executorch_dir);
    generate_bindings(&executorch_dir);
    link_libexecutorch();
}

fn download_executorch() -> PathBuf {
    let c_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("c");
    // let c_dir = Path::new(&env!("CARGO_MANIFEST_DIR")).join("c_temp");
    let executorch_dir = c_dir.join("executorch");
    if !executorch_dir.exists() {
        std::fs::create_dir_all(&c_dir).unwrap();
        // git clone --depth 1 --branch v0.2.1 https://github.com/pytorch/executorch.git
        exe_cmd(
            &[
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.2.1",
                "https://github.com/pytorch/executorch.git",
            ],
            &c_dir,
            "Failed to clone executorch",
        );
    }
    executorch_dir
}

fn build_executorch(executorch_dir: &Path) {
    exe_cmd(
        &["git", "submodule", "sync", "--recursive"],
        &executorch_dir,
        "Failed to sync submodule",
    );

    for submodule in &[
        "backends/xnnpack/third-party/cpuinfo",
        "backends/xnnpack/third-party/pthreadpool",
        "third-party/prelude",
        "third-party/gflags",
        "third-party/googletest",
        "third-party/flatbuffers",
        "third-party/flatcc",
        "backends/xnnpack/third-party/XNNPACK",
        "backends/xnnpack/third-party/FXdiv",
        "third-party/pytorch",
    ] {
        exe_cmd(
            &["git", "submodule", "update", "--init", submodule],
            &executorch_dir,
            "Failed to update submodule",
        );
    }

    // ./install_requirements.sh
    // exe_cmd(
    //     &["./install_requirements.sh"],
    //     &executorch_dir,
    //     "Failed to install requirements",
    // );

    if !executorch_dir.join("cmake-out").exists() {
        // mkdir cmake-out && cd cmake-out && cmake \
        //     -DDEXECUTORCH_SELECT_OPS_LIST="aten::add.out" \
        //     -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
        //     -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
        //     -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
        //     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        //     -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
        //     -DEXECUTORCH_ENABLE_LOGGING=ON \
        //     ..
        std::fs::create_dir_all(executorch_dir.join("cmake-out")).unwrap();
        exe_cmd(
            &[
                "cmake",
                "-DDEXECUTORCH_SELECT_OPS_LIST=aten::add.out",
                "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
                "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
                "-DBUILD_EXECUTORCH_PORTABLE_OPS=ON",
                "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
                "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
                "-DEXECUTORCH_ENABLE_LOGGING=ON",
                "..",
            ],
            &executorch_dir.join("cmake-out"),
            "Failed to cmake executorch",
        );

        // TODO check USE_ATEN_LIB=true/false in CI
    }
    exe_cmd(
        &[
            "cmake",
            "--build",
            "cmake-out",
            format!("-j{}", num_cpus::get() + 1).as_str(),
        ],
        executorch_dir,
        "Failed to build executorch",
    );

    let cmake_out = executorch_dir.join("cmake-out");
    println!("cargo::rustc-link-search={}", cmake_out.display());
    println!(
        "cargo::rustc-link-search={}/kernels/portable/",
        cmake_out.display()
    );
    println!(
        "cargo::rustc-link-search={}/extension/data_loader/",
        cmake_out.display()
    );
}

fn build_c_extension(executorch_headers: &Path) {
    let c_ext_dir = c_ext_dir();
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .files([c_ext_dir.join("api_utils.cpp")])
        .include(c_ext_dir.parent().unwrap())
        .include(executorch_headers.parent().unwrap())
        .compile("executorch_rs_ext");

    println!("cargo::rerun-if-changed={}", c_ext_dir.to_str().unwrap());
}

fn generate_bindings(executorch_headers: &Path) {
    let c_ext_dir = c_ext_dir();

    let bindings_h = Path::new(&env!("CARGO_MANIFEST_DIR")).join("bindings.hpp");
    println!("cargo::rerun-if-changed={}", bindings_h.to_str().unwrap());
    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "-I{}",
            executorch_headers.parent().unwrap().to_str().unwrap()
        ))
        .clang_arg(format!(
            "-I{}",
            c_ext_dir.parent().unwrap().to_str().unwrap()
        ))
        .clang_arg("-std=c++17")
        .enable_cxx_namespaces()
        .emit_builtins()
        .enable_function_attribute_detection()
        .generate_cstr(true)
        .allowlist_item("et_pal_init")
        .allowlist_item("torch::executor::Result")
        .allowlist_item("torch::executor::EValue")
        .allowlist_item("torch::executor::Program")
        .allowlist_item("torch::executor::DataLoader")
        .allowlist_item("torch::executor::MemoryManager")
        .allowlist_item("torch::executor::MethodMeta")
        .allowlist_item("torch::executor::util::MallocMemoryAllocator")
        .allowlist_item("torch::executor::util::FileDataLoader")
        .blocklist_item("std::.*")
        .blocklist_item("torch::executor::Method_StepState")
        .blocklist_item("torch::executor::Method_InitializationState")
        .blocklist_item("torch::executor::Program_kMinHeadBytes")
        .allowlist_file(&format!(
            "{}/[a-zA-Z0-9_/]+.hpp",
            c_ext_dir.to_str().unwrap(),
        ))
        .no_copy(".*") // TODO: specific some exact types, regex act weird
        .manually_drop_union(".*")
        .opaque_type("std::.*")
        .opaque_type("torch::executor::Program")
        .opaque_type("torch::executor::EventTracer")
        .opaque_type("torch::executor::EventTracerEntry")
        .opaque_type("torch::executor::FreeableBuffer")
        .opaque_type("torch::executor::Method")
        .opaque_type("torch::executor::MethodMeta")
        .opaque_type("torch::executor::TensorImpl")
        .opaque_type("torch::executor::DataLoader")
        .opaque_type("torch::executor::util::MallocMemoryAllocator")
        .opaque_type("torch::executor::util::FileDataLoader")
        .opaque_type("torch::executor::Half")
        .opaque_type("torch::executor::MemoryAllocator")
        .opaque_type("torch::executor::HierarchicalAllocator")
        .opaque_type("torch::executor::TensorInfo")
        .rustified_enum("torch::executor::Error")
        .rustified_enum("torch::executor::ScalarType")
        .rustified_enum("torch::executor::Tag")
        .rustified_enum("torch::executor::Program_Verification")
        .rustified_enum("torch::executor::Program_HeaderStatus")
        .rustified_enum("torch::executor::TensorShapeDynamism")
        .header(bindings_h.as_os_str().to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("executorch_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_libexecutorch() {
    println!("cargo::rustc-link-lib=c++");
    println!("cargo::rustc-link-lib=static=executorch");
    println!("cargo::rustc-link-lib=static=executorch_no_prim_ops");
    println!("cargo::rustc-link-lib=static=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");
    println!("cargo::rustc-link-lib=static=extension_data_loader");
}

fn c_ext_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR")).join("c_ext")
}

#[track_caller]
fn exe_cmd(args: &[&str], current_dir: &Path, err_msg: &str) {
    let status = Command::new(args[0])
        .args(&args[1..])
        .current_dir(current_dir)
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .status()
        .expect(err_msg);
    assert!(status.success(), "{}", err_msg);
}
