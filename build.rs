use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    compile_c_api();
    generate_bindings();
    libexecutorch_link();
}

fn compile_c_api() {
    let source_dir = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("c")
        .join("executorch_rs");
    let sources = [source_dir.join("api_utils.cpp")];
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .files(sources)
        .include(Path::new(&env!("CARGO_MANIFEST_DIR")).join("c"))
        .compile("executorch_rs");

    println!("cargo::rerun-if-changed={}", source_dir.to_str().unwrap());
}

fn generate_bindings() {
    let c_api_dir = Path::new(&env!("CARGO_MANIFEST_DIR")).join("c");
    let executorch_dir = c_api_dir.join("executorch");
    if !executorch_dir.exists() {
        let status = Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg("--branch")
            .arg("v0.2.1")
            .arg("https://github.com/pytorch/executorch.git")
            .current_dir(&c_api_dir)
            .status()
            .unwrap();
        assert!(status.success());
    }

    let wrapper_h = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("c")
        .join("bindings_wrapper.hpp");
    println!("cargo::rerun-if-changed={}", wrapper_h.to_str().unwrap());
    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", c_api_dir.to_str().unwrap()))
        .clang_arg("-std=c++17")
        .enable_cxx_namespaces()
        .emit_builtins()
        .enable_function_attribute_detection()
        .generate_cstr(true)
        .allowlist_file(&format!(
            "{}/[a-zA-Z0-9_/]+.h",
            executorch_dir.to_str().unwrap(),
        ))
        .allowlist_file(&format!(
            "{}/[a-zA-Z0-9_/]+.hpp",
            c_api_dir.join("executorch_rs").to_str().unwrap(),
        ))
        // .blocklist_function("torch::executor::Program::.*method_meta.*")
        // .blocklist_function("torch::executor::.*method_meta.*")
        .no_copy(".*") // TODO: specific some exact types, regex act weird
        .manually_drop_union(".*")
        // .vtable_generation(true)
        // .bindgen_wrapper_union(".*")
        // .no_copy("torch::executor::.*")
        // .no_copy("torch::executor::optional_storage_t")
        .opaque_type("std::.*")
        .opaque_type("torch::executor::Program")
        .opaque_type("torch::executor::MethodMeta")
        .opaque_type("torch::executor::util::MallocMemoryAllocator")
        .opaque_type("torch::executor::util::FileDataLoader")
        // .opaque_type("torch::executor::Result__bindgen_ty_1")
        .header(wrapper_h.as_os_str().to_str().unwrap())
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        // .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("executorch_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn libexecutorch_link() {
    // git clone --depth 1 --branch v0.2.1 https://github.com/pytorch/executorch.git
    // git pull --rebase; git submodule update --init --recursive --rebase; git submodule sync --recursive
    // ./install_requirements.sh
    // rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake \
    //     -DDEXECUTORCH_SELECT_OPS_LIST="aten::add.out" \
    //     -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    //     -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
    //     -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
    //     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    //     -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    //     -DEXECUTORCH_ENABLE_LOGGING=ON \
    //     .. \
    //     && cd .. \
    //     && cmake --build cmake-out -j13

    println!("cargo::rustc-link-lib=c++");
    println!("cargo::rustc-link-lib=static=executorch");
    println!("cargo::rustc-link-lib=static=executorch_no_prim_ops");
    println!("cargo::rustc-link-lib=static=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");
    println!("cargo::rustc-link-lib=static=extension_data_loader");
    let cmake_out = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("c")
        .join("executorch")
        .join("cmake-out");
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
