use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const EXECUTORCH_VERSION: &str = "0.2.1";

fn main() {
    // TODO: verify on runtime we use the correct version of executorch
    println!(
        "cargo:rustc-env=EXECUTORCH_RS_EXECUTORCH_VERSION={}",
        EXECUTORCH_VERSION
    );

    let dev_executorch = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("executorch");
    let executorch_headers = if dev_executorch.exists() {
        dev_executorch
    } else {
        download_executorch()
    };
    build_c_extension(&executorch_headers);
    generate_bindings(&executorch_headers);
    link_executorch();
}

fn download_executorch() -> PathBuf {
    let cpp_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("cpp");
    let executorch_dir = cpp_dir.join("executorch");
    if !executorch_dir.exists() {
        std::fs::create_dir_all(&cpp_dir).unwrap();
        exe_cmd(
            &[
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                format!("v{}", EXECUTORCH_VERSION).as_str(),
                "https://github.com/pytorch/executorch.git",
            ],
            &cpp_dir,
            "Failed to clone executorch",
        );
    }
    executorch_dir
}

fn build_c_extension(executorch_headers: &Path) {
    let c_ext_dir = cpp_ext_dir();
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
    let c_ext_dir = cpp_ext_dir();

    let bindings_h = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("bindings.hpp");
    let bindings_defines_h = c_ext_dir.parent().unwrap().join("executorch_rs_defines.h");
    let mut bindings_defines = String::new();
    if cfg!(feature = "extension-data-loader") {
        bindings_defines.push_str("#define EXECUTORCH_RS_EXTENSION_DATA_LOADER\n");
    }

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
        .header_contents(bindings_defines_h.to_str().unwrap(), &bindings_defines)
        .header(bindings_h.as_os_str().to_str().unwrap())
        .allowlist_file(&format!(
            "{}/[a-zA-Z0-9_/]+.hpp",
            c_ext_dir.to_str().unwrap(),
        ))
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
        .no_copy(".*") // TODO: specific some exact types, regex act weird
        .manually_drop_union(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("executorch_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_executorch() {
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
    let libs_dir = envsubst::substitute(
        libs_dir,
        &HashMap::from([(
            String::from("EXECUTORCH_RS_TOP"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .to_str()
                .unwrap()
                .to_string(),
        )]),
    )
    .unwrap();

    println!("cargo::rustc-link-lib=c++");

    println!("cargo::rustc-link-search={}", libs_dir);
    println!("cargo::rustc-link-lib=static=executorch");
    println!("cargo::rustc-link-lib=static=executorch_no_prim_ops");

    if cfg!(feature = "extension-data-loader") {
        println!(
            "cargo::rustc-link-search={}/extension/data_loader/",
            libs_dir
        );
        println!("cargo::rustc-link-lib=static=extension_data_loader");
    }
}

fn cpp_ext_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("executorch_rs_ext")
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
