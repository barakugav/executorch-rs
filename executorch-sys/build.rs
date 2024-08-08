use std::collections::HashMap;
use std::path::{Path, PathBuf};

const EXECUTORCH_VERSION: &str = "0.2.1";

fn main() {
    // TODO: verify on runtime we use the correct version of executorch
    println!(
        "cargo:rustc-env=EXECUTORCH_RS_EXECUTORCH_VERSION={}",
        EXECUTORCH_VERSION
    );

    build_c_extension();
    generate_bindings();
    link_executorch();
}

fn build_c_extension() {
    let c_ext_dir = cpp_ext_dir();
    let mut builder = cc::Build::new();
    builder.cpp(true).std("c++17");
    // TODO: cpp executorch doesnt support nostd yet
    // if !cfg!(feature = "std") {
    //     builder.cpp_set_stdlib(None);
    //     builder.flag("-nostdlib");
    // }
    builder
        .files([c_ext_dir.join("api_utils.cpp")])
        .include(c_ext_dir.parent().unwrap())
        .include(executorch_headers().parent().unwrap());
    for define in cpp_defines() {
        builder.define(define, None);
    }
    builder.compile("executorch_rs_ext");

    println!("cargo::rerun-if-changed={}", c_ext_dir.to_str().unwrap());
}

fn generate_bindings() {
    let c_ext_dir = cpp_ext_dir();

    let bindings_h = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("bindings.hpp");
    let bindings_defines_h = c_ext_dir.parent().unwrap().join("executorch_rs_defines.h");
    let mut bindings_defines = String::new();
    for define in cpp_defines() {
        bindings_defines.push_str(&format!("#define {}\n", define));
    }

    println!("cargo::rerun-if-changed={}", bindings_h.to_str().unwrap());
    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "-I{}",
            executorch_headers().parent().unwrap().to_str().unwrap()
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
        .use_core()
        .header_contents(bindings_defines_h.to_str().unwrap(), &bindings_defines)
        .header(bindings_h.as_os_str().to_str().unwrap())
        .allowlist_file(format!(
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
        // feature data-loader
        .allowlist_item("torch::executor::util::FileDataLoader")
        .allowlist_item("torch::executor::util::MmapDataLoader")
        .allowlist_item("torch::executor::util::BufferDataLoader")
        // feature module
        .allowlist_item("torch::executor::Module")
        .blocklist_item("std::.*")
        .blocklist_item("torch::executor::Method_StepState")
        .blocklist_item("torch::executor::Method_InitializationState")
        .blocklist_item("torch::executor::Program_kMinHeadBytes")
        .blocklist_item("torch::executor::EventTracerEntry")
        // feature module
        .blocklist_item("torch::executor::Module_MethodHolder")
        .blocklist_item("torch::executor::Module_load_method")
        .blocklist_item("torch::executor::Module_is_method_loaded")
        .blocklist_item("torch::executor::Module_method_meta")
        .blocklist_item("torch::executor::Module_execute")
        .blocklist_item("torch::executor::Module_Module")
        .opaque_type("std::.*")
        .opaque_type("torch::executor::Program")
        .opaque_type("torch::executor::EventTracer")
        .opaque_type("torch::executor::FreeableBuffer")
        .opaque_type("torch::executor::Method")
        .opaque_type("torch::executor::MethodMeta")
        .opaque_type("torch::executor::TensorImpl")
        .opaque_type("torch::executor::DataLoader")
        .opaque_type("torch::executor::util::MallocMemoryAllocator")
        .opaque_type("torch::executor::Half")
        .opaque_type("torch::executor::MemoryAllocator")
        .opaque_type("torch::executor::HierarchicalAllocator")
        .opaque_type("torch::executor::TensorInfo")
        // feature data-loader
        .opaque_type("torch::executor::util::FileDataLoader")
        .opaque_type("torch::executor::util::MmapDataLoader")
        .opaque_type("torch::executor::util::BufferDataLoader")
        // feature module
        .opaque_type("torch::executor::Module")
        .rustified_enum("torch::executor::Error")
        .rustified_enum("torch::executor::ScalarType")
        .rustified_enum("torch::executor::Tag")
        .rustified_enum("torch::executor::Program_Verification")
        .rustified_enum("torch::executor::Program_HeaderStatus")
        .rustified_enum("torch::executor::TensorShapeDynamism")
        // feature data-loader
        .rustified_enum("torch::executor::util::MmapDataLoader_MlockConfig")
        // feature module
        .rustified_enum("torch::executor::Module_MlockConfig")
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
    if std::env::var("DOCS_RS").is_ok() {
        // Skip linking to the static library when building documentation
        return;
    }

    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
    let libs_dir = envsubst::substitute(
        libs_dir,
        &HashMap::from([(
            String::from("EXECUTORCH_RS_SYS_TOP"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .to_str()
                .unwrap()
                .to_string(),
        )]),
    )
    .unwrap();

    // TODO: cpp executorch doesnt support nostd yet
    // if cfg!(feature = "std") {
    println!("cargo::rustc-link-lib=c++");
    // }

    println!("cargo::rustc-link-search={}", libs_dir);
    println!("cargo::rustc-link-lib=static=executorch");
    println!("cargo::rustc-link-lib=static=executorch_no_prim_ops");

    if cfg!(feature = "data-loader") {
        println!(
            "cargo::rustc-link-search={}/extension/data_loader/",
            libs_dir
        );
        println!("cargo::rustc-link-lib=static=extension_data_loader");
    }

    if cfg!(feature = "module") {
        println!("cargo::rustc-link-search={}/extension/module/", libs_dir);
        // TODO: extension_module or extension_module_static ?
        println!("cargo::rustc-link-lib=static=extension_module_static");
    }
}

fn executorch_headers() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("executorch")
}

fn cpp_ext_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("executorch_rs_ext")
}

fn cpp_defines() -> Vec<&'static str> {
    let mut defines = vec![];
    if cfg!(feature = "data-loader") {
        defines.push("EXECUTORCH_RS_DATA_LOADER");
    }
    if cfg!(feature = "module") {
        defines.push("EXECUTORCH_RS_MODULE");
    }
    if cfg!(feature = "std") {
        defines.push("EXECUTORCH_RS_STD");
    }
    defines
}
