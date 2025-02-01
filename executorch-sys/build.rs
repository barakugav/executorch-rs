use std::path::{Path, PathBuf};

const EXECUTORCH_VERSION: &str = "0.4.0";

fn main() {
    // TODO: verify on runtime we use the correct version of executorch
    println!(
        "cargo:rustc-env=EXECUTORCH_RS_EXECUTORCH_VERSION={}",
        EXECUTORCH_VERSION
    );

    build_c_bridge();
    generate_bindings();
    link_executorch();
}

fn build_c_bridge() {
    let bridge_dir = cpp_bridge_dir();
    let mut builder = cc::Build::new();
    builder.cpp(true).std("c++17");
    // TODO: cpp executorch doesnt support nostd yet
    // if !cfg!(feature = "std") {
    //     builder.cpp_set_stdlib(None);
    //     builder.flag("-nostdlib");
    // }
    builder
        .files([bridge_dir.join("bridge.cpp")])
        .include(bridge_dir.parent().unwrap())
        .include(executorch_headers().parent().unwrap());
    for define in cpp_defines() {
        builder.define(define, None);
    }
    builder.compile("executorch_rs");

    println!("cargo::rerun-if-changed={}", bridge_dir.to_str().unwrap());
}

fn generate_bindings() {
    let bridge_dir = cpp_bridge_dir();
    let cpp_dir = Path::new(&env!("CARGO_MANIFEST_DIR")).join("cpp");
    println!("cargo::rerun-if-changed={}", cpp_dir.to_str().unwrap());

    let bindings_h = cpp_dir.join("bindings.hpp");
    let bindings_defines_h = bridge_dir.join("defines.h");
    let mut bindings_defines = String::from("#pragma once\n");
    for define in cpp_defines() {
        bindings_defines.push_str(&format!("#define {}\n", define));
    }

    let rust_version: [&str; 3] = env!("CARGO_PKG_RUST_VERSION")
        .split('.')
        .collect::<Vec<_>>()
        .try_into()
        .expect("Rust version is not in the format MAJOR.MINOR.PATCH");
    let [_, minor, patch] = rust_version.map(|v| v.parse::<u64>().expect("Invalid rust version"));
    let rust_version = bindgen::RustTarget::stable(minor, patch)
        .map_err(|e| format!("{}", e))
        .expect("Rust version not supported by bindgen");

    let bindings = bindgen::Builder::default()
        .rust_target(rust_version)
        .clang_arg(format!(
            "-I{}",
            executorch_headers().parent().unwrap().to_str().unwrap()
        ))
        .clang_arg(format!(
            "-I{}",
            bridge_dir.parent().unwrap().to_str().unwrap()
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
            bridge_dir.to_str().unwrap(),
        ))
        .allowlist_item("et_pal_init")
        .allowlist_item("executorch::runtime::EValue")
        .allowlist_item("executorch::runtime::Program")
        .allowlist_item("executorch::runtime::DataLoader")
        .allowlist_item("executorch::runtime::MemoryManager")
        .allowlist_item("executorch::runtime::MethodMeta")
        .allowlist_item("executorch::extension::MallocMemoryAllocator")
        .blocklist_item("executorch::runtime::Result")
        .blocklist_item("executorch::runtime::Span")
        .blocklist_item("executorch::runtime::Span_iterator")
        .blocklist_item("executorch::runtime::ArrayRef")
        .blocklist_item("executorch::runtime::ArrayRef_iterator")
        .blocklist_item("executorch::runtime::ArrayRef_const_iterator")
        .blocklist_item("executorch::runtime::ArrayRef_size_type")
        .blocklist_item("executorch::runtime::ArrayRef_value_type")
        .blocklist_item("executorch::aten::optional")
        .blocklist_item("executorch::aten::optional_value_type")
        .blocklist_item("executorch::aten::optional_storage_t")
        .blocklist_item("executorch::runtime::BoxedEvalueList")
        .blocklist_item("executorch::runtime::Result_value_type")
        .blocklist_item("executorch::runtime::Result__bindgen_ty_1")
        // feature data-loader
        .allowlist_item("executorch::extension::FileDataLoader")
        .allowlist_item("executorch::extension::MmapDataLoader")
        .allowlist_item("executorch::extension::BufferDataLoader")
        // feature module
        .allowlist_item("executorch::extension::Module")
        .blocklist_item("std::.*")
        .blocklist_item("executorch::runtime::Method_StepState")
        .blocklist_item("executorch::runtime::Method_InitializationState")
        .blocklist_item("executorch::runtime::Program_kMinHeadBytes")
        .blocklist_item("executorch::runtime::EventTracerEntry")
        // feature module
        .blocklist_item("executorch::extension::Module_MethodHolder")
        .blocklist_item("executorch::extension::Module_load_method")
        .blocklist_item("executorch::extension::Module_is_method_loaded")
        .blocklist_item("executorch::extension::Module_method_meta")
        .blocklist_item("executorch::extension::Module_execute")
        .blocklist_item("executorch::extension::Module_Module")
        .opaque_type("std::.*")
        .opaque_type("executorch::runtime::Program")
        .opaque_type("executorch::runtime::EventTracer")
        .opaque_type("executorch::runtime::Method")
        .opaque_type("executorch::runtime::MethodMeta")
        .opaque_type("executorch::runtime::etensor::TensorImpl")
        .opaque_type("executorch::runtime::DataLoader")
        .opaque_type("executorch::extension::MallocMemoryAllocator")
        .opaque_type("executorch::runtime::Half")
        .opaque_type("executorch::runtime::MemoryAllocator")
        .opaque_type("executorch::runtime::HierarchicalAllocator")
        .opaque_type("executorch::runtime::TensorInfo")
        .opaque_type("executorch::runtime::EValue_Payload_TriviallyCopyablePayload")
        .blocklist_item("executorch::runtime::MethodMeta::input_tag")
        .blocklist_item("executorch::runtime::MethodMeta::input_tensor_meta")
        .blocklist_item("executorch::runtime::MethodMeta::output_tag")
        .blocklist_item("executorch::runtime::MethodMeta::output_tensor_meta")
        .blocklist_item("executorch::runtime::MethodMeta::memory_planned_buffer_size")
        .blocklist_item("executorch::runtime::MethodMeta_input_tag")
        .blocklist_item("executorch::runtime::MethodMeta_input_tensor_meta")
        .blocklist_item("executorch::runtime::MethodMeta_output_tag")
        .blocklist_item("executorch::runtime::MethodMeta_output_tensor_meta")
        .blocklist_item("executorch::runtime::MethodMeta_memory_planned_buffer_size")
        .blocklist_item("executorch::runtime::TensorInfo_sizes")
        .blocklist_item("executorch::runtime::TensorInfo_dim_order")
        .blocklist_item("executorch::runtime::Method_set_inputs")
        .blocklist_item("executorch::runtime::Program_load")
        .blocklist_item("executorch::runtime::Program_get_constant_buffer_data")
        .blocklist_item("executorch::runtime::Program_get_method_name")
        .blocklist_item("executorch::runtime::Program_load_method")
        .blocklist_item("executorch::runtime::Program_method_meta")
        .blocklist_item("executorch::runtime::Program_get_non_const_buffer_size")
        .blocklist_item("executorch::runtime::Program_num_non_const_buffers")
        .blocklist_item("executorch::runtime::Program_get_output_flattening_encoding")
        .blocklist_item("executorch::runtime::DelegateDebugIdType")
        .blocklist_item("executorch::runtime::DebugHandle")
        .blocklist_item("executorch::runtime::ChainID")
        .blocklist_item("executorch::runtime::AllocatorID")
        .blocklist_item("executorch::runtime::FreeableBuffer")
        .blocklist_item("executorch::runtime::LoggedEValueType")
        .blocklist_item("executorch::extension::FileDataLoader_from")
        .blocklist_item("executorch::extension::FileDataLoader_load")
        .blocklist_item("executorch::extension::FileDataLoader_size")
        .blocklist_item("executorch::extension::MmapDataLoader_from")
        .blocklist_item("executorch::extension::MmapDataLoader_load")
        .blocklist_item("executorch::extension::MmapDataLoader_size")
        .blocklist_item("executorch::extension::Module_set_input")
        .blocklist_item("executorch::extension::Module_set_inputs")
        .blocklist_item("executorch::extension::Module_set_output")
        .blocklist_item("executorch::extension::Module_Module2")
        // feature data-loader
        .opaque_type("executorch::extension::FileDataLoader")
        .opaque_type("executorch::extension::MmapDataLoader")
        .opaque_type("executorch::extension::BufferDataLoader")
        // feature module
        .opaque_type("executorch::extension::Module")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        // feature data-loader
        .rustified_enum("executorch::extension::MmapDataLoader_MlockConfig")
        // feature module
        .rustified_enum("executorch::extension::Module_MlockConfig")
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

    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");

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

fn cpp_bridge_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("cpp")
        .join("executorch_rs")
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
