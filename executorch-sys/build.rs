use std::path::{Path, PathBuf};

// const EXECUTORCH_VERSION: &str = "0.5.0";

fn main() {
    // TODO: verify on runtime we use the correct version of executorch
    // println!(
    //     "cargo:rustc-env=EXECUTORCH_RS_EXECUTORCH_VERSION={}",
    //     EXECUTORCH_VERSION
    // );

    build_c_bridge();
    #[cfg(any(feature = "tensor-ptr", feature = "module"))]
    build_cxx_bridge();
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
        .files([bridge_dir.join("c_bridge.cpp")])
        .include(bridge_dir.parent().unwrap())
        .include(executorch_headers().parent().unwrap());
    for define in cpp_defines() {
        builder.define(define, None);
    }
    let c_bridge_lib_name = format!("executorch_rs_c_bridge_{}", env!("CARGO_PKG_VERSION"));
    builder.compile(&c_bridge_lib_name);

    println!("cargo::rerun-if-changed={}", bridge_dir.to_str().unwrap());
}

#[cfg(any(feature = "tensor-ptr", feature = "module"))]
fn build_cxx_bridge() {
    let bridge_dir = cpp_bridge_dir();
    let mut bridges = Vec::new();
    if cfg!(feature = "module") {
        bridges.push("src/cxx_bridge/module.rs");
    }
    if cfg!(feature = "tensor-ptr") {
        bridges.push("src/cxx_bridge/tensor_ptr.rs");
    }
    let mut builder = cxx_build::bridges(bridges);
    builder.cpp(true).std("c++17");
    // TODO: cpp executorch doesnt support nostd yet
    // if !cfg!(feature = "std") {
    //     builder.cpp_set_stdlib(None);
    //     builder.flag("-nostdlib");
    // }
    builder
        .files([bridge_dir.join("cxx_bridge.cpp")])
        .include(bridge_dir.parent().unwrap())
        .include(executorch_headers().parent().unwrap());
    for define in cpp_defines() {
        builder.define(define, None);
    }
    let cxx_bridge_lib_name = format!("executorch_rs_cxx_bridge_{}", env!("CARGO_PKG_VERSION"));
    builder.compile(&cxx_bridge_lib_name);

    println!("cargo::rerun-if-changed={}", bridge_dir.to_str().unwrap());
}

fn generate_bindings() {
    let bridge_dir = cpp_bridge_dir();
    let cpp_dir = Path::new(&env!("CARGO_MANIFEST_DIR")).join("cpp");
    println!("cargo::rerun-if-changed={}", cpp_dir.to_str().unwrap());

    let bindings_h = cpp_dir.join("bindings.h");
    let bindings_defines_h = bridge_dir.join("defines.h");
    let mut bindings_defines = String::from("#pragma once\n");
    for define in cpp_defines() {
        bindings_defines.push_str(&format!("#define {}\n", define));
    }

    let [_, minor, patch]: [u64; 3] = env!("CARGO_PKG_RUST_VERSION")
        .split('.')
        .map(|v| v.parse::<u64>().expect("Invalid rust version"))
        .collect::<Vec<_>>()
        .try_into()
        .expect("Rust version is not in the format MAJOR.MINOR.PATCH");
    let rust_version = bindgen::RustTarget::stable(minor, patch)
        .map_err(|e| format!("{}", e))
        .expect("Rust version not supported by bindgen");

    let builder = bindgen::Builder::default()
        .rust_target(rust_version)
        .clang_arg(format!(
            "-I{}",
            bridge_dir.parent().unwrap().to_str().unwrap()
        ))
        .use_core()
        .header_contents(bindings_defines_h.to_str().unwrap(), &bindings_defines)
        .header(bindings_h.as_os_str().to_str().unwrap())
        .allowlist_file(format!("{}/c_bridge.h", bridge_dir.to_str().unwrap(),))
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .no_copy(".*") // TODO: specific some exact types, regex act weird
        .manually_drop_union(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("executorch_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_executorch() {
    if std::env::var("DOCS_RS").is_ok() || std::env::var("EXECUTORCH_RS_LINK").as_deref() == Ok("0")
    {
        // Skip linking to the static library when building documentation
        return;
    }

    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").ok();
    if libs_dir.is_none() {
        println!("cargo::warning=EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
    }

    if cfg!(feature = "std") {
        println!("cargo::rustc-link-lib=c++");
    }

    if let Some(libs_dir) = &libs_dir {
        println!("cargo::rustc-link-search=native={libs_dir}");
    }
    println!("cargo::rustc-link-lib=static:+whole-archive=executorch");
    println!("cargo::rustc-link-lib=static:+whole-archive=executorch_core");

    if cfg!(feature = "data-loader") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/extension/data_loader/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=extension_data_loader");
    }

    if cfg!(feature = "module") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/extension/module/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=extension_module_static");
    }

    if cfg!(feature = "tensor-ptr") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/extension/tensor/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=extension_tensor");
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
    if cfg!(feature = "tensor-ptr") {
        defines.push("EXECUTORCH_RS_TENSOR_PTR");
    }
    if cfg!(feature = "std") {
        defines.push("EXECUTORCH_RS_STD");
    }
    defines
}
