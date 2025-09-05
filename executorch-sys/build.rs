use std::path::{Path, PathBuf};

// const EXECUTORCH_VERSION: &str = "0.7.0";

fn main() {
    // TODO: verify on runtime we use the correct version of executorch
    // println!(
    //     "cargo:rustc-env=EXECUTORCH_RS_EXECUTORCH_VERSION={}",
    //     EXECUTORCH_VERSION
    // );

    build_c_bridge();
    #[cfg(feature = "std")]
    build_cxx_bridge();
    generate_bindings();
    link_executorch();

    println!("cargo::rerun-if-changed={}", cpp_dir().to_str().unwrap());
    println!(
        "cargo::rerun-if-changed={}",
        third_party_dir().to_str().unwrap()
    );

    let check_cfg = rustc_version().map(|v| v >= 80).unwrap_or(false);
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_DENY_WARNINGS");
    let deny_warnings = std::env::var("EXECUTORCH_RS_DENY_WARNINGS").as_deref() == Ok("1");
    if check_cfg {
        println!("cargo:rustc-check-cfg=cfg(deny_warnings)");
    }
    if deny_warnings {
        println!("cargo:rustc-cfg=deny_warnings");
    }
}

fn build_c_bridge() {
    let sources_dir = cpp_dir().join("executorch_rs");
    let mut builder = cc::Build::new();
    common_cc(&mut builder);
    builder
        .files([sources_dir.join("c_bridge.cpp")])
        .includes(cpp_includes());
    builder.compile(&format!(
        "executorch_rs_c_bridge_{}",
        env!("CARGO_PKG_VERSION")
    ));
}

#[cfg(feature = "std")]
fn build_cxx_bridge() {
    let sources_dir = cpp_dir().join("executorch_rs");
    let mut bridges = Vec::new();
    bridges.push("src/cxx_bridge/core.rs");
    if cfg!(feature = "module") {
        bridges.push("src/cxx_bridge/module.rs");
    }
    if cfg!(feature = "tensor-ptr") {
        bridges.push("src/cxx_bridge/tensor_ptr.rs");
    }
    let mut builder = cxx_build::bridges(bridges);
    common_cc(&mut builder);
    builder
        .files([sources_dir.join("cxx_bridge.cpp")])
        .includes(cpp_includes());
    builder.compile(&format!(
        "executorch_rs_cxx_bridge_{}",
        env!("CARGO_PKG_VERSION")
    ));
}

fn common_cc(builder: &mut cc::Build) {
    builder.cpp(true).std("c++17").cpp_link_stdlib(None); // linked via link-cplusplus crate
    if !cfg!(feature = "std") {
        // TODO: cpp executorch doesnt support nostd yet
        // builder.flag("-nostdlib");
    }
    for define in cpp_defines() {
        builder.define(define, None);
    }
}

fn generate_bindings() {
    let bindings_h = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("c_bindings.h");
    let mut bindings_h_content = String::from("#pragma once\n");
    for define in cpp_defines() {
        bindings_h_content.push_str(&format!("#define {define}\n"));
    }
    bindings_h_content.push_str("#include \"executorch_rs/c_bridge.h\"\n");
    std::fs::write(&bindings_h, bindings_h_content).expect("Unable to write bindings.h");

    let builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", cpp_dir().to_str().unwrap()))
        .use_core()
        .generate_cstr(true)
        .header(bindings_h.as_os_str().to_str().unwrap())
        .allowlist_file(format!(
            "{}/executorch_rs/c_bridge.h",
            cpp_dir().to_str().unwrap()
        ))
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .no_copy(".*")
        .manually_drop_union(".*")
        .opaque_type("EValueStorage")
        .opaque_type("TensorStorage")
        .opaque_type("TensorImpl")
        .opaque_type("Program")
        .opaque_type("TensorInfo")
        .opaque_type("MethodMeta")
        .opaque_type("Method")
        .opaque_type("BufferDataLoader")
        .opaque_type("FileDataLoader")
        .opaque_type("MmapDataLoader")
        .opaque_type("MemoryAllocator")
        .opaque_type("HierarchicalAllocator")
        .opaque_type("MemoryManager")
        .opaque_type("OptionalTensorStorage")
        .opaque_type("ETDumpGen")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("executorch_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_executorch() {
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_LINK");

    let link_enabled = std::env::var("EXECUTORCH_RS_LINK").as_deref() != Ok("0");

    let check_cfg = rustc_version().map(|v| v >= 80).unwrap_or(false);

    if check_cfg {
        println!("cargo::rustc-check-cfg=cfg(link_cxx)");
    }
    if link_enabled {
        println!("cargo::rustc-cfg=link_cxx");
    }

    if std::env::var("DOCS_RS").is_ok() || !link_enabled {
        // Skip linking to the static library when building documentation
        return;
    }

    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").ok();
    if libs_dir.is_none() {
        println!("cargo::warning=EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
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

    if cfg!(feature = "etdump") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/devtools/etdump/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=etdump");
    }
}

fn cpp_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR")).join("cpp")
}

fn third_party_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR")).join("third-party")
}

fn cpp_includes() -> Vec<PathBuf> {
    let third_party_dir = third_party_dir();
    let c10_dir = std::env::var_os("EXECUTORCH_RS_C10_HEADERS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| third_party_dir.join("executorch/runtime/core/portable_type/c10"));
    assert!(
        c10_dir.exists(),
        "C10 directory does not exist: {}",
        c10_dir.display()
    );
    vec![cpp_dir(), third_party_dir.clone(), c10_dir]
}

fn cpp_defines() -> Vec<&'static str> {
    let mut defines = vec!["C10_USING_CUSTOM_GENERATED_MACROS"];
    if cfg!(feature = "data-loader") {
        defines.push("EXECUTORCH_RS_DATA_LOADER");
    }
    if cfg!(feature = "module") {
        defines.push("EXECUTORCH_RS_MODULE");
    }
    if cfg!(feature = "tensor-ptr") {
        defines.push("EXECUTORCH_RS_TENSOR_PTR");
    }
    if cfg!(feature = "etdump") {
        defines.push("EXECUTORCH_RS_ETDUMP");
    }
    if cfg!(feature = "std") {
        defines.push("EXECUTORCH_RS_STD");
    }
    defines
}

fn rustc_version() -> Option<u32> {
    // Code copied from cxx crate

    let rustc = std::env::var_os("RUSTC")?;
    let output = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()?;
    let version = String::from_utf8(output.stdout).ok()?;
    let mut pieces = version.split('.');
    if pieces.next() != Some("rustc 1") {
        return None;
    }
    let minor = pieces.next()?.parse().ok()?;
    Some(minor)
}
