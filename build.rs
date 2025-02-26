use std::env;
use std::process::Command;

fn main() {
    if let Some(rustc) = rustc_version() {
        if rustc.minor >= 80 {
            println!("cargo:rustc-check-cfg=cfg(error_in_core)");
        }

        if rustc.minor >= 81 {
            // core::error::Error
            println!("cargo:rustc-cfg=error_in_core");
        }
    }
}

struct RustVersion {
    #[allow(dead_code)]
    version: String,
    minor: u32,
}

fn rustc_version() -> Option<RustVersion> {
    // Code copied from cxx crate

    let rustc = env::var_os("RUSTC")?;
    let output = Command::new(rustc).arg("--version").output().ok()?;
    let version = String::from_utf8(output.stdout).ok()?;
    let mut pieces = version.split('.');
    if pieces.next() != Some("rustc 1") {
        return None;
    }
    let minor = pieces.next()?.parse().ok()?;
    Some(RustVersion { version, minor })
}
