#![allow(dead_code)]

use core::ffi::CStr;
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};

#[cfg(feature = "std")]
pub fn add_model_path() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("models")
        .join("add.pte")
}

pub const ADD_MODEL_PATH_CSTR: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/models/add.pte", "\0").as_bytes(),
    )
};

pub const ADD_MODEL_BYTES: &[u8] = include_bytes!("../examples/models/add.pte");
