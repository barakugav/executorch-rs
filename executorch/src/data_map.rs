//! Module for `NamedDataMap` trait and implementations.

use core::ffi::CStr;

use executorch_sys as sys;

use crate::tensor::TensorLayout;
use crate::util::{try_c_new, ArrayRef};
use crate::{Error, Result};

/// Interface to access and retrieve data via name.
///
/// See executorch-cpp/extension/flat_tensor/ for an example.
pub struct NamedDataMap {
    _void: [core::ffi::c_void; 0],
}
impl NamedDataMap {
    /// Get `TensorLayout` by key.
    pub fn get_tensor_layout<'a>(&'a self, key: &str) -> Result<TensorLayout<'a>> {
        let key = ArrayRef::from_chars(crate::util::str2chars(key));

        // Safety: sys::executorch_NamedDataMap_get_tensor_layout writes to the pointer.
        let layout = unsafe {
            try_c_new(|layout| {
                sys::executorch_NamedDataMap_get_tensor_layout(
                    sys::NamedDataMapRef {
                        ptr: self as *const _ as *mut _,
                    },
                    key.0,
                    layout,
                )
            })?
        };
        // Safety: `layout` is valid for the lifetime of self
        Ok(unsafe { TensorLayout::from_raw(layout) })
    }

    //   ET_NODISCARD virtual Result<FreeableBuffer> get_data(
    //       executorch::aten::string_view key) const = 0;
    //   ET_NODISCARD virtual Error load_data_into(
    //       executorch::aten::string_view key,
    //       void* buffer,
    //       size_t size) const = 0;

    /// Get the number of keys in the NamedDataMap.
    pub fn get_num_keys(&self) -> Result<u32> {
        // Safety: sys::executorch_NamedDataMap_get_num_keys writes to the pointer.
        unsafe {
            try_c_new(|num_keys| {
                sys::executorch_NamedDataMap_get_num_keys(
                    sys::NamedDataMapRef {
                        ptr: self as *const _ as *mut _,
                    },
                    num_keys,
                )
            })
        }
    }

    /// Get the key at the given index.
    pub fn get_key(&self, index: u32) -> Result<&str> {
        // Safety: sys::executorch_NamedDataMap_get_key writes to the pointer.
        let key = unsafe {
            try_c_new(|key| {
                sys::executorch_NamedDataMap_get_key(
                    sys::NamedDataMapRef {
                        ptr: self as *const _ as *mut _,
                    },
                    index,
                    key,
                )
            })?
        };
        let key = unsafe { CStr::from_ptr(key) };
        key.to_str().map_err(|_| Error::InvalidString)
    }
}

#[cfg(feature = "flat-tensor")]
pub use flat_tensor::*;
#[cfg(feature = "flat-tensor")]
mod flat_tensor {
    use core::marker::PhantomData;

    use crate::data_loader::DataLoader;

    use super::*;

    /// A NamedDataMap implementation for FlatTensor-serialized data.
    pub struct FlatTensorDataMap<'a>(sys::FlatTensorDataMap, PhantomData<&'a ()>);
    impl<'a> FlatTensorDataMap<'a> {
        /// Creates a new DataMap that wraps FlatTensor data.
        ///
        /// # Arguments
        ///
        /// * `data_loader` - loader The DataLoader that wraps the FlatTensor file.
        ///
        pub fn load(data_loader: &'a DataLoader) -> Result<Self> {
            let data_loader = sys::DataLoaderRefMut {
                ptr: data_loader as *const _ as *mut _,
            };
            // Safety: sys::executorch_FlatTensorDataMap_load writes to the pointer.
            let data_map = unsafe {
                try_c_new(|data_map| sys::executorch_FlatTensorDataMap_load(data_loader, data_map))?
            };
            Ok(Self(data_map, PhantomData))
        }
    }
    impl AsRef<NamedDataMap> for FlatTensorDataMap<'_> {
        fn as_ref(&self) -> &NamedDataMap {
            let map = unsafe {
                sys::executorch_FlatTensorDataMap_as_named_data_map_mut(
                    &self.0 as *const _ as *mut _,
                )
            };
            unsafe { &*map.ptr.cast() }
        }
    }
}
