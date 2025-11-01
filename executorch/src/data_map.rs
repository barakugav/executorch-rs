//! Module for `NamedDataMap` trait and implementations.

use core::ffi::CStr;
use core::marker::PhantomData;

use crate::tensor::TensorLayout;
use crate::util::{try_c_new, ArrayRef, FfiChar};
use crate::{sys, Error, Result};

/// Interface to access and retrieve data via name.
///
/// See executorch-cpp/extension/flat_tensor/ for an example.
pub trait NamedDataMap {
    #[doc(hidden)]
    fn __named_data_map_ptr(&self) -> sys::NamedDataMapRefMut;

    /// Get `TensorLayout` by key.
    fn get_tensor_layout<'a>(&'a self, key: &str) -> Result<TensorLayout<'a>> {
        let key = crate::util::str2chars(key);
        let key: &[FfiChar] =
            unsafe { std::mem::transmute::<&[std::ffi::c_char], &[FfiChar]>(key) };
        let key = ArrayRef::from_slice(key);

        let layout = try_c_new(|layout| unsafe {
            sys::executorch_NamedDataMap_get_tensor_layout(
                sys::NamedDataMapRef {
                    ptr: self.__named_data_map_ptr().ptr,
                },
                key.0,
                layout,
            )
        })?;
        Ok(unsafe { TensorLayout::from_raw(layout) })
    }

    //   ET_NODISCARD virtual Result<FreeableBuffer> get_data(
    //       executorch::aten::string_view key) const = 0;
    //   ET_NODISCARD virtual Error load_data_into(
    //       executorch::aten::string_view key,
    //       void* buffer,
    //       size_t size) const = 0;

    /// Get the number of keys in the NamedDataMap.
    fn get_num_keys(&self) -> Result<u32> {
        try_c_new(|num_keys| unsafe {
            sys::executorch_NamedDataMap_get_num_keys(
                sys::NamedDataMapRef {
                    ptr: self.__named_data_map_ptr().ptr,
                },
                num_keys,
            )
        })
    }

    /// Get the key at the given index.
    fn get_key(&self, index: u32) -> Result<&str> {
        let key = try_c_new(|key| unsafe {
            sys::executorch_NamedDataMap_get_key(
                sys::NamedDataMapRef {
                    ptr: self.__named_data_map_ptr().ptr,
                },
                index,
                key,
            )
        })?;
        let key = unsafe { CStr::from_ptr(key) };
        key.to_str().map_err(|_| Error::FromCStr)
    }
}

pub(crate) unsafe fn data_map_ptr2dyn<'a>(ptr: sys::NamedDataMapRef) -> &'a dyn NamedDataMap {
    data_map_ptr2dyn_mut(sys::NamedDataMapRefMut {
        ptr: ptr.ptr.cast_mut(),
    })
}

pub(crate) unsafe fn data_map_ptr2dyn_mut<'a>(
    ptr: sys::NamedDataMapRefMut,
) -> &'a mut dyn NamedDataMap {
    struct DynNamedDataMap {
        _void: [core::ffi::c_void; 0],
        _phantom: PhantomData<sys::NamedDataMapRefMut>,
    }
    impl NamedDataMap for DynNamedDataMap {
        fn __named_data_map_ptr(&self) -> sys::NamedDataMapRefMut {
            sys::NamedDataMapRefMut {
                ptr: &self as *const _ as *mut _,
            }
        }
    }
    let ptr = ptr.ptr as *mut DynNamedDataMap;
    &mut *ptr
}

#[cfg(feature = "flat-tensor")]
pub use flat_tensor::*;
#[cfg(feature = "flat-tensor")]
mod flat_tensor {
    use crate::data_loader::DataLoader;

    use super::*;

    /// A NamedDataMap implementation for FlatTensor-serialized data.
    pub struct FlatTensorDataMap<'a>(#[allow(unused)] sys::FlatTensorDataMap, PhantomData<&'a ()>);
    impl<'a> FlatTensorDataMap<'a> {
        /// Creates a new DataMap that wraps FlatTensor data.
        ///
        /// # Arguments
        ///
        /// * `data_loader` - loader The DataLoader that wraps the FlatTensor file.
        ///
        pub fn load(data_loader: &'a dyn DataLoader) -> Result<Self> {
            let data_map = try_c_new(|data_map| unsafe {
                sys::executorch_FlatTensorDataMap_load(data_loader.__data_loader_ptr(), data_map)
            })?;
            Ok(Self(data_map, PhantomData))
        }
    }
    impl NamedDataMap for FlatTensorDataMap<'_> {
        fn __named_data_map_ptr(&self) -> sys::NamedDataMapRefMut {
            sys::NamedDataMapRefMut {
                ptr: self as *const _ as *mut _,
            }
        }
    }
}
