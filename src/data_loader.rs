//! Data loaders for loading execution plans (models) from a data source.
//!
//! Data loaders are used to load execution plans from a data source, such as a file or a buffer.
//! To include the data loader functionality, enable the `data-loader` feature.

use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::et_rs_c;

/// Loads from a data source.
///
/// This struct is like a base class for data loaders. All other data loaders implement `AsRef<DataLoader>` and other
/// structs, such as [`Program`], take a reference to [`DataLoader`] instead of the concrete data loader type.
///
/// [`Program`]: crate::program::Program
pub struct DataLoader(pub(crate) UnsafeCell<et_rs_c::DataLoader>);
impl DataLoader {
    pub(crate) fn from_inner_ref(loader: &et_rs_c::DataLoader) -> &Self {
        // Safety: Self has a single field of (UnsafeCell of) et_c::DataLoader
        unsafe { std::mem::transmute(loader) }
    }
}

/// A DataLoader that wraps a pre-allocated buffer. The FreeableBuffers
/// that it returns do not actually free any data.
///
/// This can be used to wrap data that is directly embedded into the firmware
/// image, or to wrap data that was allocated elsewhere.
pub struct BufferDataLoader<'a>(UnsafeCell<et_rs_c::BufferDataLoader>, PhantomData<&'a ()>);
impl<'a> BufferDataLoader<'a> {
    /// Creates a new BufferDataLoader that wraps the given data.
    pub fn new(data: &'a [u8]) -> Self {
        // Safety: the returned Self has a lifetime guaranteeing it will not outlive the buffer
        let loader = unsafe { et_rs_c::BufferDataLoader_new(data.as_ptr().cast(), data.len()) };
        Self(UnsafeCell::new(loader), PhantomData)
    }
}
impl AsRef<DataLoader> for BufferDataLoader<'_> {
    fn as_ref(&self) -> &DataLoader {
        let self_ = unsafe { &*self.0.get() };
        let loader = unsafe { &*et_rs_c::executorch_BufferDataLoader_as_data_loader(self_) };
        DataLoader::from_inner_ref(loader)
    }
}

#[cfg(feature = "data-loader")]
pub use file_data_loader::{FileDataLoader, MlockConfig, MmapDataLoader};

#[cfg(feature = "data-loader")]
mod file_data_loader {
    use std::cell::UnsafeCell;
    use std::ffi::CStr;

    use crate::error::try_new;
    use crate::{et_c, et_rs_c, Result};

    use super::DataLoader;

    /// A DataLoader that loads segments from a file, allocating the memory
    /// with `malloc()`.
    ///
    /// Note that this will keep the file open for the duration of its lifetime, to
    /// avoid the overhead of opening it again for every load() call.
    pub struct FileDataLoader(UnsafeCell<et_rs_c::FileDataLoader>);
    impl FileDataLoader {
        /// Creates a new FileDataLoader given a [`Path`](std::path::Path).
        ///
        /// # Arguments
        ///
        /// * `file_name` - Path to the file to read from.
        /// * `alignment` - Alignment in bytes of pointers returned by this instance. Must be a power of two.
        /// Defaults to 16.
        ///
        /// # Returns
        ///
        /// A new FileDataLoader on success.
        ///
        /// # Errors
        ///
        /// * `Error::InvalidArgument` - `alignment` is not a power of two.
        /// * `Error::AccessFailed` - `file_name` could not be opened, or its size could not be found.
        /// * `Error::MemoryAllocationFailed` - Internal memory allocation failure.
        ///
        /// # Panics
        ///
        /// Panics if `file_name` is not a valid UTF-8 string or if it contains a null byte.
        #[cfg(feature = "std")]
        pub fn from_path(
            file_name: impl AsRef<std::path::Path>,
            alignment: Option<usize>,
        ) -> Result<Self> {
            let file_name = file_name.as_ref().to_str().expect("Invalid file name");
            let file_name = std::ffi::CString::new(file_name).unwrap();
            Self::from_path_cstr(&file_name, alignment)
        }

        /// Creates a new FileDataLoader given a [`CStr`](std::ffi::CStr).
        ///
        /// This function is useful when compiling with `no_std`.
        ///
        /// # Arguments
        ///
        /// * `file_name` - Path to the file to read from.
        /// * `alignment` - Alignment in bytes of pointers returned by this instance. Must be a power of two.
        /// Defaults to 16.
        ///
        /// # Returns
        ///
        /// A new FileDataLoader on success.
        ///
        /// # Errors
        ///
        /// * `Error::InvalidArgument` - `alignment` is not a power of two.
        /// * `Error::AccessFailed` - `file_name` could not be opened, or its size could not be found.
        /// * `Error::MemoryAllocationFailed` - Internal memory allocation failure.
        ///
        /// # Safety
        ///
        /// The `file_name` should be a valid UTF-8 string and not contains a null byte other than the one at the end.
        pub fn from_path_cstr(file_name: &CStr, alignment: Option<usize>) -> Result<Self> {
            let alignment = alignment.unwrap_or(16);
            let loader = try_new(|loader| unsafe {
                et_rs_c::FileDataLoader_new(file_name.as_ptr(), alignment, loader)
            })?;
            Ok(Self(UnsafeCell::new(loader)))
        }
    }
    impl AsRef<DataLoader> for FileDataLoader {
        fn as_ref(&self) -> &DataLoader {
            let self_ = unsafe { &*self.0.get() };
            let loader = unsafe { &*et_rs_c::executorch_FileDataLoader_as_data_loader(self_) };
            DataLoader::from_inner_ref(loader)
        }
    }
    impl Drop for FileDataLoader {
        fn drop(&mut self) {
            unsafe { et_rs_c::executorch_FileDataLoader_destructor(self.0.get_mut()) };
        }
    }

    /// A DataLoader that loads segments from a file, allocating the memory
    /// with `mmap()`.
    ///
    /// Note that this will keep the file open for the duration of its lifetime, to
    /// avoid the overhead of opening it again for every load() call.
    pub struct MmapDataLoader(UnsafeCell<et_rs_c::MmapDataLoader>);
    impl MmapDataLoader {
        /// Creates a new MmapDataLoader from a [`Path`](std::path::Path).
        ///
        /// Fails if the file can't be opened for reading or if its size can't be found.
        ///
        /// # Arguments
        ///
        /// * `file_name` - Path to the file to read from. The file will be kept open until the MmapDataLoader is
        /// destroyed, to avoid the overhead of opening it again for every load() call.
        /// * `mlock_config` - How and whether to lock loaded pages with `mlock()`. Defaults to `MlockConfig::UseMlock`.
        ///
        /// # Returns
        ///
        /// A new MmapDataLoader on success.
        ///
        /// # Panics
        ///
        /// Panics if `file_name` is not a valid UTF-8 string or if it contains a null byte.
        #[cfg(feature = "std")]
        pub fn from_path(
            file_name: impl AsRef<std::path::Path>,
            mlock_config: Option<MlockConfig>,
        ) -> Result<Self> {
            let file_name = file_name.as_ref().to_str().expect("Invalid file name");
            let file_name = std::ffi::CString::new(file_name).unwrap();
            Self::from_path_cstr(&file_name, mlock_config)
        }

        /// Creates a new MmapDataLoader from a [`CStr`](std::ffi::CStr).
        ///
        /// This function is useful when compiling with `no_std`.
        /// Fails if the file can't be opened for reading or if its size can't be found.
        ///
        /// # Arguments
        ///
        /// * `file_name` - Path to the file to read from. The file will be kept open until the MmapDataLoader is
        /// destroyed, to avoid the overhead of opening it again for every load() call.
        /// * `mlock_config` - How and whether to lock loaded pages with `mlock()`. Defaults to `MlockConfig::UseMlock`.
        ///
        /// # Returns
        ///
        /// A new MmapDataLoader on success.
        ///
        /// # Safety
        ///
        /// The `file_name` should be a valid UTF-8 string and not contains a null byte other than the one at the end.
        pub fn from_path_cstr(file_name: &CStr, mlock_config: Option<MlockConfig>) -> Result<Self> {
            let mlock_config = mlock_config.unwrap_or(MlockConfig::UseMlock);
            let loader = try_new(|loader| unsafe {
                et_rs_c::MmapDataLoader_new(file_name.as_ptr(), mlock_config, loader)
            })?;
            Ok(Self(UnsafeCell::new(loader)))
        }
    }
    impl AsRef<DataLoader> for MmapDataLoader {
        fn as_ref(&self) -> &DataLoader {
            let self_ = unsafe { &*self.0.get() };
            let loader = unsafe { &*et_rs_c::executorch_MmapDataLoader_as_data_loader(self_) };
            DataLoader::from_inner_ref(loader)
        }
    }
    impl Drop for MmapDataLoader {
        fn drop(&mut self) {
            unsafe { et_rs_c::executorch_MmapDataLoader_destructor(self.0.get_mut()) };
        }
    }

    /// Describes how and whether to lock loaded pages with `mlock()`.
    ///
    /// Using `mlock()` typically loads all of the pages immediately, and will
    /// typically ensure that they are not swapped out. The actual behavior
    /// will depend on the host system.
    pub type MlockConfig = et_c::extension::MmapDataLoader_MlockConfig;
}
