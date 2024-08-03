//! Data loaders for loading execution plans (models) from a data source.
//!
//! Data loaders are used to load execution plans from a data source, such as a file or a buffer.
//! To include the data loader functionality, enable the `data-loader` feature.

use std::cell::UnsafeCell;

use crate::et_c;

/// Loads from a data source.
///
/// This struct is like a base class for data loaders. All other data loaders implement `AsRef<DataLoader>` and other
/// structs, such as `Program`, take a reference to `DataLoader` instead of the concrete data loader type.
pub struct DataLoader(pub(crate) UnsafeCell<et_c::DataLoader>);

#[cfg(feature = "data-loader")]
pub use file_data_loader::{BufferDataLoader, FileDataLoader, MlockConfig, MmapDataLoader};

#[cfg(feature = "data-loader")]
mod file_data_loader {
    use std::cell::UnsafeCell;
    use std::ffi::CString;
    use std::marker::PhantomData;
    use std::path::Path;

    use crate::error::Result;
    use crate::util::IntoRust;
    use crate::{et_c, et_rs_c};

    use super::DataLoader;

    /// A DataLoader that loads segments from a file, allocating the memory
    /// with `malloc()`.
    ///
    /// Note that this will keep the file open for the duration of its lifetime, to
    /// avoid the overhead of opening it again for every Load() call.
    pub struct FileDataLoader(UnsafeCell<et_c::util::FileDataLoader>);
    impl FileDataLoader {
        /// Creates a new FileDataLoader that wraps the named file.
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
        pub fn new(file_name: impl AsRef<Path>, alignment: Option<usize>) -> Result<Self> {
            let file_name = file_name.as_ref().to_str().expect("Invalid file name");
            let file_name = CString::new(file_name).unwrap();
            let alignment = alignment.unwrap_or(16);
            let loader =
                unsafe { et_c::util::FileDataLoader::from(file_name.as_ptr(), alignment) }.rs()?;
            Ok(Self(UnsafeCell::new(loader)))
        }
    }
    impl AsRef<DataLoader> for FileDataLoader {
        fn as_ref(&self) -> &DataLoader {
            // SAFETY: FileDataLoader has a single field of (UnsafeCell of) et_c::util::FileDataLoader, which is a
            // subclass of et_c::DataLoader, and DataLoaders has a single field of (UnsafeCell of) et_c::DataLoader.
            unsafe { std::mem::transmute::<&FileDataLoader, &DataLoader>(self) }
        }
    }
    impl Drop for FileDataLoader {
        fn drop(&mut self) {
            unsafe { et_c::util::FileDataLoader_FileDataLoader_destructor(self.0.get_mut()) };
        }
    }

    /// A DataLoader that loads segments from a file, allocating the memory
    /// with `mmap()`.
    ///
    /// Note that this will keep the file open for the duration of its lifetime, to
    /// avoid the overhead of opening it again for every Load() call.
    pub struct MmapDataLoader(UnsafeCell<et_c::util::MmapDataLoader>);
    impl MmapDataLoader {
        /// Creates a new MmapDataLoader that wraps the named file.
        /// Fails if the file can't be opened for reading or if its size can't be found.
        ///
        /// # Arguments
        ///
        /// * `file_name` - Path to the file to read from. The file will be kept open until the MmapDataLoader is
        /// destroyed, to avoid the overhead of opening it again for every Load() call.
        /// * `mlock_config` - How and whether to lock loaded pages with `mlock()`. Defaults to `MlockConfig::UseMlock`.
        ///
        /// # Returns
        ///
        /// A new MmapDataLoader on success.
        pub fn new(file_name: impl AsRef<Path>, mlock_config: Option<MlockConfig>) -> Result<Self> {
            let file_name = file_name.as_ref().to_str().expect("Invalid file name");
            let file_name = CString::new(file_name).unwrap();
            let mlock_config = mlock_config.unwrap_or(MlockConfig::UseMlock);
            let loader: et_c::util::MmapDataLoader =
                unsafe { et_c::util::MmapDataLoader::from(file_name.as_ptr(), mlock_config) }
                    .rs()?;
            Ok(Self(UnsafeCell::new(loader)))
        }
    }
    impl AsRef<DataLoader> for MmapDataLoader {
        fn as_ref(&self) -> &DataLoader {
            // SAFETY: MmapDataLoader has a single field of (UnsafeCell of) et_c::util::MmapDataLoader, which is a
            // subclass of et_c::DataLoader, and DataLoaders has a single field of (UnsafeCell of) et_c::DataLoader.
            unsafe { std::mem::transmute::<&MmapDataLoader, &DataLoader>(self) }
        }
    }
    impl Drop for MmapDataLoader {
        fn drop(&mut self) {
            unsafe { et_c::util::MmapDataLoader_MmapDataLoader_destructor(self.0.get_mut()) };
        }
    }

    /// A DataLoader that wraps a pre-allocated buffer. The FreeableBuffers
    /// that it returns do not actually free any data.
    ///
    /// This can be used to wrap data that is directly embedded into the firmware
    /// image, or to wrap data that was allocated elsewhere.
    #[allow(dead_code)]
    pub struct BufferDataLoader<'a>(
        UnsafeCell<et_c::util::BufferDataLoader>,
        PhantomData<&'a ()>,
    );
    impl<'a> BufferDataLoader<'a> {
        /// Creates a new BufferDataLoader that wraps the given data.
        pub fn new(data: &'a [u8]) -> Self {
            let loader =
                unsafe { et_rs_c::BufferDataLoader_new(data.as_ptr() as *const _, data.len()) };
            Self(UnsafeCell::new(loader), PhantomData)
        }
    }
    impl AsRef<DataLoader> for BufferDataLoader<'_> {
        fn as_ref(&self) -> &DataLoader {
            // SAFETY: BufferDataLoader has a single field of (UnsafeCell of) et_c::util::BufferDataLoader, which is a
            // subclass of et_c::DataLoader, and DataLoaders has a single field of (UnsafeCell of) et_c::DataLoader.
            unsafe { std::mem::transmute::<&BufferDataLoader, &DataLoader>(self) }
        }
    }

    /// Describes how and whether to lock loaded pages with `mlock()`.
    ///
    /// Using `mlock()` typically loads all of the pages immediately, and will
    /// typically ensure that they are not swapped out. The actual behavior
    /// will depend on the host system.
    pub type MlockConfig = et_c::util::MmapDataLoader_MlockConfig;
}
