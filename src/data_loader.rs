use std::cell::RefMut;

use crate::et_c;

pub trait DataLoader {
    fn data_loader(&self) -> RefMut<et_c::DataLoader>;
}

#[cfg(feature = "extension-data-loader")]
pub use file_data_loader::FileDataLoader;

#[cfg(feature = "extension-data-loader")]
mod file_data_loader {
    use std::cell::{RefCell, RefMut};
    use std::ffi::CString;
    use std::path::Path;

    use crate::util::IntoRust;
    use crate::{et_c, Result};

    use super::DataLoader;

    pub struct FileDataLoader(RefCell<et_c::util::FileDataLoader>);
    impl FileDataLoader {
        pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
            let file_name = file_name.as_ref().to_str().expect("Invalid file name");
            let file_name = CString::new(file_name).unwrap();
            let loader =
                unsafe { et_c::util::FileDataLoader::from(file_name.as_ptr(), 16) }.rs()?;
            Ok(Self(RefCell::new(loader)))
        }
    }
    impl DataLoader for FileDataLoader {
        fn data_loader(&self) -> RefMut<et_c::DataLoader> {
            RefMut::map(self.0.borrow_mut(), |loader| {
                let ptr = loader as *mut _ as *mut et_c::DataLoader;
                unsafe { &mut *ptr }
            })
        }
    }
    impl Drop for FileDataLoader {
        fn drop(&mut self) {
            unsafe {
                et_c::util::FileDataLoader_FileDataLoader_destructor(&mut *self.0.borrow_mut())
            };
        }
    }
}
