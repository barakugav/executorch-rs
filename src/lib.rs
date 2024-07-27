#![deny(warnings)]

mod c_link;
use c_link::executorch_c::root::executorch_rs as et_rs_c;
use c_link::executorch_c::root::torch::executor as et_c;

use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;
use std::path::Path;
use std::{ffi::CString, ptr};

use evalue::EValue;
use util::IntoRust;

mod error;
pub mod evalue;
pub mod tensor;
pub use error::{Error, Result};

mod util;

pub fn pal_init() {
    unsafe { c_link::executorch_c::root::et_pal_init() };
}

pub trait DataLoader {
    fn data_loader(&self) -> RefMut<et_c::DataLoader>;
}
pub struct FileDataLoader(RefCell<et_c::util::FileDataLoader>);
impl FileDataLoader {
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
        let file_name = file_name.as_ref().to_str().expect("Invalid file name");
        let file_name = CString::new(file_name).unwrap();
        let loader = unsafe { et_c::util::FileDataLoader::from(file_name.as_ptr(), 16) }.rs()?;
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

pub struct Program<'a>(et_c::Program, PhantomData<&'a ()>);
impl<'a> Program<'a> {
    pub fn load(
        data_loader: &'a impl DataLoader,
        verification: ProgramVerification,
    ) -> Result<Self> {
        let mut data_loader = data_loader.data_loader();
        Ok(Self(
            unsafe { et_c::Program::load(&mut *data_loader, verification) }.rs()?,
            PhantomData,
        ))
    }

    pub fn method_meta(&self, method_name: &str) -> Result<MethodMeta> {
        let method_name = CString::new(method_name).unwrap();
        let meta = unsafe { et_rs_c::Program_method_meta(&self.0, method_name.as_ptr()) }.rs()?;
        Ok(MethodMeta::new(meta))
    }

    pub fn load_method(
        &self,
        method_name: &str,
        memory_manager: &'a MemoryManager,
    ) -> Result<Method> {
        let method_name = CString::new(method_name).unwrap();
        let mut memory_manager = memory_manager.0.borrow_mut();
        let method = unsafe {
            self.0
                .load_method(method_name.as_ptr(), &mut *memory_manager, ptr::null_mut())
        };
        Ok(Method(method.rs()?))
    }
}

pub type ProgramVerification = et_c::Program_Verification;

pub struct MethodMeta(et_c::MethodMeta);
impl MethodMeta {
    fn new(method_meta: et_c::MethodMeta) -> Self {
        Self(method_meta)
    }

    pub fn num_memory_planned_buffers(&self) -> usize {
        unsafe { self.0.num_memory_planned_buffers() }
    }

    pub fn memory_planned_buffer_size(&self, idx: usize) -> Result<usize> {
        unsafe { et_rs_c::MethodMeta_memory_planned_buffer_size(&self.0, idx) }
            .rs()
            .map(|v| v as usize)
    }
}

pub struct Method(et_c::Method);
impl Method {
    pub fn start_execution(&mut self) -> Execution {
        Execution::new(self)
    }

    pub fn inputs_size(&self) -> usize {
        unsafe { self.0.inputs_size() }
    }
}

pub struct Execution<'a> {
    method: &'a mut Method,
    set_inputs: u64,
}
impl<'a> Execution<'a> {
    fn new(method: &'a mut Method) -> Self {
        assert!(
            method.inputs_size() <= u64::BITS as usize,
            "more that 64 inputs for method, unsupported"
        );
        Self {
            method,
            set_inputs: 0,
        }
    }

    pub fn set_input(&mut self, input: &'a EValue<'a>, input_idx: usize) -> Result<()> {
        unsafe { self.method.0.set_input(input.inner(), input_idx) }.rs()?;
        self.set_inputs |= 1 << input_idx;
        Ok(())
    }

    pub fn execute(self) -> Result<Outputs<'a>> {
        assert_eq!(
            self.set_inputs,
            (1 << self.method.inputs_size()) - 1,
            "some inputs were not set"
        );
        unsafe { self.method.0.execute() }.rs()?;
        Ok(Outputs::new(self.method))
    }
}
pub struct Outputs<'a> {
    method: &'a mut Method,
}
impl<'a> Outputs<'a> {
    fn new(method: &'a mut Method) -> Self {
        Self { method }
    }

    pub fn get_output(&self, output_idx: usize) -> &'a EValue<'a> {
        let val = unsafe { self.method.0.get_output(output_idx) };
        // SAFETY: et_c::EValue as EValue has the same memory layout
        let ptr = val as *const EValue;
        unsafe { &*ptr }
    }
}

pub struct MallocMemoryAllocator(et_c::util::MallocMemoryAllocator);
impl MallocMemoryAllocator {
    pub fn new() -> Self {
        Self(unsafe { et_rs_c::MallocMemoryAllocator_new() })
    }
}
impl AsMut<et_c::MemoryAllocator> for MallocMemoryAllocator {
    fn as_mut(&mut self) -> &mut et_c::MemoryAllocator {
        let ptr = &mut self.0 as *mut _ as *mut et_c::MemoryAllocator;
        unsafe { &mut *ptr }
    }
}

pub struct HierarchicalAllocator(et_c::HierarchicalAllocator);
impl HierarchicalAllocator {
    pub fn new(buffers: Span<Span<u8>>) -> Self {
        Self(unsafe { et_rs_c::HierarchicalAllocator_new(std::mem::transmute(buffers)) })
    }
}

pub struct MemoryManager<'a>(RefCell<et_c::MemoryManager>, PhantomData<&'a ()>);
impl<'a> MemoryManager<'a> {
    pub fn new(
        method_allocator: &'a mut impl AsMut<et_c::MemoryAllocator>,
        planned_memory: &'a mut HierarchicalAllocator,
    ) -> Self {
        Self(
            RefCell::new(et_c::MemoryManager {
                method_allocator_: method_allocator.as_mut(),
                planned_memory_: &mut planned_memory.0,
                temp_allocator_: ptr::null_mut(),
            }),
            PhantomData,
        )
    }
}

#[allow(dead_code)]
pub struct Span<'a, T>(et_c::Span<T>, PhantomData<&'a T>);
impl<'a, T> Span<'a, T> {
    pub fn new(s: &'a mut [T]) -> Self {
        Self(
            et_c::Span {
                data_: s.as_mut_ptr(),
                length_: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }
}
