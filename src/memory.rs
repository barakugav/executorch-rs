use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr;

use crate::{et_c, et_rs_c, Span};

pub trait MemoryAllocator {
    fn memory_allocator(&mut self) -> &mut et_c::MemoryAllocator;
}
pub struct MallocMemoryAllocator(et_c::util::MallocMemoryAllocator);
impl MallocMemoryAllocator {
    pub fn new() -> Self {
        Self(unsafe { et_rs_c::MallocMemoryAllocator_new() })
    }
}
impl MemoryAllocator for MallocMemoryAllocator {
    fn memory_allocator(&mut self) -> &mut et_c::MemoryAllocator {
        let ptr = &mut self.0 as *mut _ as *mut et_c::MemoryAllocator;
        unsafe { &mut *ptr }
    }
}
impl Drop for MallocMemoryAllocator {
    fn drop(&mut self) {
        unsafe { et_rs_c::MallocMemoryAllocator_destructor(&mut self.0) };
    }
}

pub struct HierarchicalAllocator(et_c::HierarchicalAllocator);
impl HierarchicalAllocator {
    pub fn new(buffers: Span<Span<u8>>) -> Self {
        // Safety: The transmute is safe because the memory layout of SpanMut<u8> and et_c::Span<et_c::Span<u8>> is the same.
        let buffers: et_c::Span<et_c::Span<u8>> = unsafe { std::mem::transmute(buffers) };
        Self(unsafe { et_rs_c::HierarchicalAllocator_new(buffers) })
    }
}
impl Drop for HierarchicalAllocator {
    fn drop(&mut self) {
        unsafe { et_rs_c::HierarchicalAllocator_destructor(&mut self.0) };
    }
}

pub struct MemoryManager<'a>(pub(crate) RefCell<et_c::MemoryManager>, PhantomData<&'a ()>);
impl<'a> MemoryManager<'a> {
    pub fn new(
        method_allocator: &'a mut impl MemoryAllocator,
        planned_memory: &'a mut HierarchicalAllocator,
    ) -> Self {
        Self(
            RefCell::new(et_c::MemoryManager {
                method_allocator_: method_allocator.memory_allocator(),
                planned_memory_: &mut planned_memory.0,
                temp_allocator_: ptr::null_mut(),
            }),
            PhantomData,
        )
    }
}
