use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr;

use crate::util::Span;
use crate::{et_c, et_rs_c};

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

pub struct MemoryManager<'a>(pub(crate) RefCell<et_c::MemoryManager>, PhantomData<&'a ()>);
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
