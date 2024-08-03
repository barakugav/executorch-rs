//! Memory management classes.
//!
//! The ExecuTorch library allow the user to control memory allocation using the structs in the module.
//! This enable using the library in embedded systems where dynamic memory allocation is not allowed, or when allocation
//! is a performance bottleneck.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr;

use crate::util::Span;
use crate::{et_c, et_rs_c};

/// A class that does simple allocation based on a size and returns the pointer
/// to the memory address. It bookmarks a buffer with certain size. The
/// allocation is simply checking space and growing the cur_ pointer with each
/// allocation request.
pub struct MemoryAllocator(pub(crate) et_c::MemoryAllocator);

/// Dynamically allocates memory using malloc() and frees all pointers at
/// destruction time.
///
/// For systems with malloc(), this can be easier than using a fixed-sized
/// MemoryAllocator.
pub struct MallocMemoryAllocator(et_c::util::MallocMemoryAllocator);
impl Default for MallocMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}
impl MallocMemoryAllocator {
    /// Construct a new Malloc memory allocator.
    pub fn new() -> Self {
        Self(unsafe { et_rs_c::MallocMemoryAllocator_new() })
    }
}
impl AsMut<MemoryAllocator> for MallocMemoryAllocator {
    fn as_mut(&mut self) -> &mut MemoryAllocator {
        let allocator = unsafe {
            std::mem::transmute::<&mut et_c::util::MallocMemoryAllocator, &mut et_c::MemoryAllocator>(
                &mut self.0,
            )
        };
        unsafe {
            std::mem::transmute::<&mut et_c::MemoryAllocator, &mut MemoryAllocator>(allocator)
        }
    }
}
impl Drop for MallocMemoryAllocator {
    fn drop(&mut self) {
        unsafe { et_rs_c::MallocMemoryAllocator_destructor(&mut self.0) };
    }
}

/// A group of buffers that can be used to represent a device's memory hierarchy.
pub struct HierarchicalAllocator(et_c::HierarchicalAllocator);
impl HierarchicalAllocator {
    /// Constructs a new HierarchicalAllocator.
    ///
    /// # Arguments
    ///
    /// * `buffers` - The buffers to use for memory allocation.
    /// `buffers.size()` must be >= `MethodMeta::num_non_const_buffers()`.
    /// `buffers[N].size()` must be >= `MethodMeta::non_const_buffer_size(N)`.
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

/// A container class for allocators used during Method load and execution.
///
/// This class consolidates all dynamic memory needs for Method load and
/// execution. This can allow for heap-based as well as heap-less execution
/// (relevant to some embedded scenarios), and overall provides more control over
/// memory use.
///
/// This class, however, cannot ensure all allocation is accounted for since
/// kernel and backend implementations are free to use a separate way to allocate
/// memory (e.g., for things like scratch space). But we do suggest that backends
/// and kernels use these provided allocators whenever possible.
pub struct MemoryManager<'a>(
    pub(crate) UnsafeCell<et_c::MemoryManager>,
    PhantomData<&'a ()>,
);
impl<'a> MemoryManager<'a> {
    /// Constructs a new MemoryManager.
    ///
    /// # Arguments
    ///
    /// * `method_allocator` - The allocator to use when loading a Method and allocating its internal structures.
    /// Must outlive the Method that uses it.
    /// * `planned_memory` - The memory-planned buffers to use for mutable tensor data when executing a Method.
    /// Must outlive the Method that uses it. May be `None` if the Method does not use any memory-planned tensor data.
    /// The sizes of the buffers in this HierarchicalAllocator must agree with the corresponding
    /// `MethodMeta::num_memory_planned_buffers()` and `MethodMeta::memory_planned_buffer_size(N)` values,
    /// which are embedded in the Program.
    /// * `temp_allocator` - The allocator to use when allocating temporary data during kernel or delegate execution.
    /// Must outlive the Method that uses it. May be `None` if the Method does not use kernels or delegates that
    /// allocate temporary data. This allocator will be reset after every kernel or delegate call during execution.
    pub fn new(
        method_allocator: &'a mut impl AsMut<MemoryAllocator>,
        planned_memory: Option<&'a mut HierarchicalAllocator>,
        temp_allocator: Option<&'a mut MemoryAllocator>,
    ) -> Self {
        let planned_memory = planned_memory
            .map(|x| &mut x.0 as *mut _)
            .unwrap_or(ptr::null_mut());
        let temp_allocator = temp_allocator
            .map(|x| &mut x.0 as *mut _)
            .unwrap_or(ptr::null_mut());
        Self(
            UnsafeCell::new(et_c::MemoryManager {
                method_allocator_: &mut method_allocator.as_mut().0,
                planned_memory_: planned_memory,
                temp_allocator_: temp_allocator,
            }),
            PhantomData,
        )
    }
}
