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
pub struct MemoryAllocator<'a>(
    pub(crate) UnsafeCell<et_c::MemoryAllocator>,
    PhantomData<&'a ()>,
);
impl<'a> MemoryAllocator<'a> {
    /// Constructs a new memory allocator using a fixed-size buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to use for memory allocation.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is larger than `u32::MAX` bytes.
    pub fn new(buffer: &'a mut [u8]) -> Self {
        let size = buffer.len().try_into().expect("usize -> u32");
        let base_addr = buffer.as_mut_ptr();
        let allocator = unsafe { et_rs_c::MemoryAllocator_new(size, base_addr) };
        Self(UnsafeCell::new(allocator), PhantomData)
    }

    /// Allocates memory of a certain size and alignment.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes to allocate.
    /// * `alignment` - The alignment of the memory to allocate.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the returned pointer is not dereferenced after the allocator is dropped.
    pub unsafe fn allocate_raw(
        &self,
        size: usize,
        alignment: usize,
    ) -> Option<*mut std::ffi::c_void> {
        let ptr = unsafe { et_rs_c::MemoryAllocator_allocate(self.0.get(), size, alignment) };
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    /// Allocates memory for a type `T` and initializes it with `Default::default()`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated memory, or `None` if allocation failed.
    ///
    /// Allocation may failed if the allocator is out of memory.
    pub fn allocate<T: Default>(&self) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        let ptr = unsafe { self.allocate_raw(size, alignment) }? as *mut T;
        unsafe { ptr.write(Default::default()) };
        Some(unsafe { &mut *ptr })
    }

    /// Allocates memory for an array of type `T` and initializes each element with `Default::default()`.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated array, or `None` if allocation failed.
    ///
    /// Allocation may failed if the allocator is out of memory.
    pub fn allocate_arr<T: Default>(&self, len: usize) -> Option<&mut [T]> {
        self.allocate_arr_fn(len, |_| Default::default())
    }

    /// Allocates memory for an array of type `T` and initializes each element with the result of the closure `f`.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the array.
    /// * `f` - The closure that initializes each element.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated array, or `None` if allocation failed.
    ///
    /// Allocation may failed if the allocator is out of memory.
    pub fn allocate_arr_fn<T>(&self, len: usize, f: impl Fn(usize) -> T) -> Option<&mut [T]> {
        let elm_size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        let actual_elm_size = (elm_size + alignment - 1) & !(alignment - 1);
        let total_size = actual_elm_size * len;

        let ptr = unsafe { self.allocate_raw(total_size, alignment) }? as *mut T;
        assert_eq!(actual_elm_size, {
            let elm0_addr =
                (&unsafe { std::slice::from_raw_parts_mut(ptr, 2) }[0]) as *const T as usize;
            let elm1_addr =
                (&unsafe { std::slice::from_raw_parts_mut(ptr, 2) }[1]) as *const T as usize;
            elm1_addr - elm0_addr
        });
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        for (i, elm) in slice.iter_mut().enumerate() {
            let ptr = elm as *mut T;
            unsafe { ptr.write(f(i)) };
        }
        Some(slice)
    }
}
impl<'a> AsRef<MemoryAllocator<'a>> for MemoryAllocator<'a> {
    fn as_ref(&self) -> &MemoryAllocator<'a> {
        self
    }
}

/// Dynamically allocates memory using malloc() and frees all pointers at
/// destruction time.
///
/// For systems with malloc(), this can be easier than using a fixed-sized
/// MemoryAllocator.
pub struct MallocMemoryAllocator(UnsafeCell<et_c::util::MallocMemoryAllocator>);
impl Default for MallocMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}
impl MallocMemoryAllocator {
    /// Construct a new Malloc memory allocator.
    pub fn new() -> Self {
        Self(UnsafeCell::new(unsafe {
            et_rs_c::MallocMemoryAllocator_new()
        }))
    }
}
impl AsRef<MemoryAllocator<'static>> for MallocMemoryAllocator {
    fn as_ref(&self) -> &MemoryAllocator<'static> {
        // Safety: MallocMemoryAllocator contains a single field of (UnsafeCell of) et_c::MemoryAllocator which is a
        // sub class of et_c::MemoryAllocator, and MemoryAllocator contains a single field of (UnsafeCell of)
        // et_c::MemoryAllocator.
        // The returned allocator have a lifetime of 'static because it does not depend on any external buffer, malloc
        // objects are alive until the program ends.
        unsafe { std::mem::transmute::<&MallocMemoryAllocator, &MemoryAllocator>(self) }
    }
}
impl Drop for MallocMemoryAllocator {
    fn drop(&mut self) {
        unsafe { et_rs_c::MallocMemoryAllocator_destructor(self.0.get()) };
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
        // Safety: The transmute is safe because the memory layout of Span<Span<u8>> and et_c::Span<et_c::Span<u8>>
        // is the same.
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
    pub fn new<'b: 'a>(
        method_allocator: &'a impl AsRef<MemoryAllocator<'b>>,
        planned_memory: Option<&'a mut HierarchicalAllocator>,
        temp_allocator: Option<&'a mut MemoryAllocator>,
    ) -> Self {
        let planned_memory = planned_memory
            .map(|x| &mut x.0 as *mut _)
            .unwrap_or(ptr::null_mut());
        let temp_allocator = temp_allocator
            .map(|x| x.0.get() as *mut _)
            .unwrap_or(ptr::null_mut());
        Self(
            UnsafeCell::new(et_c::MemoryManager {
                method_allocator_: method_allocator.as_ref().0.get(),
                planned_memory_: planned_memory,
                temp_allocator_: temp_allocator,
            }),
            PhantomData,
        )
    }
}
