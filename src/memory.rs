//! Memory management classes.
//!
//! The ExecuTorch library allow the user to control memory allocation using the structs in the module.
//! This enable using the library in embedded systems where dynamic memory allocation is not allowed, or when allocation
//! is a performance bottleneck.

use core::marker::PhantomPinned;
use core::mem::MaybeUninit;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::pin::Pin;
use std::ptr;

use crate::util::Span;
use crate::{et_c, et_rs_c};

/// A class that does simple allocation based on a size and returns the pointer
/// to the memory address. It bookmarks a buffer with certain size. The
/// allocation is simply checking space and growing the cur_ pointer with each
/// allocation request.
pub struct MemoryAllocator<'a>(
    pub(crate) UnsafeCell<et_c::runtime::MemoryAllocator>,
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
    /// # Returns
    ///
    /// A mutable reference to the allocated memory, or [`None`] if allocation failed.
    pub fn allocate_raw(&self, size: usize, alignment: usize) -> Option<&mut [u8]> {
        let ptr = unsafe { et_rs_c::MemoryAllocator_allocate(self.0.get(), size, alignment) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) })
        }
    }

    /// Allocates memory for a type `T` and initializes it with `Default::default()`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated memory, or [`None`] if allocation failed.
    ///
    /// Allocation may failed if the allocator is out of memory.
    pub fn allocate<T: Default>(&self) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        let ptr = self.allocate_raw(size, alignment)?.as_mut_ptr() as *mut T;
        unsafe { ptr.write(Default::default()) };
        Some(unsafe { &mut *ptr })
    }

    /// Allocates a pinned memory for a type `T` and initializes it with `Default::default()`.
    ///
    /// # Returns
    ///
    /// A pinned mutable reference to the allocated memory, or [`None`] if allocation failed.
    ///
    /// Allocation may failed if the allocator is out of memory.
    pub fn allocate_pinned<T: Default>(&self) -> Option<Pin<&mut T>> {
        let size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        let ptr = self.allocate_raw(size, alignment)?.as_mut_ptr() as *mut T;
        unsafe { ptr.write(Default::default()) };
        Some(unsafe { Pin::new_unchecked(&mut *ptr) })
    }

    /// Allocates memory for an array of type `T` and initializes each element with `Default::default()`.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated array, or [`None`] if allocation failed.
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
    /// A mutable reference to the allocated array, or [`None`] if allocation failed.
    ///
    /// Allocation may failed if the allocator is out of memory.
    pub fn allocate_arr_fn<T>(&self, len: usize, f: impl Fn(usize) -> T) -> Option<&mut [T]> {
        let elm_size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        let actual_elm_size = (elm_size + alignment - 1) & !(alignment - 1);
        let total_size = actual_elm_size * len;

        let ptr = self.allocate_raw(total_size, alignment)?.as_mut_ptr() as *mut T;
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

#[cfg(feature = "std")]
pub use malloc_allocator::MallocMemoryAllocator;
#[cfg(feature = "std")]
mod malloc_allocator {
    use super::*;

    /// Dynamically allocates memory using malloc() and frees all pointers at
    /// destruction time.
    ///
    /// For systems with malloc(), this can be easier than using a fixed-sized
    /// MemoryAllocator.
    pub struct MallocMemoryAllocator(UnsafeCell<et_c::extension::MallocMemoryAllocator>);
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
}

/// A group of buffers that can be used to represent a device's memory hierarchy.
pub struct HierarchicalAllocator<'a>(et_c::runtime::HierarchicalAllocator, PhantomData<&'a ()>);
impl<'a> HierarchicalAllocator<'a> {
    /// Constructs a new HierarchicalAllocator.
    ///
    /// # Arguments
    ///
    /// * `buffers` - The buffers to use for memory allocation.
    ///     `buffers.size()` must be >= `MethodMeta::num_non_const_buffers()`.
    ///     `buffers[N].size()` must be >= `MethodMeta::non_const_buffer_size(N)`.
    pub fn new(buffers: &'a mut [Span<'a, u8>]) -> Self {
        // Safety: safe because the memory layout of [Span<u8>] and [et_rs_c::SpanU8] is the same.
        let buffers: &'a mut [et_rs_c::SpanU8] = unsafe { std::mem::transmute(buffers) };
        let buffers = et_rs_c::SpanSpanU8 {
            data: buffers.as_mut_ptr(),
            len: buffers.len(),
        };
        Self(
            unsafe { et_rs_c::HierarchicalAllocator_new(buffers) },
            PhantomData,
        )
    }
}
impl Drop for HierarchicalAllocator<'_> {
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
    pub(crate) UnsafeCell<et_c::runtime::MemoryManager>,
    PhantomData<&'a ()>,
);
impl<'a> MemoryManager<'a> {
    /// Constructs a new MemoryManager.
    ///
    /// # Arguments
    ///
    /// * `method_allocator` - The allocator to use when loading a Method and allocating its internal structures.
    ///     Must outlive the Method that uses it.
    /// * `planned_memory` - The memory-planned buffers to use for mutable tensor data when executing a Method.
    ///     Must outlive the Method that uses it. May be [`None`] if the Method does not use any memory-planned tensor data.
    ///     The sizes of the buffers in this HierarchicalAllocator must agree with the corresponding
    ///     `MethodMeta::num_memory_planned_buffers()` and `MethodMeta::memory_planned_buffer_size(N)` values,
    ///     which are embedded in the Program.
    /// * `temp_allocator` - The allocator to use when allocating temporary data during kernel or delegate execution.
    ///     Must outlive the Method that uses it. May be [`None`] if the Method does not use kernels or delegates that
    ///     allocate temporary data. This allocator will be reset after every kernel or delegate call during execution.
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
            UnsafeCell::new(et_c::runtime::MemoryManager {
                method_allocator_: method_allocator.as_ref().0.get(),
                planned_memory_: planned_memory,
                temp_allocator_: temp_allocator,
            }),
            PhantomData,
        )
    }
}

/// Storage for a non-trivially movable Cpp object.
///
/// TL;DR: helper class used to store a Cpp object without moving it, relevant only in systems where the heap can not
/// be used. If allocations are available, Cpp objects can be allocated on the heap, preventing moves.
///
/// There are many differences between Rust and Cpp objects, one of them is that in Rust every object must be trivially
/// movable, namely that it should be OK to just copy the object's bytes to another location while forgetting the
/// original object. This is not the case for Cpp objects, which may implement a move constructor that does more than
/// just copying the object's bytes. If a Rust object that encapsulates a Cpp object is moved, the Cpp object may be
/// left in an invalid state. For example, this code is invalid:
/// ```rust,ignore
/// // Struct generated by bindgen matching a Cpp struct with an unsafe interface
/// #[repr(C)]
/// #[repr(align(8))]
/// pub struct StdString {
///    pub _bindgen_opaque_blob: [u64; 3usize],
/// }
///
/// // Rust struct that encapsulates the Cpp object and expose a (wrong) safe interface
/// pub struct RustString(StdString);
/// impl RustString {
///     pub fn new(s: &CStr) {
///         // Call to a Cpp function generated by bindgen
///         RustString(unsafe { StdString_new(s) })
///     }
///
///     ...
/// }
///
/// fn main() {
///    let s = RustString::new(CStr::from_bytes_with_nul(b"Hello, world!\0").unwrap());
///     do_something(s); // move occurs here, without calling the Cpp move constructor!
/// }
///
/// fn do_something(s: RustString) {
///     ...
/// }
/// ```
///
/// There is no way (currently) to create an unmovable struct in Rust, or to implement some trait that execute some code
/// when a move occur, and it is likely to remain that way as Rust gains a lot of performance from these simplifying
/// constraints. To solve this issue, most Rust libraries that use Cpp under the hood usually choose to allocate the Cpp
/// objects on the heap, and the encapsulating Rust objects stores pointers to them, rather than the Cpp objects
/// themselves:
/// ```rust,ignore
/// pub struct RustString(NonNull<StdString>);
/// ```
/// This solution works as it guarantees that the Cpp object is never moved from its original memory location. However,
/// this solution is not viable in systems where the heap can not be used, for example in embedded systems. This is
/// where the [`Storage`] struct comes in. It is a wrapper around a [`MaybeUninit`] that can be used to store a Cpp object,
/// and it is intended to be pinned in memory and later be used to allocate the Cpp object in place. This way, the
/// Cpp object is never moved, and the Rust object that keeps a pointer to the storage can be moved around freely as
/// usual.
///
/// The [`executorch`](crate) crate encapsulates Cpp objects which are trivially movable in Rust objects using the simple
/// `struct RustStruct(CppStruct)`, similar to the first example, but never does so for non-trivially movable objects.
/// For such structs, Rust structs don't have an in-place field of the underlying Cpp object, rather a pointer.
/// The pointer can point to one of three, according to what the Rust struct owns:
/// - Owns the memory, owns the object: the pointer points to an allocated Cpp object on the heap, owned by the Rust
///     struct using a [`Box`]. The destructor of the Cpp object is called when the Rust object is dropped, and the [`Box`]
///     is deallocated.
/// - Does not own the memory, owns the object: the pointer points to an allocated Cpp object in a [`Storage`], which is
///     pinned in memory, possibly on the stack (see later example). The destructor of the Cpp object is called when
///     Rust object is dropped, but the [`Storage`] is not deallocated.
/// - Does not own the memory, does not own the object: the pointer points to a Cpp object that is owned by another
///     entity, like a regular Rust reference. The destructor of the Cpp object is not called when the Rust object is
///     dropped and no deallocation is done.
///
/// Non-trivially movable objects such as [`Tensor`](crate::tensor::Tensor) and [`EValue`](crate::evalue::EValue) expose
/// a straight forward `new` function that allocate the object on the heap which should be used in most cases.
/// The `new` function is available when the `alloc` feature is enabled. When allocations are not available, the
/// [`Storage`] struct should be used in one of two ways:
/// - Create on the stack and pin a [`Storage`], and initialize the object in place through a variant of the `new` function such
///     as `new_in_storage`:
///     ```rust,ignore
///     let tensor_impl: TensorImpl = ...;
///
///     // Create a Tensor on the heap
///     let tensor = Tensor::new(&tensor_impl);
///
///     // Create a Tensor on the stack
///     let storage = pin::pin!(Storage::<Tensor<f32>>::default());
///     let tensor = Tensor::new_in_storage(&tensor_impl, storage);
///     ```
/// - Use a [`MemoryAllocator`] to allocate a [`Storage`] object, and use it to allocate the object in place:
///     ```rust,ignore
///     let tensor: Tensor = ...;
///
///     // Create am EValue on the heap
///     let evalue = EValue::new(tensor);
///
///     // Create an EValue in a memory allocated by the allocator
///     let allocator: impl AsRef<MemoryAllocator> = ...; // usually global
///     let evalue = EValue::new_in_storage(tensor, allocator.as_ref().allocate_pinned().unwrap());
///     ```
#[repr(transparent)]
pub struct Storage<T: Storable>(MaybeUninit<T::Storage>, PhantomPinned);
impl<T: Storable> Default for Storage<T> {
    /// Create a new [`Storage`] object with an uninitialized inner value.
    fn default() -> Self {
        Self(MaybeUninit::uninit(), PhantomPinned)
    }
}
impl<T: Storable> Storage<T> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T::Storage {
        self.0.as_mut_ptr()
    }
}
/// A marker trait for types that can be stored in a [`Storage`].
///
/// Usually the type is a Cpp object that is not trivially movable. See the [`Storage`] struct for more information.
pub trait Storable {
    /// The underlying Cpp object type, defining the memory layout of a [`Storage`] object.
    type Storage;
}
