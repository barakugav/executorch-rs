//! Memory management classes.
//!
//! The ExecuTorch library allow the user to control memory allocation using the structs in the module.
//! This enable using the library in embedded systems where dynamic memory allocation is not allowed, or when allocation
//! is a performance bottleneck.

use std::cell::UnsafeCell;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::pin::Pin;
use std::ptr;

use crate::util::Span;
use executorch_sys as et_c;

/// An allocator used to allocate objects for the runtime.
///
/// The allocator does not have a 'free' method, and the memory is deallocated when the allocator is destroyed.
/// Allocated objects are NOT DESTROYED when the allocator is destroyed, therefore most method require the allocated
/// objects to implement the [`NoDrop`] trait.
/// When using [`allocate_raw`](MemoryAllocator::allocate_raw) as a way to allocate bytes and than populate an object in
/// them, the user is responsible dropping the allocated object when it is no longer needed.
pub struct MemoryAllocator<'a>(UnsafeCell<et_c::MemoryAllocator>, PhantomData<&'a ()>);
impl MemoryAllocator<'_> {
    unsafe fn from_inner_ref(allocator: &et_c::MemoryAllocator) -> &Self {
        // Safety: Self has a single field of (UnsafeCell of) et_c::MemoryAllocator
        unsafe { std::mem::transmute(allocator) }
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
    /// A mutable reference to the allocated memory, or [`None`] if allocation failed due to insufficient memory or
    /// an alignment that is not a power of 2.
    pub fn allocate_raw(&self, size: usize, alignment: usize) -> Option<&mut [u8]> {
        let ptr =
            unsafe { et_c::executorch_MemoryAllocator_allocate(self.0.get(), size, alignment) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) })
        }
    }

    /// Allocates memory for a type `T` with uninitialized memory.
    ///
    /// Once a valid object is written to the allocated memory by the user, its also the responsibility of user to
    /// drop the initialized value.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated memory, or [`None`] if allocation failed due to insufficient memory.
    pub fn allocate_uninit<T>(&self) -> Option<&mut MaybeUninit<T>> {
        let ptr = self.allocate_raw(std::mem::size_of::<T>(), std::mem::align_of::<T>())?;
        let ptr = ptr.as_mut_ptr() as *mut MaybeUninit<T>;
        Some(unsafe { &mut *ptr })
    }

    /// Allocates memory for a type `T` and initializes it with `Default::default()`.
    ///
    /// The method require the type `T` to implement the [`NoDrop`] trait because the allocator does not call the
    /// destructor of objects allocated by it when the allocator is dropped.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated memory, or [`None`] if allocation failed due to insufficient memory.
    pub fn allocate<T>(&self) -> Option<&mut T>
    where
        T: NoDrop + Default,
    {
        let val = self.allocate_uninit::<T>()?;
        val.write(Default::default());
        Some(unsafe { val.assume_init_mut() })
    }

    /// Allocates a pinned memory for a type `T` and initializes it with `Default::default()`.
    ///
    /// The method require the type `T` to implement the [`NoDrop`] trait because the allocator does not call the
    /// destructor of objects allocated by it when the allocator is dropped.
    ///
    /// # Returns
    ///
    /// A pinned mutable reference to the allocated memory, or [`None`] if allocation failed due to insufficient memory.
    pub fn allocate_pinned<T>(&self) -> Option<Pin<&mut T>>
    where
        T: NoDrop + Default,
    {
        let val = self.allocate::<T>()?;
        // Safety: the value was just allocated in a fixed address and was not moved
        Some(unsafe { Pin::new_unchecked(val) })
    }

    /// Allocates memory for an array of type `T` and initializes each element with `Default::default()`.
    ///
    /// The method require the type `T` to implement the [`NoDrop`] trait because the allocator does not call the
    /// destructor of objects allocated by it when the allocator is dropped.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated array, or [`None`] if allocation failed due to insufficient memory.
    pub fn allocate_arr<T>(&self, len: usize) -> Option<&mut [T]>
    where
        T: NoDrop + Default,
    {
        self.allocate_arr_fn(len, |_| Default::default())
    }

    /// Allocates memory for an array of type `T` and initializes each element with the result of the closure `f`.
    ///
    /// The method require the type `T` to implement the [`NoDrop`] trait because the allocator does not call the
    /// destructor of objects allocated by it when the allocator is dropped.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of elements in the array.
    /// * `f` - The closure that initializes each element.
    ///
    /// # Returns
    ///
    /// A mutable reference to the allocated array, or [`None`] if allocation failed due to insufficient memory.
    pub fn allocate_arr_fn<T>(&self, len: usize, f: impl Fn(usize) -> T) -> Option<&mut [T]>
    where
        T: NoDrop,
    {
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

/// A marker trait indicating it's OK to not drop an instance of that type.
///
/// The [`MemoryAllocator`] allocates memory for objects and return a pointer or reference to the allocated memory.
/// While doing so, it does NOT maintain the type or destructors of allocated objects, and does not call their
/// destructor (if exit). This trait marks types that can safely be not dropped.
///
/// # Safety
/// When implementing this trait for a type, the user must ensure the type does not implement the [`Drop`] trait.
pub unsafe trait NoDrop {}
// Safety: any type that implement Copy can not implement Drop, enforced by the compiler
unsafe impl<T: Copy> NoDrop for T {}
// unsafe impl<T: Drop> NoDrop for MaybeUninit<T> {}
// unsafe impl<T: Drop> NoDrop for ManuallyDrop<T> {}
// Safety: Storage doesn't implement drop
unsafe impl<T: Storable> NoDrop for Storage<T> {}
// Safety: Span doesn't implement drop
unsafe impl<T: crate::util::SpanElement> NoDrop for crate::util::Span<'_, T> {}

/// A class that does simple allocation based on a size and returns the pointer
/// to the memory address. It bookmarks a buffer with certain size. The
/// allocation is simply checking space and growing the cur_ pointer with each
/// allocation request.
pub struct BufferMemoryAllocator<'a>(
    pub(crate) UnsafeCell<et_c::MemoryAllocator>,
    PhantomData<&'a ()>,
);
impl<'a> BufferMemoryAllocator<'a> {
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
        let size = buffer.len().try_into().unwrap();
        let base_addr = buffer.as_mut_ptr();
        let allocator = unsafe { et_c::executorch_MemoryAllocator_new(size, base_addr) };
        Self(UnsafeCell::new(allocator), PhantomData)
    }
}
impl<'a> AsRef<MemoryAllocator<'a>> for BufferMemoryAllocator<'a> {
    fn as_ref(&self) -> &MemoryAllocator<'a> {
        unsafe { MemoryAllocator::from_inner_ref(&*self.0.get()) }
    }
}
impl<'a> Deref for BufferMemoryAllocator<'a> {
    type Target = MemoryAllocator<'a>;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
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
    pub struct MallocMemoryAllocator(
        UnsafeCell<et_c::cxx::UniquePtr<et_c::cpp::MallocMemoryAllocator>>,
    );
    impl Default for MallocMemoryAllocator {
        fn default() -> Self {
            Self::new()
        }
    }
    impl MallocMemoryAllocator {
        /// Construct a new Malloc memory allocator.
        pub fn new() -> Self {
            Self(UnsafeCell::new(et_c::cpp::MallocMemoryAllocator_new()))
        }
    }
    impl AsRef<MemoryAllocator<'static>> for MallocMemoryAllocator {
        fn as_ref(&self) -> &MemoryAllocator<'static> {
            // Safety: MallocMemoryAllocator contains a single field of (UnsafeCell of) et_c::MemoryAllocator which is a
            // sub class of et_c::MemoryAllocator, and MemoryAllocator contains a single field of (UnsafeCell of)
            // et_c::MemoryAllocator.
            // The returned allocator have a lifetime of 'static because it does not depend on any external buffer, malloc
            // objects are alive until the program ends.
            let self_ = unsafe { &mut *self.0.get() }.as_mut().unwrap();
            let allocator = unsafe { et_c::cpp::MallocMemoryAllocator_as_memory_allocator(self_) };
            unsafe { MemoryAllocator::from_inner_ref(&*allocator) }
        }
    }
    impl Deref for MallocMemoryAllocator {
        type Target = MemoryAllocator<'static>;
        fn deref(&self) -> &Self::Target {
            self.as_ref()
        }
    }
}

/// A group of buffers that can be used to represent a device's memory hierarchy.
pub struct HierarchicalAllocator<'a>(et_c::HierarchicalAllocator, PhantomData<&'a ()>);
impl<'a> HierarchicalAllocator<'a> {
    /// Constructs a new HierarchicalAllocator.
    ///
    /// # Arguments
    ///
    /// * `buffers` - The buffers to use for memory allocation.
    ///     `buffers.size()` must be >= `MethodMeta::num_non_const_buffers()`.
    ///     `buffers[N].size()` must be >= `MethodMeta::non_const_buffer_size(N)`.
    pub fn new(buffers: &'a mut [Span<'a, u8>]) -> Self {
        // Safety: safe because the memory layout of [Span<u8>] and [et_c::SpanU8] is the same.
        let buffers: &'a mut [et_c::SpanU8] = unsafe { std::mem::transmute(buffers) };
        let buffers = et_c::SpanSpanU8 {
            data: buffers.as_mut_ptr(),
            len: buffers.len(),
        };
        Self(
            unsafe { et_c::executorch_HierarchicalAllocator_new(buffers) },
            PhantomData,
        )
    }
}
impl Drop for HierarchicalAllocator<'_> {
    fn drop(&mut self) {
        unsafe { et_c::executorch_HierarchicalAllocator_destructor(&mut self.0) };
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
    ///     Must outlive the Method that uses it.
    /// * `planned_memory` - The memory-planned buffers to use for mutable tensor data when executing a Method.
    ///     Must outlive the Method that uses it. May be [`None`] if the Method does not use any memory-planned tensor data.
    ///     The sizes of the buffers in this HierarchicalAllocator must agree with the corresponding
    ///     `MethodMeta::num_memory_planned_buffers()` and `MethodMeta::memory_planned_buffer_size(N)` values,
    ///     which are embedded in the Program.
    /// * `temp_allocator` - The allocator to use when allocating temporary data during kernel or delegate execution.
    ///     Must outlive the Method that uses it. May be [`None`] if the Method does not use kernels or delegates that
    ///     allocate temporary data. This allocator will be reset after every kernel or delegate call during execution.
    pub fn new(
        method_allocator: &'a MemoryAllocator<'a>,
        planned_memory: Option<&'a mut HierarchicalAllocator>,
        temp_allocator: Option<&'a MemoryAllocator<'a>>,
    ) -> Self {
        let planned_memory = planned_memory
            .map(|x| &mut x.0 as *mut _)
            .unwrap_or(ptr::null_mut());
        let temp_allocator = temp_allocator.map(|x| x.0.get()).unwrap_or(ptr::null_mut());
        Self(
            UnsafeCell::new(unsafe {
                et_c::executorch_MemoryManager_new(
                    method_allocator.0.get(),
                    planned_memory,
                    temp_allocator,
                )
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
///     let storage = executorch::storage!(Tensor<f32>);
///     // macro expands to:
///     // let storage = pin::pin!(Storage::<Tensor<f32>>::default());
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
pub struct Storage<T: Storable>(MaybeUninit<T::__Storage>, PhantomPinned);
impl<T: Storable> Default for Storage<T> {
    /// Create a new [`Storage`] object with an uninitialized inner value.
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Storable> Storage<T> {
    pub(crate) const fn new() -> Self {
        Self(MaybeUninit::uninit(), PhantomPinned)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T::__Storage {
        self.0.as_mut_ptr()
    }
}

/// A macro to create a pinned [`Storage`] object(s).
///
/// Some types in the library required dedicated memory management, either to avoid heap allocations, to prevent moving
/// the object, etc.
/// The [`Storage`] struct helper function used to allocate memory for these types, see its description for more
/// information.
/// The macro is a convenient way to create a pinned [`Storage`] object(s):
/// - `storage!(T)` creates a single storage for type `T`, `Pin<&mut Storage<T>>`.
/// - `storage!(T, [N])` creates an array on the stack, `Pin<&mut [Storage<T>, N]>`.
/// - `storage!(T, (N))` creates a vector in the heap, `Pin<Box<[Storage<T>]>>`. Usually converted to
///     `Pin<&mut [Storage<T>, N]>` with [`as_mut()`](Pin::as_mut).
///
/// ```rust,ignore
/// let tensor_impl: TensorImpl = ...;
/// let storage: Pin<&mut Storage<Tensor<f32>>> = executorch::storage!(Tensor<f32>);
/// let tensor = Tensor::new_in_storage(&tensor_impl, storage);
///
/// let (evalue1, evalue2, evalue3) = (EValue::new(42), EValue::new(17), EValue::new(6));
/// let wrapped_vals = EValuePtrList::new([&evalue1, &evalue2, &evalue3]);
/// let mut unwrapped_vals: Pin<&mut [Storage<i64>, 3]> = storage!(i64, [3]); // an array allocation on the stack
/// let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();
///
/// let evalues = vec![EValue::new(42), EValue::new(17), EValue::new(6)];
/// let wrapped_vals = EValuePtrList::new(evalues.iter());
/// let mut unwrapped_vals: Pin<Box<[Storage<i64>]>> = storage!(i64, (evalues.len())); // a vector allocation on the heap
/// let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals.as_mut()).unwrap(); // .as_mut()
/// ```
#[macro_export]
macro_rules! storage {
    ($t:ty) => {
        core::pin::pin!($crate::memory::Storage::<$t>::default())
    };
    ($t:ty, [$n:expr]) => {
        core::pin::pin!([0; $n].map(|_| $crate::memory::Storage::<$t>::default()))
    };
    ($t:ty, ($n:expr)) => {{
        let n = $n;
        let mut vec = $crate::__private::alloc::Vec::with_capacity(n);
        vec.resize_with(n, || $crate::memory::Storage::<$t>::default());
        std::pin::Pin::from(vec.into_boxed_slice())
    }};
}

/// A marker trait for types that can be stored in a [`Storage`].
///
/// Usually the type is a Cpp object that is not trivially movable. See the [`Storage`] struct for more information.
pub trait Storable {
    /// The underlying Cpp object type, defining the memory layout of a [`Storage`] object.
    #[doc(hidden)]
    type __Storage;
}

macro_rules! impl_default_storable {
    ($($t:ty),*) => {
        $(
            impl Storable for $t {
                type __Storage = $t;
            }
        )*
    };
}
impl_default_storable!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

#[cfg(test)]
mod tests {
    use core::ops::Deref;

    use super::*;

    #[test]
    fn buffer_memory_allocator() {
        let mut buffer: [u8; 16384] = [0; 16384];
        let allocator = BufferMemoryAllocator::new(&mut buffer);
        let allocator_init = |size| {
            let buffer = allocator.allocate_raw(size, 1).unwrap();
            BufferMemoryAllocator::new(buffer)
        };
        test_memory_allocator(allocator_init, true);
    }

    #[cfg(feature = "std")]
    #[test]
    fn malloc_memory_allocator() {
        let mut idx = 0;
        let allocator_init = |_size| {
            idx += 1;
            if idx % 2 == 0 {
                MallocMemoryAllocator::default()
            } else {
                MallocMemoryAllocator::new()
            }
        };
        test_memory_allocator(allocator_init, false);
    }

    fn test_memory_allocator<'a, T>(mut allocator_init: impl FnMut(usize) -> T, is_bounded: bool)
    where
        T: Deref<Target = super::MemoryAllocator<'a>>,
    {
        let sizes = [1, 2, 4, 5, 8, 13, 31];
        let alignments = [1, 2, 4, 8, 16, 32];
        let allocations = sizes
            .into_iter()
            .flat_map(|size| alignments.map(|alignment| (size, alignment)));

        let raw_allocations_size = allocations.clone().map(|(size, _)| size).sum::<usize>() * 2;
        let allocator = allocator_init(raw_allocations_size);
        for (size, alignment) in allocations.clone() {
            let allocation = allocator.allocate_raw(size, alignment).unwrap_or_else(|| {
                panic!(
                    "Failed to allocate {} bytes with alignment {}",
                    size, alignment
                )
            });
            assert_eq!(allocation.len(), size);
            assert_eq!(allocation.as_ptr() as usize % alignment, 0);
        }
        if is_bounded {
            let allocator = allocator_init(0);
            assert!(allocator.allocate_raw(5, 8).is_none());
        }

        let allocator = allocator_init(1024);
        assert!(allocator.allocate::<[u8; 1]>().is_some());
        assert!(allocator.allocate::<[u8; 2]>().is_some());
        assert!(allocator.allocate::<[u8; 4]>().is_some());
        assert!(allocator.allocate::<[u8; 8]>().is_some());
        assert!(allocator.allocate::<[f64; 15]>().is_some());

        let allocator = allocator_init(1024);
        assert!(allocator.allocate_pinned::<[u8; 1]>().is_some());
        assert!(allocator.allocate_pinned::<[u8; 2]>().is_some());
        assert!(allocator.allocate_pinned::<[u8; 4]>().is_some());
        assert!(allocator.allocate_pinned::<[u8; 8]>().is_some());
        assert!(allocator.allocate_pinned::<[f64; 15]>().is_some());

        let allocator = allocator_init(4096);
        for sizes in sizes {
            let arr = allocator.allocate_arr::<u8>(sizes).unwrap();
            assert_eq!(arr.len(), sizes);
            assert!(arr.iter().all(|&x| x == 0));
            let arr = allocator.allocate_arr::<f32>(sizes).unwrap();
            assert_eq!(arr.len(), sizes);
            assert!(arr.iter().all(|&x| x == 0.0));
        }

        let allocator = allocator_init(4096);
        for sizes in sizes {
            let arr = allocator.allocate_arr_fn(sizes, |i| i as u8).unwrap();
            assert_eq!(arr.len(), sizes);
            assert!(arr.iter().enumerate().all(|(i, &x)| x == i as u8));
            let arr = allocator.allocate_arr_fn(sizes, |i| i as f32).unwrap();
            assert_eq!(arr.len(), sizes);
            assert!(arr.iter().enumerate().all(|(i, &x)| x == i as f32));
        }
    }

    #[test]
    fn storage_macro() {
        let _: std::pin::Pin<&mut super::Storage<i32>> = storage!(i32);

        let s: std::pin::Pin<&mut [super::Storage<i32>; 3]> = storage!(i32, [3]);
        assert_eq!(s.len(), 3);

        #[cfg(feature = "std")]
        {
            let dynamic_size = 2 + std::env::var("unknown-at-compile-time").is_ok() as usize;
            let s: std::pin::Pin<crate::alloc::Box<[super::Storage<i32>]>> =
                storage!(i32, (dynamic_size));
            assert_eq!(s.len(), dynamic_size);
        }
    }
}
