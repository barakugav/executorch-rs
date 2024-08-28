//! Utility functions and types.
//!
//! Most of the structs in this module may seems redundant in Rust, but they are wrappers around C++ types
//! that are used in the C++ API. Some structs and functions accept these types as arguments, so they are
//! necessary to interact with the C++ API.

use std::fmt::Debug;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::MaybeUninit;
use std::pin::Pin;

#[cfg(feature = "alloc")]
use crate::et_alloc;
use crate::et_c;

pub(crate) trait Destroy {
    /// Destroy the object without deallocating its memory.
    ///
    /// # Safety
    ///
    /// This function can not be called twice, and the object must not be used after it is destroyed.
    unsafe fn destroy(&mut self);
}

pub(crate) enum NonTriviallyMovable<'a, T: Destroy> {
    // We own the value. We allocated its memory, called its constructor and we are responsible for calling its
    // destructor and deallocating its memory.
    // We can mutate T freely.
    // Created via `new_boxed`.
    #[cfg(feature = "alloc")]
    Boxed(Pin<et_alloc::Box<(T, PhantomPinned)>>),
    // We own the reference. We called its constructor and we are responsible for calling its destructor.
    // We did not allocate its memory and we are not responsible for deallocating it.
    // We can mutate T freely.
    // Created via `new_in_storage`.
    OwnedRef(Pin<&'a mut (T, PhantomPinned)>),
    // We don't own the reference. We did not called its constructor and we are not responsible for calling its
    // destructor.
    // We did not allocate its memory and we are not responsible for deallocating it.
    // We can not mutate T.
    // Created via `from_ref`.
    Ref(Pin<&'a (T, PhantomPinned)>),
    // We don't own the reference. We did not called its constructor and we are not responsible for calling its
    // destructor, but we are allowed to mutate it.
    // We did not allocate its memory and we are not responsible for deallocating it.
    // We can mutate T freely.
    // Created via `from_mut_ref`.
    RefMut(Pin<&'a mut (T, PhantomPinned)>),
}
impl<'a, T: Destroy> NonTriviallyMovable<'a, T> {
    /// Create a new [`NonTriviallyMovable`] object with an inner value in a box.
    ///
    /// # Safety
    ///
    /// The inner value must be initialized by the given closure.
    #[cfg(feature = "alloc")]
    pub(crate) unsafe fn new_boxed(init: impl FnOnce(*mut T)) -> Self {
        let mut p = et_alloc::Box::pin(MaybeUninit::<T>::uninit());
        // Safety: we get a mut ref out of the pin, but we dont move out of it
        init(unsafe { p.as_mut().get_unchecked_mut().as_mut_ptr() });
        // Safety: MaybeUninit<T> and (T, PhantomPinned) have the same memory layout, and the `init` closure should have
        // initialized the value.
        let p = unsafe {
            std::mem::transmute::<
                Pin<et_alloc::Box<MaybeUninit<T>>>,
                Pin<et_alloc::Box<(T, PhantomPinned)>>,
            >(p)
        };
        NonTriviallyMovable::Boxed(p)
    }

    /// Create a new [`NonTriviallyMovable`] object with an inner value in a [`Storage`].
    ///
    /// # Safety
    ///
    /// The inner value must be initialized by the given closure.
    pub(crate) unsafe fn new_in_storage<S>(
        init: impl FnOnce(*mut T),
        storage: Pin<&'a mut Storage<S>>,
    ) -> Self
    where
        S: Storable<Storage = T>,
    {
        // Safety: we get a mut ref out of the pin, but we dont move out of it
        let storage = unsafe { storage.get_unchecked_mut() };
        init(storage.as_mut_ptr());
        // Safety: the `init` function should have initialized the value.
        let p = unsafe { &mut *storage.as_mut_ptr() };
        // Safety: T and (T, PhantomPinned) have the same memory layout.
        let p = unsafe { std::mem::transmute::<&'a mut T, &'a mut (T, PhantomPinned)>(p) };
        // Safety: p is a reference to a valid memory location and it is not moved for at least 'a.
        let p = unsafe { Pin::new_unchecked(p) };

        Self::OwnedRef(p)
    }

    pub(crate) fn from_ref(p: &'a T) -> Self {
        // Safety: T and (T, PhantomPinned) have the same memory layout.
        let p = unsafe { std::mem::transmute::<&'a T, &'a (T, PhantomPinned)>(p) };
        // Safety: p is a reference to a valid memory location and it is not moved for at least 'a.
        let p = unsafe { Pin::new_unchecked(p) };

        Self::Ref(p)
    }

    pub(crate) fn from_mut_ref(p: &'a mut T) -> Self {
        // Safety: T and (T, PhantomPinned) have the same memory layout.
        let p = unsafe { std::mem::transmute::<&'a mut T, &'a mut (T, PhantomPinned)>(p) };
        // Safety: p is a reference to a valid memory location and it is not moved for at least 'a.
        let p = unsafe { Pin::new_unchecked(p) };

        Self::RefMut(p)
    }
}
impl<'a, T: Destroy> Drop for NonTriviallyMovable<'a, T> {
    fn drop(&mut self) {
        let shoud_destroy = match self {
            #[cfg(feature = "alloc")]
            NonTriviallyMovable::Boxed(_) => true,
            NonTriviallyMovable::OwnedRef(_) => true,
            NonTriviallyMovable::Ref(_) | NonTriviallyMovable::RefMut(_) => false,
        };

        if shoud_destroy {
            // Safety: we dont move (or swap) the inner value.
            let p = unsafe { self.as_mut() }.unwrap();
            // Safety: we call the destroy function only once and the object is not used after it is destroyed.
            unsafe { p.destroy() };
        }
    }
}
impl<'a, T: Destroy> AsRef<T> for NonTriviallyMovable<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            #[cfg(feature = "alloc")]
            NonTriviallyMovable::Boxed(p) => &p.0,
            NonTriviallyMovable::OwnedRef(p) => &p.0,
            NonTriviallyMovable::Ref(p) => &p.0,
            NonTriviallyMovable::RefMut(p) => &p.0,
        }
    }
}
impl<'a, T: Destroy> NonTriviallyMovable<'a, T> {
    /// Get a mutable reference to the inner value if we own it (we constructed it).
    ///
    /// # Safety
    ///
    /// The caller can not move out of the returned mut reference.
    pub(crate) unsafe fn as_mut(&mut self) -> Option<&mut T> {
        match self {
            #[cfg(feature = "alloc")]
            NonTriviallyMovable::Boxed(p) => Some(&mut unsafe { p.as_mut().get_unchecked_mut() }.0),
            NonTriviallyMovable::OwnedRef(p) => Some(&mut p.as_mut().get_unchecked_mut().0),
            NonTriviallyMovable::Ref(_) => None,
            NonTriviallyMovable::RefMut(p) => Some(&mut p.as_mut().get_unchecked_mut().0),
        }
    }
}

#[cfg(feature = "alloc")]
#[allow(dead_code)]
pub(crate) struct NonTriviallyMovableVec<T: Destroy>(Pin<et_alloc::Box<(PhantomPinned, [T])>>);
#[cfg(feature = "alloc")]
#[allow(dead_code)]
impl<T: Destroy> NonTriviallyMovableVec<T> {
    pub(crate) unsafe fn new(len: usize, init: impl Fn(usize, &mut MaybeUninit<T>)) -> Self {
        let vec = (0..len)
            .map(|_| MaybeUninit::<T>::uninit())
            .collect::<et_alloc::Vec<_>>()
            .into_boxed_slice();
        let mut vec = Pin::new_unchecked(vec);
        // Safety: we dont move out of the vec
        for (i, elem) in vec.as_mut().get_unchecked_mut().iter_mut().enumerate() {
            init(i, elem);
        }
        // Safety: [MaybeUninit<T>] and (PhantomPinned, [T]) have the same memory layout.
        let vec = unsafe {
            std::mem::transmute::<
                Pin<et_alloc::Box<[MaybeUninit<T>]>>,
                Pin<et_alloc::Box<(PhantomPinned, [T])>>,
            >(vec)
        };
        NonTriviallyMovableVec(vec)
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.0 .1
    }

    // pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
    //     &mut self.0
    // }
}
#[cfg(feature = "alloc")]
impl<T: Destroy> Drop for NonTriviallyMovableVec<T> {
    fn drop(&mut self) {
        // Safety: we dont move out of the pinned value
        for elem in unsafe { &mut self.0.as_mut().get_unchecked_mut().1 } {
            // Safety: we call the destroy function only once and the object is not used after it is destroyed
            unsafe { elem.destroy() };
        }
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
/// and it is intended to be pinned in memory and later be used as to allocate the Cpp object in place. This way, the
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
///     pinned in memory, possible on the stack (see later example). The destructor of the Cpp object is called when
///     Rust object is dropped, but the [`Storage`] is not deallocated.
/// - Does not own the memory, does not own the object: the pointer points to a Cpp object that is owned by another
///     entity, like a regular Rust reference. The destructor of the Cpp object is not called when the Rust object is
///     dropped and no deallocation is done.
///
/// Non-trivially movable objects such as [`Tensor`](crate::tensor::Tensor) and [`EValue`](crate::evalue::EValue) expose
/// a straight forward `new` function that allocate the object on the heap which should be used in most cases.
/// The `new` function is available when the `alloc` feature is enabled. When allocations are not available, the
/// [`Storage`] struct should be created and pinned, which expose an identical `new` function that allocates the object
/// in place:
/// ```rust,ignore
/// let tensor_impl: TensorImpl = ...;
///
/// // Create a Tensor and an EValue on the heap
/// let tensor = Tensor::new(&tensor_impl);
/// let evalue = EValue::new(tensor); // allocate on the heap
///
/// // Create a Tensor and an EValue on the stack
/// let storage1: Pin<&mut Storage<Tensor<f32>>> = executorch::storage!(Tensor<f32>);
/// let storage2: Pin<&mut Storage<EValue>> = executorch::storage!(EValue);
/// let tensor: Tensor = storage1.new(&tensor_impl);
/// let evalue: EValue = storage2.new(tensor);
/// ```
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

/// Create and pin a [`Storage`] object.
///
/// See the [`Storage`] struct for more information.
#[macro_export]
macro_rules! storage {
    ($type:path) => {
        core::pin::pin!(executorch::util::Storage::<$type>::default())
    };
}

pub(crate) trait IntoRust {
    type RsType;
    fn rs(self) -> Self::RsType;
}

/// Represents a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// Span and ArrayRef are extremely similar with the difference being ArrayRef
/// views a list of constant elements and Span views a list of mutable elements.
/// Clients should decide between the two based on if the list elements for their
/// use case should be mutable.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
#[allow(dead_code)]
pub struct ArrayRef<'a, T>(pub(crate) et_c::ArrayRef<T>, PhantomData<&'a ()>);
impl<'a, T> ArrayRef<'a, T> {
    // pub(crate) unsafe fn from_inner(arr: &et_c::ArrayRef<T>) -> Self {
    //     Self(
    //         et_c::ArrayRef::<T> {
    //             Data: arr.Data,
    //             Length: arr.Length,
    //             _phantom_0: PhantomData,
    //         },
    //         PhantomData,
    //     )
    // }

    /// Create an ArrayRef from a slice.
    ///
    /// The given slice must outlive the ArrayRef.
    pub fn from_slice(s: &'a [T]) -> Self {
        Self(
            et_c::ArrayRef {
                Data: s.as_ptr(),
                Length: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.0.Data, self.0.Length) }
    }
}
impl<T: Debug> Debug for ArrayRef<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// Represent a reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the Span.
///
/// Span and ArrayRef are extremely similar with the difference being ArrayRef
/// views a list of constant elements and Span views a list of mutable elements.
/// Clients should decide between the two based on if the list elements for their
/// use case should be mutable.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
#[allow(dead_code)]
pub struct Span<'a, T>(pub(crate) et_c::Span<T>, PhantomData<&'a T>);
impl<'a, T> Span<'a, T> {
    pub(crate) unsafe fn new(span: et_c::Span<T>) -> Self {
        Self(span, PhantomData)
    }

    /// Create a Span from a mutable slice.
    ///
    /// The given slice must outlive the Span.
    pub fn from_slice(s: &'a mut [T]) -> Self {
        Self(
            et_c::Span {
                data_: s.as_mut_ptr(),
                length_: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.data_, self.0.length_) }
    }
}
impl<T: Debug> Debug for Span<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

// /// Leaner optional class, subset of c10, std, and boost optional APIs.
// pub struct Optional<T>(et_c::optional<T>);
// impl<T> Optional<T> {
//     /// Create a new Optional based on the given Option.
//     pub fn new(val: Option<T>) -> Self {
//         let is_some = val.is_some();
//         Self(et_c::optional::<T> {
//             trivial_init: et_c::optional_trivial_init_t { _address: 0 },
//             storage_: val
//                 .map(|value| et_c::optional_storage_t {
//                     value_: ManuallyDrop::new(value),
//                 })
//                 .unwrap_or(et_c::optional_storage_t {
//                     dummy_: ManuallyDrop::new(0),
//                 }),
//             init_: is_some,
//             _phantom_0: PhantomData,
//         })
//     }

//     /// Get an optional reference to the value.
//     pub fn as_ref(&self) -> Option<&T> {
//         self.0.init_.then(|| unsafe { &*self.0.storage_.value_ })
//     }

//     /// Convert this Optional into an Option.
//     pub fn into_option(mut self) -> Option<T> {
//         self.0.init_.then(|| {
//             self.0.init_ = false;
//             unsafe { ManuallyDrop::take(&mut self.0.storage_.value_) }
//         })
//     }
// }
// impl<T> Drop for Optional<T> {
//     fn drop(&mut self) {
//         if self.0.init_ {
//             unsafe {
//                 ManuallyDrop::drop(&mut self.0.storage_.value_);
//             }
//         }
//     }
// }
// impl<T> From<Optional<T>> for Option<T> {
//     fn from(opt: Optional<T>) -> Option<T> {
//         opt.into_option()
//     }
// }
// impl<T> From<Option<T>> for Optional<T> {
//     fn from(opt: Option<T>) -> Optional<T> {
//         Optional::new(opt)
//     }
// }
// impl<T: Clone> Clone for Optional<T> {
//     fn clone(&self) -> Self {
//         Self::new(self.as_ref().cloned())
//     }
// }
// impl<T: PartialEq> PartialEq for Optional<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.as_ref() == other.as_ref()
//     }
// }
// impl<T: Eq> Eq for Optional<T> {}
// impl<T: Hash> Hash for Optional<T> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.as_ref().hash(state)
//     }
// }
// impl<T: Debug> Debug for Optional<T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         self.as_ref().fmt(f)
//     }
// }

#[allow(dead_code)]
pub(crate) fn str2chars(s: &str) -> Result<&[std::ffi::c_char], &'static str> {
    let bytes = s.as_bytes();
    if bytes.iter().any(|&b| b == 0) {
        return Err("String contains null byte");
    }
    let chars = bytes.as_ptr().cast::<std::ffi::c_char>();
    Ok(unsafe { std::slice::from_raw_parts(chars, bytes.len()) })
}
#[allow(dead_code)]
#[cfg(feature = "std")]
pub(crate) fn chars2string(chars: Vec<std::ffi::c_char>) -> String {
    let bytes = unsafe { std::mem::transmute::<Vec<std::ffi::c_char>, Vec<u8>>(chars) };
    String::from_utf8(bytes).unwrap()
}

#[cfg(feature = "std")]
#[allow(dead_code)]
pub(crate) mod cpp_vec {
    use super::IntoRust;
    use crate::et_rs_c;

    pub(crate) fn vec_as_slice<T>(vec: &et_rs_c::Vec<T>) -> &[T] {
        unsafe { std::slice::from_raw_parts(vec.data, vec.len) }
    }

    pub(crate) fn vec_as_mut_slice<T>(vec: &mut et_rs_c::Vec<T>) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(vec.data, vec.len) }
    }

    pub(crate) struct CppVec<T: CppVecElm>(et_rs_c::Vec<T>);
    impl<T: CppVecElm> CppVec<T> {
        pub fn as_slice(&self) -> &[T] {
            vec_as_slice(&self.0)
        }

        pub fn to_vec(&self) -> Vec<T>
        where
            T: Clone,
        {
            self.as_slice().to_vec()
        }
    }
    impl<T: CppVecElm> IntoRust for et_rs_c::Vec<T> {
        type RsType = CppVec<T>;
        fn rs(self) -> Self::RsType {
            CppVec(self)
        }
    }
    impl IntoRust for et_rs_c::Vec<et_rs_c::Vec<std::ffi::c_char>> {
        type RsType = CppVec<CppVec<std::ffi::c_char>>;
        fn rs(self) -> Self::RsType {
            // Safety: et_rs_c::Vec<T> has the same memory layout as CppVec<T>.
            unsafe {
                std::mem::transmute::<
                    et_rs_c::Vec<et_rs_c::Vec<std::ffi::c_char>>,
                    CppVec<CppVec<std::ffi::c_char>>,
                >(self)
            }
        }
    }
    impl<T: CppVecElm> Drop for CppVec<T> {
        fn drop(&mut self) {
            T::drop_vec(self);
        }
    }
    pub(crate) trait CppVecElm: Sized {
        fn drop_vec(vec: &mut CppVec<Self>);
    }
    impl CppVecElm for std::ffi::c_char {
        fn drop_vec(vec: &mut CppVec<Self>) {
            unsafe { et_rs_c::Vec_char_destructor(&mut vec.0) }
        }
    }
    impl CppVecElm for CppVec<std::ffi::c_char> {
        fn drop_vec(vec: &mut CppVec<Self>) {
            // Safety: CppVec<T> has the same memory layout as et_rs_c::Vec<T>.
            let vec = unsafe {
                std::mem::transmute::<
                    &mut CppVec<CppVec<std::ffi::c_char>>,
                    &mut et_rs_c::Vec<et_rs_c::Vec<std::ffi::c_char>>,
                >(vec)
            };
            unsafe { et_rs_c::Vec_Vec_char_destructor(vec) }
        }
    }
    // impl<'a> CppVecElm for EValue<'a> {
    //     fn drop_vec(vec: &mut CppVec<Self>) {
    //         let vec = unsafe {
    //             std::mem::transmute::<&mut et_rs_c::Vec<EValue<'a>>, &mut et_rs_c::Vec<et_c::EValue>>(
    //                 &mut vec.0,
    //             )
    //         };
    //         unsafe { et_rs_c::Vec_EValue_destructor(vec) }
    //     }
    // }
}

// Debug func
#[allow(dead_code)]
#[cfg(feature = "std")]
pub(crate) fn to_bytes<T>(val: &T) -> Vec<u8> {
    (0..std::mem::size_of_val(val))
        .map(|i| unsafe {
            let ptr = val as *const T as *const u8;
            *ptr.add(i)
        })
        .collect()
}

/// A marker trait for dimensions that have a fixed size.
///
/// This trait is useful for functions that avoid allocations and want to define additional arrays with the same size as
/// a given dimension.
pub trait FixedSizeDim: ndarray::Dimension {
    /// An array with the same fixed size as the dimension.
    type Arr<T: Clone + Copy + Default>: DimArr<T>;
    private_decl! {}
}
macro_rules! impl_fixed_size_dim {
    ($size:expr) => {
        impl FixedSizeDim for ndarray::Dim<[ndarray::Ix; $size]> {
            type Arr<T: Clone + Copy + Default> = [T; $size];
            private_impl! {}
        }
    };
}
impl_fixed_size_dim!(0);
impl_fixed_size_dim!(1);
impl_fixed_size_dim!(2);
impl_fixed_size_dim!(3);
impl_fixed_size_dim!(4);
impl_fixed_size_dim!(5);
impl_fixed_size_dim!(6);

/// An abstraction over fixed-size arrays and regular vectors if the `alloc` feature is enabled.
pub trait DimArr<T>: AsRef<[T]> + AsMut<[T]> {
    /// Create an array of zeros with the given number of dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the given number of dimensions is not supported by the array. For example, if the array is a fixed
    /// size array of size 3, it will panic if the given number of dimensions is 4. Regular vectors will never panic.
    fn zeros(ndim: usize) -> Self;
}

macro_rules! impl_dim_arr {
    ($size:expr) => {
        impl<T: Clone + Copy + Default> DimArr<T> for [T; $size] {
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, $size, "Invalid dimension size");
                [T::default(); $size]
            }
        }
    };
}
impl_dim_arr!(0);
impl_dim_arr!(1);
impl_dim_arr!(2);
impl_dim_arr!(3);
impl_dim_arr!(4);
impl_dim_arr!(5);
impl_dim_arr!(6);

#[cfg(feature = "alloc")]
impl<T: Clone + Copy + Default> DimArr<T> for et_alloc::Vec<T> {
    fn zeros(ndim: usize) -> Self {
        et_alloc::Vec::from_iter(std::iter::repeat(T::default()).take(ndim))
    }
}
