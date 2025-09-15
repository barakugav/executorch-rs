//! Utility functions and types.
//!
//! Most of the structs in this module may seems redundant in Rust, but they are wrappers around C++ types
//! that are used in the C++ API. Some structs and functions accept these types as arguments, so they are
//! necessary to interact with the C++ API.

use et_c::Error as CError;
use std::ffi::CStr;
use std::fmt::Debug;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::MaybeUninit;
use std::pin::Pin;

#[cfg(feature = "alloc")]
use crate::alloc;
use crate::memory::{Storable, Storage};
use executorch_sys as et_c;

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
    Boxed(Pin<alloc::Box<(T, PhantomPinned)>>),
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
    pub(crate) unsafe fn try_new_boxed<E>(
        init: impl FnOnce(*mut T) -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut p = alloc::Box::pin(MaybeUninit::<T>::uninit());
        // Safety: we get a mut ref out of the pin, but we dont move out of it
        init(unsafe { p.as_mut().get_unchecked_mut().as_mut_ptr() })?;
        // Safety: MaybeUninit<T> and (T, PhantomPinned) have the same memory layout, and the `init` closure should have
        // initialized the value.
        let p = unsafe {
            std::mem::transmute::<
                Pin<alloc::Box<MaybeUninit<T>>>,
                Pin<alloc::Box<(T, PhantomPinned)>>,
            >(p)
        };
        Ok(NonTriviallyMovable::Boxed(p))
    }

    /// Create a new [`NonTriviallyMovable`] object with an inner value in a box.
    ///
    /// # Safety
    ///
    /// The inner value must be initialized by the given closure.
    #[cfg(feature = "alloc")]
    pub(crate) unsafe fn new_boxed(init: impl FnOnce(*mut T)) -> Self {
        use core::convert::Infallible;

        let res = Self::try_new_boxed::<Infallible>(|p| {
            init(p);
            Ok(())
        });
        match res {
            Ok(p) => p,
            Err(_) => unsafe { std::hint::unreachable_unchecked() },
        }
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
        S: Storable<__Storage = T>,
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
impl<T: Destroy> Drop for NonTriviallyMovable<'_, T> {
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
impl<T: Destroy> AsRef<T> for NonTriviallyMovable<'_, T> {
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
impl<T: Destroy> NonTriviallyMovable<'_, T> {
    /// Get a mutable reference to the inner value if we own it (we constructed it).
    ///
    /// # Safety
    ///
    /// The caller can not move out of the returned mut reference.
    pub(crate) unsafe fn as_mut(&mut self) -> Option<&mut T> {
        match self {
            #[cfg(feature = "alloc")]
            NonTriviallyMovable::Boxed(p) => Some(&mut unsafe { p.as_mut().get_unchecked_mut() }.0),
            NonTriviallyMovable::OwnedRef(p) => {
                Some(unsafe { &mut p.as_mut().get_unchecked_mut().0 })
            }
            NonTriviallyMovable::Ref(_) => None,
            NonTriviallyMovable::RefMut(p) => {
                Some(unsafe { &mut p.as_mut().get_unchecked_mut().0 })
            }
        }
    }
}

#[cfg(feature = "alloc")]
#[allow(unused)]
pub(crate) struct NonTriviallyMovableVec<T: Destroy>(Pin<alloc::Box<(PhantomPinned, [T])>>);
#[cfg(feature = "alloc")]
#[allow(unused)]
impl<T: Destroy> NonTriviallyMovableVec<T> {
    pub(crate) unsafe fn new(len: usize, init: impl Fn(usize, &mut MaybeUninit<T>)) -> Self {
        let vec = (0..len)
            .map(|_| MaybeUninit::<T>::uninit())
            .collect::<alloc::Vec<_>>()
            .into_boxed_slice();
        let mut vec = unsafe { Pin::new_unchecked(vec) };
        // Safety: we dont move out of the vec
        for (i, elem) in unsafe { vec.as_mut().get_unchecked_mut().iter_mut().enumerate() } {
            init(i, elem);
        }
        // Safety: [MaybeUninit<T>] and (PhantomPinned, [T]) have the same memory layout.
        let vec = unsafe {
            std::mem::transmute::<
                Pin<alloc::Box<[MaybeUninit<T>]>>,
                Pin<alloc::Box<(PhantomPinned, [T])>>,
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

pub(crate) fn try_c_new<T>(f: impl FnOnce(*mut T) -> CError) -> crate::Result<T> {
    let mut value = MaybeUninit::uninit();
    let err = f(value.as_mut_ptr());
    err.rs().map(|_| unsafe { value.assume_init() })
}
pub(crate) fn c_new<T>(f: impl FnOnce(*mut T)) -> T {
    let mut value = MaybeUninit::uninit();
    f(value.as_mut_ptr());
    unsafe { value.assume_init() }
}

pub(crate) trait IntoRust {
    type RsType;
    fn rs(self) -> Self::RsType;
}
pub(crate) trait IntoCpp {
    type CppType;
    fn cpp(self) -> Self::CppType;
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
#[allow(unused)]
pub(crate) struct ArrayRef<'a, T: ArrayRefElement>(
    pub(crate) T::__ArrayRefImpl,
    PhantomData<&'a ()>,
);
impl<'a, T: ArrayRefElement> ArrayRef<'a, T> {
    pub(crate) unsafe fn from_inner(arr: T::__ArrayRefImpl) -> Self {
        Self(arr, PhantomData)
    }

    /// Create an ArrayRef from a slice.
    ///
    /// The given slice must outlive the ArrayRef.
    pub fn from_slice(s: &'a [T]) -> Self {
        Self(unsafe { T::__ArrayRefImpl::from_slice(s) }, PhantomData)
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &'a [T]
    where
        T: 'static,
    {
        unsafe { self.0.as_slice() }
    }
}
impl<T: ArrayRefElement + Debug + 'static> Debug for ArrayRef<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// An element type that can be used in an ArrayRef.
pub(crate) trait ArrayRefElement {
    /// The Cpp type that represents an ArrayRef of this element type.
    #[doc(hidden)]
    type __ArrayRefImpl: __ArrayRefImpl<Element = Self>;
    private_decl! {}
}

/// A Cpp type that represents an ArrayRef of elements of type Element.
#[doc(hidden)]
pub(crate) trait __ArrayRefImpl {
    /// The element type of the ArrayRef.
    type Element: ArrayRefElement<__ArrayRefImpl = Self>;

    /// Create an ArrayRef from a slice.
    ///
    /// # Safety
    ///
    /// The given slice must outlive the ArrayRef.
    unsafe fn from_slice(slice: &[Self::Element]) -> Self;

    /// Get the underlying slice.
    ///
    /// # Safety
    ///
    /// The returned slice must not outlive the input slice, but may outlive the ArrayRef.
    unsafe fn as_slice(&self) -> &'static [Self::Element];

    private_decl! {}
}

macro_rules! impl_array_ref {
    ($element:path, $span:path) => {
        impl ArrayRefElement for $element {
            type __ArrayRefImpl = $span;
            private_impl! {}
        }
        impl __ArrayRefImpl for $span {
            type Element = $element;
            unsafe fn from_slice(slice: &[$element]) -> Self {
                Self {
                    data: slice.as_ptr(),
                    len: slice.len(),
                }
            }
            unsafe fn as_slice(&self) -> &'static [$element] {
                unsafe { std::slice::from_raw_parts(self.data, self.len) }
            }
            private_impl! {}
        }
    };
}
impl_array_ref!(std::ffi::c_char, et_c::ArrayRefChar);
impl_array_ref!(u8, et_c::ArrayRefU8);
impl_array_ref!(i32, et_c::ArrayRefI32);
impl_array_ref!(i64, et_c::ArrayRefI64);
impl_array_ref!(f64, et_c::ArrayRefF64);
impl_array_ref!(usize, et_c::ArrayRefUsizeType);
impl_array_ref!(bool, et_c::ArrayRefBool);
// impl_array_ref!(et_c::Tensor, et_c::ArrayRefTensor);
// impl_array_ref!(et_c::EValue, et_c::ArrayRefEValue);
impl ArrayRefElement for et_c::EValueStorage {
    type __ArrayRefImpl = et_c::ArrayRefEValue;
    private_impl! {}
}
impl __ArrayRefImpl for et_c::ArrayRefEValue {
    type Element = et_c::EValueStorage;
    unsafe fn from_slice(slice: &[et_c::EValueStorage]) -> Self {
        Self {
            data: et_c::EValueRef {
                ptr: slice.as_ptr() as *const _,
            },
            len: slice.len(),
        }
    }
    unsafe fn as_slice(&self) -> &'static [et_c::EValueStorage] {
        let data = self.data.ptr as *const et_c::EValueStorage;
        unsafe { std::slice::from_raw_parts(data, self.len) }
    }
    private_impl! {}
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
#[allow(unused)]
pub struct Span<'a, T: SpanElement>(pub(crate) T::__SpanImpl, PhantomData<&'a T>);
impl<'a, T: SpanElement> Span<'a, T> {
    // pub(crate) unsafe fn new(span: T::SpanImpl) -> Self {
    //     Self(span, PhantomData)
    // }

    /// Create a Span from a mutable slice.
    ///
    /// The given slice must outlive the Span.
    pub fn from_slice(s: &'a mut [T]) -> Self {
        Self(unsafe { T::__SpanImpl::from_slice(s) }, PhantomData)
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &'a mut [T]
    where
        T: 'static,
    {
        unsafe { self.0.as_slice() }
    }
}
impl<T: SpanElement + Debug + 'static> Debug for Span<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// An element type that can be used in a Span.
pub trait SpanElement {
    /// The Cpp type that represents a Span of this element type.
    #[doc(hidden)]
    type __SpanImpl: __SpanImpl<Element = Self>;
    private_decl! {}
}

/// A Cpp type that represents a Span of elements of type Element.
#[doc(hidden)]
pub trait __SpanImpl {
    /// The element type of the Span.
    type Element: SpanElement<__SpanImpl = Self>;

    /// Create a Span from a mutable slice.
    ///
    /// # Safety
    ///
    /// The given slice must outlive the Span.
    unsafe fn from_slice(slice: &mut [Self::Element]) -> Self;

    /// Get the underlying slice.
    ///
    /// # Safety
    ///
    /// The return slice must not outlive the input slice, but may outlive the Span.
    unsafe fn as_slice(&self) -> &'static mut [Self::Element];

    private_decl! {}
}

macro_rules! impl_span {
    ($element:path, $span:path) => {
        impl SpanElement for $element {
            type __SpanImpl = $span;
            private_impl! {}
        }
        impl __SpanImpl for $span {
            type Element = $element;
            unsafe fn from_slice(slice: &mut [$element]) -> Self {
                Self {
                    data: slice.as_mut_ptr(),
                    len: slice.len(),
                }
            }
            unsafe fn as_slice(&self) -> &'static mut [$element] {
                unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
            }
            private_impl! {}
        }
    };
}
impl_span!(u8, et_c::SpanU8);

pub(crate) fn cstr2chars(s: &CStr) -> &[std::ffi::c_char] {
    unsafe { std::slice::from_raw_parts(s.as_ptr(), s.to_bytes().len()) }
}
pub(crate) fn str2chars(s: &str) -> &[std::ffi::c_char] {
    assert_eq!(
        core::alloc::Layout::new::<std::ffi::c_char>(),
        core::alloc::Layout::new::<u8>()
    );
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), s.len()) }
}

pub(crate) fn chars2str(s: &[std::ffi::c_char]) -> Result<&str, std::str::Utf8Error> {
    assert_eq!(
        core::alloc::Layout::new::<std::ffi::c_char>(),
        core::alloc::Layout::new::<u8>()
    );
    let bytes = unsafe { std::mem::transmute::<&[std::ffi::c_char], &[u8]>(s) };
    std::str::from_utf8(bytes)
}

#[cfg(feature = "std")]
pub(crate) fn chars2cstring(s: &[std::ffi::c_char]) -> Option<std::ffi::CString> {
    assert_eq!(
        core::alloc::Layout::new::<std::ffi::c_char>(),
        core::alloc::Layout::new::<u8>()
    );
    let s = unsafe { std::mem::transmute::<&[std::ffi::c_char], &[u8]>(s) };

    let mut buf = alloc::Vec::with_capacity(s.len() + 1);
    buf.extend_from_slice(s);
    buf.push(0); // null terminator

    std::ffi::CString::from_vec_with_nul(buf).ok()
}

#[cfg(feature = "std")]
#[allow(unused)]
pub(crate) fn path2cstring(path: &std::path::Path) -> Result<std::ffi::CString, crate::Error> {
    let path_bytes = path.as_os_str().as_encoded_bytes();
    std::ffi::CString::new(path_bytes).map_err(|_| crate::Error::ToCStr)
}

#[cfg(feature = "std")]
#[allow(unused)]
pub(crate) mod cpp_vec {
    use super::IntoRust;
    use executorch_sys as et_c;

    // pub(crate) fn vec_as_slice<T: CppVecElement>(vec: &T::VecImpl) -> &[T] {
    //     unsafe { std::slice::from_raw_parts(vec.data, vec.len) }
    // }

    // pub(crate) fn vec_as_mut_slice<T: CppVecElement>(vec: &mut T::VecImpl) -> &mut [T] {
    //     unsafe { std::slice::from_raw_parts_mut(vec.data, vec.len) }
    // }

    pub(crate) struct CppVec<T: CppVecElement>(T::VecImpl);
    impl<T: CppVecElement> CppVec<T> {
        pub fn new(vec: T::VecImpl) -> Self {
            Self(vec)
        }

        pub fn as_slice(&self) -> &[T] {
            self.0.as_slice()
        }

        pub fn as_mut_slice(&mut self) -> &mut [T] {
            self.0.as_mut_slice()
        }

        pub fn to_vec(&self) -> Vec<T>
        where
            T: Clone,
        {
            self.as_slice().to_vec()
        }
    }

    impl<V: CppVecImpl> IntoRust for V {
        type RsType = CppVec<V::Element>;
        fn rs(self) -> Self::RsType {
            CppVec(self)
        }
    }
    impl<T: CppVecElement> Drop for CppVec<T> {
        fn drop(&mut self) {
            T::drop_vec(self);
        }
    }
    pub(crate) trait CppVecElement: Sized {
        type VecImpl: CppVecImpl<Element = Self>;
        fn drop_vec(vec: &mut CppVec<Self>);
    }
    pub(crate) trait CppVecImpl {
        type Element: CppVecElement<VecImpl = Self>;
        fn as_slice(&self) -> &[Self::Element];
        fn as_mut_slice(&mut self) -> &mut [Self::Element];
    }
    impl CppVecElement for std::ffi::c_char {
        type VecImpl = et_c::VecChar;
        fn drop_vec(vec: &mut CppVec<Self>) {
            unsafe { et_c::executorch_VecChar_destructor(&mut vec.0) }
        }
    }
    impl CppVecImpl for et_c::VecChar {
        type Element = std::ffi::c_char;
        fn as_slice(&self) -> &[std::ffi::c_char] {
            unsafe { std::slice::from_raw_parts(self.data, self.len) }
        }
        fn as_mut_slice(&mut self) -> &mut [std::ffi::c_char] {
            unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
        }
    }
    impl CppVecElement for et_c::EValueStorage {
        type VecImpl = et_c::VecEValue;
        fn drop_vec(vec: &mut CppVec<Self>) {
            unsafe { et_c::executorch_VecEValue_destructor(&mut vec.0) }
        }
    }
    impl CppVecImpl for et_c::VecEValue {
        type Element = et_c::EValueStorage;
        fn as_slice(&self) -> &[et_c::EValueStorage] {
            let data = self.data.ptr as *const et_c::EValueStorage;
            unsafe { std::slice::from_raw_parts(data, self.len) }
        }
        fn as_mut_slice(&mut self) -> &mut [et_c::EValueStorage] {
            let data = self.data.ptr as *mut et_c::EValueStorage;
            unsafe { std::slice::from_raw_parts_mut(data, self.len) }
        }
    }
    impl CppVecElement for et_c::VecChar {
        type VecImpl = et_c::VecVecChar;
        fn drop_vec(vec: &mut CppVec<Self>) {
            unsafe { et_c::executorch_VecVecChar_destructor(&mut vec.0) }
        }
    }
    impl CppVecImpl for et_c::VecVecChar {
        type Element = et_c::VecChar;
        fn as_slice(&self) -> &[et_c::VecChar] {
            unsafe { std::slice::from_raw_parts(self.data, self.len) }
        }
        fn as_mut_slice(&mut self) -> &mut [et_c::VecChar] {
            unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
        }
    }
}

// Debug func
#[allow(unused)]
#[cfg(feature = "std")]
pub(crate) fn to_bytes<T>(val: &T) -> Vec<u8> {
    (0..std::mem::size_of_val(val))
        .map(|i| unsafe {
            let ptr = val as *const T as *const u8;
            *ptr.add(i)
        })
        .collect()
}
