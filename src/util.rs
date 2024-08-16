//! Utility functions and types.
//!
//! Most of the structs in this module may seems redundant in Rust, but they are wrappers around C++ types
//! that are used in the C++ API. Some structs and functions accept these types as arguments, so they are
//! necessary to interact with the C++ API.

use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;

#[cfg(feature = "alloc")]
use crate::et_alloc;
use crate::et_c;

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
    pub(crate) unsafe fn from_inner(arr: &et_c::ArrayRef<T>) -> Self {
        Self(
            et_c::ArrayRef::<T> {
                Data: arr.Data,
                Length: arr.Length,
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }

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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// Leaner optional class, subset of c10, std, and boost optional APIs.
pub struct Optional<T>(et_c::optional<T>);
impl<T> Optional<T> {
    /// Create a new Optional based on the given Option.
    pub fn new(val: Option<T>) -> Self {
        let is_some = val.is_some();
        Self(et_c::optional::<T> {
            trivial_init: et_c::optional_trivial_init_t { _address: 0 },
            storage_: val
                .map(|value| et_c::optional_storage_t {
                    value_: ManuallyDrop::new(value),
                })
                .unwrap_or(et_c::optional_storage_t {
                    dummy_: ManuallyDrop::new(0),
                }),
            init_: is_some,
            _phantom_0: PhantomData,
        })
    }

    /// Get an optional reference to the value.
    pub fn as_ref(&self) -> Option<&T> {
        self.0.init_.then(|| unsafe { &*self.0.storage_.value_ })
    }

    /// Convert this Optional into an Option.
    pub fn into_option(mut self) -> Option<T> {
        self.0.init_.then(|| {
            self.0.init_ = false;
            unsafe { ManuallyDrop::take(&mut self.0.storage_.value_) }
        })
    }
}
impl<T> Drop for Optional<T> {
    fn drop(&mut self) {
        if self.0.init_ {
            unsafe {
                ManuallyDrop::drop(&mut self.0.storage_.value_);
            }
        }
    }
}
impl<T> From<Optional<T>> for Option<T> {
    fn from(opt: Optional<T>) -> Option<T> {
        opt.into_option()
    }
}
impl<T> From<Option<T>> for Optional<T> {
    fn from(opt: Option<T>) -> Optional<T> {
        Optional::new(opt)
    }
}
impl<T: Clone> Clone for Optional<T> {
    fn clone(&self) -> Self {
        Self::new(self.as_ref().cloned())
    }
}
impl<T: PartialEq> PartialEq for Optional<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl<T: Eq> Eq for Optional<T> {}
impl<T: Hash> Hash for Optional<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}
impl<T: Debug> Debug for Optional<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

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
mod cpp_vec {
    use super::IntoRust;
    use crate::{et_c, et_rs_c, evalue::EValue};

    pub(crate) struct CppVec<T: CppVecElm>(et_rs_c::Vec<T>);
    impl<T: CppVecElm> CppVec<T> {
        pub fn as_slice(&self) -> &[T] {
            self.0.as_slice()
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
    impl IntoRust for et_rs_c::Vec<et_rs_c::Vec<core::ffi::c_char>> {
        type RsType = CppVec<CppVec<core::ffi::c_char>>;
        fn rs(self) -> Self::RsType {
            // Safety: et_rs_c::Vec<T> has the same memory layout as CppVec<T>.
            unsafe {
                std::mem::transmute::<
                    et_rs_c::Vec<et_rs_c::Vec<core::ffi::c_char>>,
                    CppVec<CppVec<core::ffi::c_char>>,
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
    impl CppVecElm for core::ffi::c_char {
        fn drop_vec(vec: &mut CppVec<Self>) {
            unsafe { et_rs_c::Vec_char_destructor(&mut vec.0) }
        }
    }
    impl CppVecElm for CppVec<core::ffi::c_char> {
        fn drop_vec(vec: &mut CppVec<Self>) {
            // Safety: CppVec<T> has the same memory layout as et_rs_c::Vec<T>.
            let vec = unsafe {
                std::mem::transmute::<
                    &mut CppVec<CppVec<core::ffi::c_char>>,
                    &mut et_rs_c::Vec<et_rs_c::Vec<core::ffi::c_char>>,
                >(vec)
            };
            unsafe { et_rs_c::Vec_Vec_char_destructor(vec) }
        }
    }
    impl<'a> CppVecElm for EValue<'a> {
        fn drop_vec(vec: &mut CppVec<Self>) {
            let vec = unsafe {
                std::mem::transmute::<&mut et_rs_c::Vec<EValue<'a>>, &mut et_rs_c::Vec<et_c::EValue>>(
                    &mut vec.0,
                )
            };
            unsafe { et_rs_c::Vec_EValue_destructor(vec) }
        }
    }
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
impl<T: Clone + Copy + Default> DimArr<T> for et_alloc::vec::Vec<T> {
    fn zeros(ndim: usize) -> Self {
        et_alloc::vec::Vec::from_iter(std::iter::repeat(T::default()).take(ndim))
    }
}
