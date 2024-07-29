use std::marker::PhantomData;

use crate::{et_c, et_rs_c};

pub trait IntoRust {
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
    // pub(crate) unsafe fn new(arr: et_c::ArrayRef<T>) -> Self {
    //     Self(arr, PhantomData)
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

#[allow(dead_code)]
pub(crate) fn str2chars<'a>(s: &'a str) -> Result<&'a [std::os::raw::c_char], &'static str> {
    let bytes = s.as_bytes();
    if let Some(_) = bytes.iter().position(|&b| b == 0) {
        return Err("String contains null byte");
    }
    let chars: *const std::os::raw::c_char = bytes.as_ptr().cast();
    Ok(unsafe { std::slice::from_raw_parts(chars, bytes.len()) })
}
#[allow(dead_code)]
pub(crate) fn chars2string(chars: Vec<std::os::raw::c_char>) -> String {
    let bytes = unsafe { std::mem::transmute::<Vec<std::os::raw::c_char>, Vec<u8>>(chars) };
    String::from_utf8(bytes).unwrap()
}

impl<T> IntoRust for et_rs_c::RawVec<T> {
    type RsType = Vec<T>;
    fn rs(self) -> Self::RsType {
        unsafe { Vec::from_raw_parts(self.data, self.len, self.cap) }
    }
}

// Debug func
#[allow(dead_code)]
pub(crate) fn to_bytes<T>(val: &T) -> Vec<u8> {
    (0..std::mem::size_of_val(val))
        .map(|i| unsafe {
            let ptr = val as *const _;
            let ptr = ptr as usize;
            let ptr = ptr as *const u8;
            *ptr.add(i)
        })
        .collect()
}
