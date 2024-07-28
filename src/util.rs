use std::marker::PhantomData;

use crate::{et_c, et_rs_c};

pub trait IntoRust {
    type RsType;
    fn rs(self) -> Self::RsType;
}

#[allow(dead_code)]
pub struct Span<'a, T>(pub(crate) et_c::Span<T>, PhantomData<&'a T>);
impl<'a, T> Span<'a, T> {
    pub fn new(s: &'a [T]) -> Self {
        Self(
            et_c::Span {
                data_: s.as_ptr() as *mut T,
                length_: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }
}

#[allow(dead_code)]
pub struct SpanMut<'a, T>(pub(crate) et_c::Span<T>, PhantomData<&'a T>);
impl<'a, T> SpanMut<'a, T> {
    pub fn new(s: &'a mut [T]) -> Self {
        Self(
            et_c::Span {
                data_: s.as_mut_ptr(),
                length_: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
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
