use std::marker::PhantomData;

use crate::{et_c, et_rs_c};

pub trait IntoRust {
    type RsType;
    fn rs(self) -> Self::RsType;
}

#[allow(dead_code)]
pub struct Span<'a, T>(et_c::Span<T>, PhantomData<&'a T>);
impl<'a, T> Span<'a, T> {
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

impl<T> IntoRust for et_rs_c::RawVec<T> {
    type RsType = Vec<T>;
    fn rs(self) -> Self::RsType {
        unsafe { Vec::from_raw_parts(self.data, self.len, self.cap) }
    }
}
