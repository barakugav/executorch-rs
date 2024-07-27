use std::marker::PhantomData;

use crate::et_c;

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
