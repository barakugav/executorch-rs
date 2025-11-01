use core::marker::PhantomData;

use crate::sys;

use crate::tensor::ScalarType;
use crate::util::{IntoRust, __ArrayRefImpl};

/// Describes the layout of a tensor.
#[repr(transparent)]
pub struct TensorLayout<'a>(sys::TensorLayout, PhantomData<&'a ()>);
impl<'a> TensorLayout<'a> {
    pub(crate) unsafe fn from_raw(raw: sys::TensorLayout) -> TensorLayout<'a> {
        Self(raw, PhantomData)
    }

    /// Returns the sizes of the tensor.
    pub fn sizes(&self) -> &[i32] {
        unsafe { sys::executorch_TensorLayout_sizes(&self.0 as *const _).as_slice() }
    }

    /// Returns the dim order of the tensor.
    pub fn dim_order(&self) -> &[u8] {
        unsafe { sys::executorch_TensorLayout_dim_order(&self.0 as *const _).as_slice() }
    }

    /// Returns the scalar type of the tensor.
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { sys::executorch_TensorLayout_scalar_type(&self.0 as *const _) }.rs()
    }

    /// Returns the size of the tensor in bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { sys::executorch_TensorLayout_nbytes(&self.0 as *const _) }
    }
}
