use core::marker::PhantomData;

use executorch_sys as et_c;

use crate::tensor::ScalarType;
use crate::util::{IntoRust, __ArrayRefImpl};

/// Describes the layout of a tensor.
#[repr(transparent)]
pub struct TensorLayout<'a>(et_c::TensorLayout, PhantomData<&'a ()>);
impl<'a> TensorLayout<'a> {
    /// Returns the sizes of the tensor.
    pub fn sizes(&self) -> &[i32] {
        unsafe { et_c::executorch_TensorLayout_sizes(&self.0 as *const _).as_slice() }
    }

    /// Returns the dim order of the tensor.
    pub fn dim_order(&self) -> &[u8] {
        unsafe { et_c::executorch_TensorLayout_dim_order(&self.0 as *const _).as_slice() }
    }

    /// Returns the scalar type of the tensor.
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { et_c::executorch_TensorLayout_scalar_type(&self.0 as *const _) }.rs()
    }

    /// Returns the size of the tensor in bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { et_c::executorch_TensorLayout_nbytes(&self.0 as *const _) }
    }
}
