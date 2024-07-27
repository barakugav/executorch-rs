use std::marker::PhantomData;

use ndarray::{ArrayViewD, ArrayViewMut, IxDyn, ShapeBuilder};

use crate::{c_link, et_c, et_rs_c, Error, Result};

pub type SizesType = c_link::executorch_c::root::exec_aten::SizesType;
pub type DimOrderType = c_link::executorch_c::root::exec_aten::DimOrderType;
pub type StridesType = c_link::executorch_c::root::exec_aten::StridesType;

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarType {
    Byte = et_c::ScalarType::Byte as u8,
    Char = et_c::ScalarType::Char as u8,
    Short = et_c::ScalarType::Short as u8,
    Int = et_c::ScalarType::Int as u8,
    Long = et_c::ScalarType::Long as u8,
    Half = et_c::ScalarType::Half as u8,
    Float = et_c::ScalarType::Float as u8,
    Double = et_c::ScalarType::Double as u8,
    ComplexHalf = et_c::ScalarType::ComplexHalf as u8,
    ComplexFloat = et_c::ScalarType::ComplexFloat as u8,
    ComplexDouble = et_c::ScalarType::ComplexDouble as u8,
    Bool = et_c::ScalarType::Bool as u8,
    QInt8 = et_c::ScalarType::QInt8 as u8,
    QUInt8 = et_c::ScalarType::QUInt8 as u8,
    QInt32 = et_c::ScalarType::QInt32 as u8,
    BFloat16 = et_c::ScalarType::BFloat16 as u8,
    QUInt4x2 = et_c::ScalarType::QUInt4x2 as u8,
    QUInt2x4 = et_c::ScalarType::QUInt2x4 as u8,
    Bits1x8 = et_c::ScalarType::Bits1x8 as u8,
    Bits2x4 = et_c::ScalarType::Bits2x4 as u8,
    Bits4x2 = et_c::ScalarType::Bits4x2 as u8,
    Bits8 = et_c::ScalarType::Bits8 as u8,
    Bits16 = et_c::ScalarType::Bits16 as u8,
}
impl ScalarType {
    fn from_c_scalar_type(scalar_type: et_c::ScalarType) -> Option<Self> {
        Some(match scalar_type {
            et_c::ScalarType::Byte => ScalarType::Byte,
            et_c::ScalarType::Char => ScalarType::Char,
            et_c::ScalarType::Short => ScalarType::Short,
            et_c::ScalarType::Int => ScalarType::Int,
            et_c::ScalarType::Long => ScalarType::Long,
            et_c::ScalarType::Half => ScalarType::Half,
            et_c::ScalarType::Float => ScalarType::Float,
            et_c::ScalarType::Double => ScalarType::Double,
            et_c::ScalarType::ComplexHalf => ScalarType::ComplexHalf,
            et_c::ScalarType::ComplexFloat => ScalarType::ComplexFloat,
            et_c::ScalarType::ComplexDouble => ScalarType::ComplexDouble,
            et_c::ScalarType::Bool => ScalarType::Bool,
            et_c::ScalarType::QInt8 => ScalarType::QInt8,
            et_c::ScalarType::QUInt8 => ScalarType::QUInt8,
            et_c::ScalarType::QInt32 => ScalarType::QInt32,
            et_c::ScalarType::BFloat16 => ScalarType::BFloat16,
            et_c::ScalarType::QUInt4x2 => ScalarType::QUInt4x2,
            et_c::ScalarType::QUInt2x4 => ScalarType::QUInt2x4,
            et_c::ScalarType::Bits1x8 => ScalarType::Bits1x8,
            et_c::ScalarType::Bits2x4 => ScalarType::Bits2x4,
            et_c::ScalarType::Bits4x2 => ScalarType::Bits4x2,
            et_c::ScalarType::Bits8 => ScalarType::Bits8,
            et_c::ScalarType::Bits16 => ScalarType::Bits16,
            et_c::ScalarType::Undefined => return None,
            et_c::ScalarType::NumOptions => panic!("Invalid scalar type"),
        })
    }

    fn into_c_scalar_type(self) -> et_c::ScalarType {
        match self {
            ScalarType::Byte => et_c::ScalarType::Byte,
            ScalarType::Char => et_c::ScalarType::Char,
            ScalarType::Short => et_c::ScalarType::Short,
            ScalarType::Int => et_c::ScalarType::Int,
            ScalarType::Long => et_c::ScalarType::Long,
            ScalarType::Half => et_c::ScalarType::Half,
            ScalarType::Float => et_c::ScalarType::Float,
            ScalarType::Double => et_c::ScalarType::Double,
            ScalarType::ComplexHalf => et_c::ScalarType::ComplexHalf,
            ScalarType::ComplexFloat => et_c::ScalarType::ComplexFloat,
            ScalarType::ComplexDouble => et_c::ScalarType::ComplexDouble,
            ScalarType::Bool => et_c::ScalarType::Bool,
            ScalarType::QInt8 => et_c::ScalarType::QInt8,
            ScalarType::QUInt8 => et_c::ScalarType::QUInt8,
            ScalarType::QInt32 => et_c::ScalarType::QInt32,
            ScalarType::BFloat16 => et_c::ScalarType::BFloat16,
            ScalarType::QUInt4x2 => et_c::ScalarType::QUInt4x2,
            ScalarType::QUInt2x4 => et_c::ScalarType::QUInt2x4,
            ScalarType::Bits1x8 => et_c::ScalarType::Bits1x8,
            ScalarType::Bits2x4 => et_c::ScalarType::Bits2x4,
            ScalarType::Bits4x2 => et_c::ScalarType::Bits4x2,
            ScalarType::Bits8 => et_c::ScalarType::Bits8,
            ScalarType::Bits16 => et_c::ScalarType::Bits16,
        }
    }
}
pub trait Scalar {
    const TYPE: ScalarType;
}
impl Scalar for u8 {
    const TYPE: ScalarType = ScalarType::Byte;
}
impl Scalar for i8 {
    const TYPE: ScalarType = ScalarType::Char;
}
impl Scalar for i16 {
    const TYPE: ScalarType = ScalarType::Short;
}
impl Scalar for i32 {
    const TYPE: ScalarType = ScalarType::Int;
}
impl Scalar for i64 {
    const TYPE: ScalarType = ScalarType::Long;
}
// impl Scalar for f16 {
//     const TYPE: ScalarType = ScalarType::Half;
// }
impl Scalar for f32 {
    const TYPE: ScalarType = ScalarType::Float;
}
impl Scalar for f64 {
    const TYPE: ScalarType = ScalarType::Double;
}
impl Scalar for bool {
    const TYPE: ScalarType = ScalarType::Bool;
}

pub struct Tensor<'a>(pub(crate) et_c::Tensor, PhantomData<&'a ()>);
impl<'a> Tensor<'a> {
    pub fn new(tensor_impl: &'a mut TensorImpl<'a>) -> Self {
        let impl_ = &mut tensor_impl.0;
        Self(et_c::Tensor { impl_ }, PhantomData)
    }

    pub(crate) unsafe fn from_inner(tensor: et_c::Tensor) -> Self {
        Self(tensor, PhantomData)
    }

    pub fn nbytes(&self) -> usize {
        unsafe { et_rs_c::Tensor_nbytes(&self.0) }
    }

    pub fn size(&self, dim: isize) -> isize {
        unsafe { et_rs_c::Tensor_size(&self.0, dim) }
    }

    pub fn dim(&self) -> isize {
        unsafe { et_rs_c::Tensor_dim(&self.0) }
    }

    pub fn numel(&self) -> isize {
        unsafe { et_rs_c::Tensor_numel(&self.0) }
    }

    pub fn scalar_type(&self) -> Option<ScalarType> {
        let scalar_type = unsafe { et_rs_c::Tensor_scalar_type(&self.0) };
        ScalarType::from_c_scalar_type(scalar_type)
    }

    pub fn element_size(&self) -> isize {
        unsafe { et_rs_c::Tensor_element_size(&self.0) }
    }

    pub fn sizes(&self) -> &'a [SizesType] {
        unsafe {
            let arr = et_rs_c::Tensor_sizes(&self.0);
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    pub fn dim_order(&self) -> &'a [DimOrderType] {
        unsafe {
            let arr = et_rs_c::Tensor_dim_order(&self.0);
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    pub fn strides(&self) -> &'a [StridesType] {
        unsafe {
            let arr = et_rs_c::Tensor_strides(&self.0);
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    pub fn as_array<S: Scalar>(&self) -> ArrayViewD<'a, S> {
        if self.scalar_type() != Some(S::TYPE) {
            panic!("Invalid type");
        }

        let ptr = unsafe { et_rs_c::Tensor_const_data_ptr(&self.0) } as *const S;

        let shape = self
            .sizes()
            .iter()
            .map(|&size| size as usize)
            .collect::<Vec<_>>();
        let strides = self
            .strides()
            .iter()
            .map(|&stride| stride as usize)
            .collect::<Vec<_>>();

        let dim_order = self
            .dim_order()
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();

        unsafe { ArrayViewD::from_shape_ptr(shape.strides(strides), ptr) }.permuted_axes(dim_order)
    }

    pub unsafe fn as_array_mut<S: Scalar>(&self) -> ArrayViewMut<'a, S, IxDyn> {
        if self.scalar_type() != Some(S::TYPE) {
            panic!("Invalid type");
        }

        let ptr = unsafe { et_rs_c::Tensor_mutable_data_ptr(&self.0) } as *mut S;

        let shape = self
            .sizes()
            .iter()
            .map(|&size| size as usize)
            .collect::<Vec<_>>();
        let strides = self
            .strides()
            .iter()
            .map(|&stride| stride as usize)
            .collect::<Vec<_>>();

        let dim_order = self
            .dim_order()
            .iter()
            .map(|&dim| dim as usize)
            .collect::<Vec<_>>();

        unsafe { ArrayViewMut::from_shape_ptr(shape.strides(strides), ptr) }
            .permuted_axes(dim_order)
    }
}

impl<'a, S: Scalar> TryFrom<Tensor<'a>> for ArrayViewD<'a, S> {
    type Error = Error;
    fn try_from(tensor: Tensor<'a>) -> Result<Self> {
        if tensor.scalar_type() != Some(S::TYPE) {
            return Err(Error::InvalidType);
        }
        Ok(tensor.as_array())
    }
}

pub struct TensorImpl<'a>(et_c::TensorImpl, PhantomData<&'a ()>);
impl<'a> TensorImpl<'a> {
    pub fn new<S: Scalar>(
        sizes: &'a [SizesType],
        data: &mut [S],
        data_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        let dim = sizes.len();
        assert_eq!(dim, data_order.len());
        assert_eq!(dim, strides.len());
        let sizes = sizes as *const _ as *mut SizesType;
        let data = data.as_mut_ptr() as *mut _;
        let dim_order = data_order as *const _ as *mut DimOrderType;
        let strides = strides as *const _ as *mut StridesType;
        let impl_ = unsafe {
            et_c::TensorImpl::new(
                S::TYPE.into_c_scalar_type(),
                dim as isize,
                sizes,
                data,
                dim_order,
                strides,
                et_c::TensorShapeDynamism::STATIC,
            )
        };
        Self(impl_, PhantomData)
    }
}
