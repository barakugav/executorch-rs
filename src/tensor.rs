use std::any::TypeId;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;

use ndarray::{ArrayView, ArrayViewD, ArrayViewMut, Axis, Dimension, IxDyn, ShapeBuilder};

use crate::{c_link, et_c, et_rs_c, util, Span};

pub type SizesType = c_link::executorch_c::root::exec_aten::SizesType;
pub type DimOrderType = c_link::executorch_c::root::exec_aten::DimOrderType;
pub type StridesType = c_link::executorch_c::root::exec_aten::StridesType;

/// Data types (dtypes) that can be used as element types in Tensors.
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
    private_decl! {}
}
macro_rules! impl_scalar {
    ($rust_type:ident, $scalar_type_variant:ident) => {
        impl Scalar for $rust_type {
            const TYPE: ScalarType = ScalarType::$scalar_type_variant;
            private_impl! {}
        }
    };
}
impl_scalar!(u8, Byte);
impl_scalar!(i8, Char);
impl_scalar!(i16, Short);
impl_scalar!(i32, Int);
impl_scalar!(i64, Long);
// impl_scalar!(f16, Half);
impl_scalar!(f32, Float);
impl_scalar!(f64, Double);
impl_scalar!(bool, Bool);

/// A minimal Tensor type whose API is a source compatible subset of at::Tensor.
///
/// NOTE: Instances of this class do not own the TensorImpl given to it,
/// which means that the caller must guarantee that the TensorImpl lives longer
/// than any Tensor instances that point to it.
///
/// See the documentation on TensorImpl for details about the return/parameter
/// types used here and how they relate to at::Tensor.
pub struct TensorBase<'a, D: Data>(pub(crate) et_c::Tensor, PhantomData<(&'a (), D)>);
impl<'a, D: Data> TensorBase<'a, D> {
    unsafe fn new_impl(tensor_impl: &'a TensorImplBase<'_, D>) -> Self {
        let impl_ = &tensor_impl.0 as *const _ as *mut _;
        Self(et_c::Tensor { impl_ }, PhantomData)
    }

    pub(crate) unsafe fn from_inner(tensor: et_c::Tensor) -> Self {
        Self(tensor, PhantomData)
    }

    /// Returns the size of the tensor in bytes.
    ///
    /// NOTE: Only the alive space is returned not the total capacity of the
    /// underlying data blob.
    pub fn nbytes(&self) -> usize {
        unsafe { et_rs_c::Tensor_nbytes(&self.0) }
    }

    /// Returns the size of the tensor at the given dimension.
    ///
    /// NOTE: that size() intentionally does not return SizeType even though it
    /// returns an element of an array of SizeType. This is to help make calls of
    /// this method more compatible with at::Tensor, and more consistent with the
    /// rest of the methods on this class and in ETensor.
    pub fn size(&self, dim: isize) -> isize {
        unsafe { et_rs_c::Tensor_size(&self.0, dim) }
    }

    /// Returns the tensor's number of dimensions.
    pub fn dim(&self) -> isize {
        unsafe { et_rs_c::Tensor_dim(&self.0) }
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> isize {
        unsafe { et_rs_c::Tensor_numel(&self.0) }
    }

    /// Returns the type of the elements in the tensor (int32, float, bool, etc).
    pub fn scalar_type(&self) -> Option<ScalarType> {
        let scalar_type = unsafe { et_rs_c::Tensor_scalar_type(&self.0) };
        ScalarType::from_c_scalar_type(scalar_type)
    }

    /// Returns the size in bytes of one element of the tensor.
    pub fn element_size(&self) -> isize {
        unsafe { et_rs_c::Tensor_element_size(&self.0) }
    }

    /// Returns the sizes of the tensor at each dimension.
    pub fn sizes(&self) -> &'a [SizesType] {
        unsafe {
            let arr = et_rs_c::Tensor_sizes(&self.0);
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    /// Returns the order the dimensions are laid out in memory.
    pub fn dim_order(&self) -> &'a [DimOrderType] {
        unsafe {
            let arr = et_rs_c::Tensor_dim_order(&self.0);
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    /// Returns the strides of the tensor at each dimension.
    pub fn strides(&self) -> &'a [StridesType] {
        unsafe {
            let arr = et_rs_c::Tensor_strides(&self.0);
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    /// Returns a pointer of type S to the constant underlying data blob.
    pub fn as_ptr<S: Scalar>(&self) -> *const S {
        assert_eq!(self.scalar_type(), Some(S::TYPE), "Invalid type");
        (unsafe { et_rs_c::Tensor_const_data_ptr(&self.0) }) as *const S
    }

    /// Returns a pointer to the constant underlying data blob.
    ///
    /// Safety: The caller must access the values in the returned pointer according to the type of the tensor.
    pub unsafe fn as_ptr_bytes(&self) -> *const u8 {
        (unsafe { et_rs_c::Tensor_const_data_ptr(&self.0) }) as *const u8
    }

    pub fn as_array<'b, S: Scalar + 'static, Dim: Dimension + 'static>(
        &'b self,
    ) -> ArrayView<'b, S, Dim> {
        let ptr = self.as_ptr::<S>();
        match Dim::NDIM {
            None => {
                // dynamic array
                assert_eq!(TypeId::of::<Dim>(), TypeId::of::<IxDyn>());
                let shape = self.sizes().iter().map(|d| *d as usize).collect::<Vec<_>>();
                let strides = self
                    .strides()
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>();
                let dim_order = self
                    .dim_order()
                    .iter()
                    .map(|o| *o as usize)
                    .collect::<Vec<_>>();
                let arr = unsafe {
                    ArrayViewD::from_shape_ptr(shape.strides(strides), ptr).permuted_axes(dim_order)
                };
                assert_eq!(
                    TypeId::of::<ArrayView<'_, S, IxDyn>>(),
                    TypeId::of::<ArrayView<'_, S, Dim>>()
                );
                unsafe {
                    util::unlimited_transmute::<ArrayView<'_, S, IxDyn>, ArrayView<'_, S, Dim>>(arr)
                }
            }
            Some(ndim) if ndim == self.dim() as usize => {
                // safe because Dim == D2
                let mut dim = Dim::default();
                let mut strides = Dim::default();
                let mut dim_order = Dim::default();
                assert_eq!(dim.ndim(), self.dim() as usize);
                assert_eq!(strides.ndim(), self.dim() as usize);
                assert_eq!(dim_order.ndim(), self.dim() as usize);
                for (i, d) in self.sizes().iter().enumerate() {
                    dim[i] = *d as usize;
                }
                for (i, s) in self.strides().iter().enumerate() {
                    strides[i] = *s as usize;
                }
                for (i, s) in self.dim_order().iter().enumerate() {
                    dim_order[i] = *s as usize;
                }

                unsafe { ArrayView::from_shape_ptr(dim.strides(strides), ptr) }
                    .permuted_axes(dim_order)
            }
            Some(ndim) => {
                panic!(
                    "Invalid number of dimensions: expected {}, got {}",
                    ndim,
                    self.dim()
                );
            }
        }
    }

    pub fn as_array_dyn<'b, S: Scalar + 'static>(&'b self) -> ArrayViewD<'b, S> {
        self.as_array()
    }
}
impl Drop for et_c::Tensor {
    fn drop(&mut self) {
        unsafe { et_rs_c::Tensor_destructor(self) }
    }
}

pub type Tensor<'a> = TensorBase<'a, View>;
impl<'a> Tensor<'a> {
    pub fn new(tensor_impl: &'a TensorImpl<'_>) -> Self {
        unsafe { Tensor::new_impl(tensor_impl) }
    }
}
pub type TensorMut<'a> = TensorBase<'a, ViewMut>;
impl<'a> TensorMut<'a> {
    pub fn new(tensor_impl: &'a mut TensorImplMut<'_>) -> Self {
        unsafe { Self::new_impl(tensor_impl) }
    }

    pub fn as_mut_ptr<S: Scalar>(&self) -> *mut S {
        assert_eq!(self.scalar_type(), Some(S::TYPE), "Invalid type");
        (unsafe { et_rs_c::Tensor_mutable_data_ptr(&self.0) }) as *mut S
    }

    pub fn as_tensor<'b>(&'b self) -> Tensor<'b> {
        unsafe {
            Tensor::from_inner(et_c::Tensor {
                impl_: self.0.impl_,
            })
        }
    }

    pub fn into_tensor(self) -> Tensor<'a> {
        unsafe {
            Tensor::from_inner(et_c::Tensor {
                impl_: self.0.impl_,
            })
        }
    }

    pub fn as_array_mut<S: Scalar + 'static, D: Dimension + 'static>(
        &mut self,
    ) -> ArrayViewMut<'a, S, D> {
        let ptr = self.as_mut_ptr::<S>();
        match D::NDIM {
            None => {
                // dynamic array
                assert_eq!(TypeId::of::<D>(), TypeId::of::<IxDyn>());
                let shape = self.sizes().iter().map(|d| *d as usize).collect::<Vec<_>>();
                let strides = TensorBase::strides(self)
                    .iter()
                    .map(|s| *s as usize)
                    .collect::<Vec<_>>();
                let dim_order = self
                    .dim_order()
                    .iter()
                    .map(|o| *o as usize)
                    .collect::<Vec<_>>();
                let arr = unsafe {
                    ArrayViewMut::from_shape_ptr(shape.strides(strides), ptr)
                        .permuted_axes(dim_order)
                };
                assert_eq!(
                    TypeId::of::<ArrayViewMut<'_, S, IxDyn>>(),
                    TypeId::of::<ArrayViewMut<'_, S, D>>()
                );
                unsafe {
                    util::unlimited_transmute::<ArrayViewMut<'_, S, IxDyn>, ArrayViewMut<'_, S, D>>(
                        arr,
                    )
                }
            }
            Some(ndim) if ndim == self.dim() as usize => {
                // safe because D == D2
                let mut dim = D::default();
                let mut strides = D::default();
                let mut dim_order = D::default();
                assert_eq!(dim.ndim(), self.dim() as usize);
                assert_eq!(strides.ndim(), self.dim() as usize);
                assert_eq!(dim_order.ndim(), self.dim() as usize);
                for (i, d) in self.sizes().iter().enumerate() {
                    dim[i] = *d as usize;
                }
                for (i, s) in TensorBase::strides(self).iter().enumerate() {
                    strides[i] = *s as usize;
                }
                for (i, s) in self.dim_order().iter().enumerate() {
                    dim_order[i] = *s as usize;
                }

                unsafe { ArrayViewMut::from_shape_ptr(dim.strides(strides), ptr) }
                    .permuted_axes(dim_order)
            }
            Some(ndim) => {
                panic!(
                    "Invalid number of dimensions: expected {}, got {}",
                    ndim,
                    self.dim()
                );
            }
        }
    }

    pub fn as_array_mut_dyn<S: Scalar + 'static>(&mut self) -> ArrayViewMut<'a, S, IxDyn> {
        self.as_array_mut()
    }
}

pub struct TensorImplBase<'a, D: Data>(et_c::TensorImpl, PhantomData<(&'a (), D)>);
pub type TensorImpl<'a> = TensorImplBase<'a, View>;
pub type TensorImplMut<'a> = TensorImplBase<'a, ViewMut>;

impl<'a, D: Data> TensorImplBase<'a, D> {
    unsafe fn from_slice_impl<S: Scalar>(
        sizes: &'a [SizesType],
        data: *mut S,
        data_order: Option<&'a [DimOrderType]>,
        strides: &'a [StridesType],
    ) -> Self {
        let dim = sizes.len();
        if let Some(data_order) = &data_order {
            assert_eq!(dim, data_order.len());
        }
        assert_eq!(dim, strides.len());
        let sizes = sizes.as_ptr() as *mut SizesType;
        let dim_order = data_order
            .map(|p| p.as_ptr() as *mut DimOrderType)
            .unwrap_or(ptr::null_mut());
        let strides = strides.as_ptr() as *mut StridesType;
        let impl_ = unsafe {
            et_c::TensorImpl::new(
                S::TYPE.into_c_scalar_type(),
                dim as isize,
                sizes,
                data as *mut _,
                dim_order,
                strides,
                et_c::TensorShapeDynamism::STATIC,
            )
        };
        Self(impl_, PhantomData)
    }
}

impl<'a> TensorImpl<'a> {
    pub fn from_slice<S: Scalar>(
        sizes: &'a [SizesType],
        data: &'a [S],
        data_order: Option<&'a [DimOrderType]>,
        strides: &'a [StridesType],
    ) -> Self {
        let data = data.as_ptr() as *mut S;
        unsafe { Self::from_slice_impl(sizes, data, data_order, strides) }
    }

    pub fn from_array<S: Scalar, D: Dimension>(array: ArrayView<'a, S, D>) -> impl AsRef<Self> {
        struct Wrapper<'a, S: Scalar> {
            sizes: Vec<SizesType>,
            strides: Vec<StridesType>,
            tensor_impl: MaybeUninit<TensorImpl<'a>>,
            _phantom: PhantomData<S>,
        }
        impl<'a, S: Scalar> AsRef<TensorImpl<'a>> for Wrapper<'a, S> {
            fn as_ref(&self) -> &TensorImpl<'a> {
                unsafe { self.tensor_impl.assume_init_ref() }
            }
        }

        let mut wrapper = Wrapper::<'a, S> {
            sizes: array
                .shape()
                .iter()
                .map(|&size| size as SizesType)
                .collect(),
            strides: (0..array.ndim())
                .map(|d| array.stride_of(Axis(d)) as StridesType)
                .collect(),
            tensor_impl: MaybeUninit::uninit(),
            _phantom: PhantomData,
        };
        let impl_ = unsafe {
            et_c::TensorImpl::new(
                S::TYPE.into_c_scalar_type(),
                array.ndim() as isize,
                wrapper.sizes.as_slice().as_ptr() as *mut SizesType,
                array.as_ptr() as *mut _,
                ptr::null_mut(),
                wrapper.strides.as_slice().as_ptr() as *mut StridesType,
                et_c::TensorShapeDynamism::STATIC,
            )
        };
        wrapper
            .tensor_impl
            .write(TensorImplBase(impl_, PhantomData));
        wrapper
    }
}

impl<'a> TensorImplMut<'a> {
    pub fn from_slice<S: Scalar>(
        sizes: &'a [SizesType],
        data: &'a mut [S],
        data_order: Option<&'a [DimOrderType]>,
        strides: &'a [StridesType],
    ) -> Self {
        unsafe { Self::from_slice_impl(sizes, data.as_mut_ptr(), data_order, strides) }
    }

    pub fn from_array<S: Scalar, D: Dimension>(
        mut array: ArrayViewMut<'a, S, D>,
    ) -> impl AsRef<Self> {
        struct Wrapper<'a, S: Scalar> {
            sizes: Vec<SizesType>,
            strides: Vec<StridesType>,
            tensor_impl: MaybeUninit<TensorImplMut<'a>>,
            _phantom: PhantomData<S>,
        }
        impl<'a, S: Scalar> AsRef<TensorImplMut<'a>> for Wrapper<'a, S> {
            fn as_ref(&self) -> &TensorImplMut<'a> {
                unsafe { self.tensor_impl.assume_init_ref() }
            }
        }

        let mut wrapper = Wrapper::<'a, S> {
            sizes: array
                .shape()
                .iter()
                .map(|&size| size as SizesType)
                .collect(),
            strides: (0..array.ndim())
                .map(|d| array.stride_of(Axis(d)) as StridesType)
                .collect(),
            tensor_impl: MaybeUninit::uninit(),
            _phantom: PhantomData,
        };
        let impl_ = unsafe {
            et_c::TensorImpl::new(
                S::TYPE.into_c_scalar_type(),
                array.ndim() as isize,
                wrapper.sizes.as_slice().as_ptr() as *mut SizesType,
                array.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                wrapper.strides.as_slice().as_ptr() as *mut StridesType,
                et_c::TensorShapeDynamism::STATIC,
            )
        };
        wrapper
            .tensor_impl
            .write(TensorImplBase(impl_, PhantomData));
        wrapper
    }
}

pub trait Data {}
#[allow(dead_code)]
pub trait DataMut: Data {}
pub struct View {}
impl Data for View {}
pub struct ViewMut {}
impl Data for ViewMut {}
impl DataMut for ViewMut {}

/// Metadata about a specific tensor of an ExecuTorch Program.
///
/// The program used to create the MethodMeta object that created this
/// TensorInfo must outlive this TensorInfo.
pub struct TensorInfo<'a>(et_c::TensorInfo, PhantomData<&'a ()>);
impl<'a> TensorInfo<'a> {
    pub(crate) unsafe fn new(info: et_c::TensorInfo) -> Self {
        Self(info, PhantomData)
    }

    /// Returns the sizes of the tensor.
    pub fn sizes(&self) -> &'a [i32] {
        let span = unsafe { et_c::TensorInfo_sizes(&self.0) };
        unsafe { Span::new(span) }.as_slice()
    }

    /// Returns the dim order of the tensor.
    pub fn dim_order(&self) -> &'a [u8] {
        let span = unsafe { et_c::TensorInfo_dim_order(&self.0) };
        unsafe { Span::new(span) }.as_slice()
    }

    /// Returns the scalar type of the input/output.
    pub fn scalar_type(&self) -> Option<ScalarType> {
        let scalar_type = unsafe { et_c::TensorInfo_scalar_type(&self.0) };
        ScalarType::from_c_scalar_type(scalar_type)
    }

    /// Returns the size of the tensor in bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { et_c::TensorInfo_nbytes(&self.0) }
    }
}
