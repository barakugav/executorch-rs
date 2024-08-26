//! Tensor struct is a type erased input or output tensor to a executorch program.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::pin::Pin;

use ndarray::{ArrayBase, ArrayView, ArrayViewMut, ShapeBuilder};

#[cfg(feature = "alloc")]
use crate::et_alloc;
use crate::util::{Destroy, DimArr, FixedSizeDim, NonTriviallyMovable, Span, Storable, Storage};
use crate::{et_c, et_rs_c};

/// A type that represents the sizes (dimensions) of a tensor.
pub type SizesType = executorch_sys::exec_aten::SizesType;
/// A type that represents the order of the dimensions of a tensor.
pub type DimOrderType = executorch_sys::exec_aten::DimOrderType;
/// A type that represents the strides of a tensor.
pub type StridesType = executorch_sys::exec_aten::StridesType;

/// Data types (dtypes) that can be used as element types in Tensors.
///
/// The enum contain all the scalar types supported by the Cpp ExecuTorch library.
/// Not all of these types are supported by the Rust library, see [`Scalar`].
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarType {
    /// 8-bit unsigned integer, `u8`
    Byte = et_c::ScalarType::Byte as u8,
    /// 8-bit signed, integer, `i8`
    Char = et_c::ScalarType::Char as u8,
    /// 16-bit signed integer, `i16`
    Short = et_c::ScalarType::Short as u8,
    /// 32-bit signed integer, `i32`
    Int = et_c::ScalarType::Int as u8,
    /// 64-bit signed integer, `i64`
    Long = et_c::ScalarType::Long as u8,
    /// 16-bit floating point, `half::f16`, enabled by the `f16` feature
    Half = et_c::ScalarType::Half as u8,
    /// 32-bit floating point, `f32`
    Float = et_c::ScalarType::Float as u8,
    /// 64-bit floating point, `f64`
    Double = et_c::ScalarType::Double as u8,
    /// 16-bit complex floating point, `num_complex::Complex<half::f16>`, enabled by the `complex` and `f16` features
    ComplexHalf = et_c::ScalarType::ComplexHalf as u8,
    /// 32-bit complex floating point, `num_complex::Complex32`, enabled by the `complex` feature
    ComplexFloat = et_c::ScalarType::ComplexFloat as u8,
    /// 64-bit complex floating point, `num_complex::Complex64`, enabled by the `complex` feature
    ComplexDouble = et_c::ScalarType::ComplexDouble as u8,
    /// Boolean, `bool`
    Bool = et_c::ScalarType::Bool as u8,
    /// **\[Unsupported\]** 8-bit quantized integer
    QInt8 = et_c::ScalarType::QInt8 as u8,
    /// **\[Unsupported\]** 8-bit quantized unsigned integer
    QUInt8 = et_c::ScalarType::QUInt8 as u8,
    /// **\[Unsupported\]** 32-bit quantized integer
    QInt32 = et_c::ScalarType::QInt32 as u8,
    /// 16-bit floating point using the bfloat16 format, `half::bf16`, enabled by the `f16` feature
    BFloat16 = et_c::ScalarType::BFloat16 as u8,
    /// **\[Unsupported\]**
    QUInt4x2 = et_c::ScalarType::QUInt4x2 as u8,
    /// **\[Unsupported\]**
    QUInt2x4 = et_c::ScalarType::QUInt2x4 as u8,
    /// **\[Unsupported\]**
    Bits1x8 = et_c::ScalarType::Bits1x8 as u8,
    /// **\[Unsupported\]**
    Bits2x4 = et_c::ScalarType::Bits2x4 as u8,
    /// **\[Unsupported\]**
    Bits4x2 = et_c::ScalarType::Bits4x2 as u8,
    /// **\[Unsupported\]**
    Bits8 = et_c::ScalarType::Bits8 as u8,
    /// **\[Unsupported\]**
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

/// A trait for types that can be used as scalar types in Tensors.
pub trait Scalar {
    /// The [`ScalarType`] enum variant of the implementing type.
    const TYPE: ScalarType;
    private_decl! {}
}
macro_rules! impl_scalar {
    ($rust_type:path, $scalar_type_variant:ident) => {
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
#[cfg(feature = "f16")]
impl_scalar!(half::f16, Half);
impl_scalar!(f32, Float);
impl_scalar!(f64, Double);
#[cfg(all(feature = "complex", feature = "f16"))]
impl_scalar!(num_complex::Complex<half::f16>, ComplexHalf);
#[cfg(feature = "complex")]
impl_scalar!(num_complex::Complex32, ComplexFloat);
#[cfg(feature = "complex")]
impl_scalar!(num_complex::Complex64, ComplexDouble);
impl_scalar!(bool, Bool);
#[cfg(feature = "f16")]
impl_scalar!(half::bf16, BFloat16);

/// A minimal Tensor type whose API is a source compatible subset of at::Tensor.
///
/// This class is a base class for [`Tensor`] and [`TensorMut`] and is not meant to be
/// used directly. It is used to provide a common API for both of them.
///
/// NOTE: Instances of this class do not own the TensorImpl given to it,
/// which means that the caller must guarantee that the TensorImpl lives longer
/// than any Tensor instances that point to it.
///
/// See the documentation on TensorImpl for details about the return/parameter
/// types used here and how they relate to at::Tensor.
pub struct TensorBase<'a, D: Data>(
    NonTriviallyMovable<'a, et_c::Tensor>,
    PhantomData<(
        // phantom for the lifetime of the TensorImpl we depends on
        &'a (),
        D,
    )>,
);
impl<'a, D: Data> TensorBase<'a, D> {
    #[cfg(feature = "alloc")]
    fn new_boxed(tensor_impl: &'a TensorImplBase<D>) -> Self {
        let impl_ = &tensor_impl.0 as *const et_c::TensorImpl;
        let impl_ = impl_.cast_mut();
        // Safety: the closure init the pointer
        let tensor = unsafe { NonTriviallyMovable::new_boxed(|p| et_rs_c::Tensor_new(p, impl_)) };
        Self(tensor, PhantomData)
    }

    fn new_in_storage(
        tensor_impl: &'a TensorImplBase<D>,
        storage: Pin<&'a mut Storage<TensorBase<D>>>,
    ) -> Self {
        let impl_ = &tensor_impl.0 as *const et_c::TensorImpl;
        let impl_ = impl_.cast_mut();
        // Safety: the closure init the pointer
        let tensor = unsafe {
            NonTriviallyMovable::new_in_storage(|p| et_rs_c::Tensor_new(p, impl_), storage)
        };
        Self(tensor, PhantomData)
    }

    pub(crate) fn tensor_ref(&self) -> &et_c::Tensor {
        self.0.as_ref()
    }

    /// Returns the size of the tensor in bytes.
    ///
    /// NOTE: Only the alive space is returned not the total capacity of the
    /// underlying data blob.
    pub fn nbytes(&self) -> usize {
        unsafe { et_rs_c::Tensor_nbytes(self.tensor_ref()) }
    }

    /// Returns the size of the tensor at the given dimension.
    ///
    /// NOTE: that size() intentionally does not return SizeType even though it
    /// returns an element of an array of SizeType. This is to help make calls of
    /// this method more compatible with at::Tensor, and more consistent with the
    /// rest of the methods on this class and in ETensor.
    pub fn size(&self, dim: isize) -> isize {
        unsafe { et_rs_c::Tensor_size(self.tensor_ref(), dim) }
    }

    /// Returns the tensor's number of dimensions.
    pub fn dim(&self) -> isize {
        unsafe { et_rs_c::Tensor_dim(self.tensor_ref()) }
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> isize {
        unsafe { et_rs_c::Tensor_numel(self.tensor_ref()) }
    }

    /// Returns the type of the elements in the tensor (int32, float, bool, etc).
    pub fn scalar_type(&self) -> Option<ScalarType> {
        let scalar_type = unsafe { et_rs_c::Tensor_scalar_type(self.tensor_ref()) };
        ScalarType::from_c_scalar_type(scalar_type)
    }

    /// Returns the size in bytes of one element of the tensor.
    pub fn element_size(&self) -> isize {
        unsafe { et_rs_c::Tensor_element_size(self.tensor_ref()) }
    }

    /// Returns the sizes of the tensor at each dimension.
    pub fn sizes(&self) -> &[SizesType] {
        unsafe {
            let arr = et_rs_c::Tensor_sizes(self.tensor_ref());
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    /// Returns the order the dimensions are laid out in memory.
    pub fn dim_order(&self) -> &[DimOrderType] {
        unsafe {
            let arr = et_rs_c::Tensor_dim_order(self.tensor_ref());
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    /// Returns the strides of the tensor at each dimension.
    pub fn strides(&self) -> &[StridesType] {
        unsafe {
            let arr = et_rs_c::Tensor_strides(self.tensor_ref());
            std::slice::from_raw_parts(arr.Data, arr.Length)
        }
    }

    /// Returns a pointer of type S to the constant underlying data blob.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the type `S`.
    pub fn as_ptr<S: Scalar>(&self) -> *const S {
        assert_eq!(self.scalar_type(), Some(S::TYPE), "Invalid type");
        (unsafe { self.as_ptr_bytes() }) as *const S
    }

    /// Returns a pointer to the constant underlying data blob.
    ///
    /// # Safety
    ///
    /// The caller must access the values in the returned pointer according to the type of the tensor.
    pub unsafe fn as_ptr_bytes(&self) -> *const u8 {
        (unsafe { et_rs_c::Tensor_const_data_ptr(self.tensor_ref()) }) as *const u8
    }

    /// Get an array view of the tensor.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the type `S`.
    /// If the number of dimensions of the tensor does not match the number of dimensions of the type `Dim`.
    pub fn as_array<S: Scalar, Dim: Dimension>(&self) -> ArrayView<S, Dim> {
        let ndim = self.dim() as usize;
        let mut dim = Dim::zeros(ndim);
        let mut strides = Dim::zeros(ndim);
        let mut dim_order = Dim::zeros(ndim);
        for (i, d) in self.sizes().iter().enumerate() {
            dim[i] = *d as usize;
        }
        for (i, s) in self.strides().iter().enumerate() {
            strides[i] = *s as usize;
        }
        for (i, s) in self.dim_order().iter().enumerate() {
            dim_order[i] = *s as usize;
        }
        let ptr = self.as_ptr::<S>();
        unsafe { ArrayView::from_shape_ptr(dim.strides(strides), ptr) }.permuted_axes(dim_order)
    }

    /// Get an array view of the tensor with dynamic number of dimensions.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the type `S`.
    #[cfg(feature = "alloc")]
    pub fn as_array_dyn<S: Scalar>(&self) -> ArrayView<S, ndarray::IxDyn> {
        self.as_array()
    }
}
impl Destroy for et_c::Tensor {
    unsafe fn destroy(&mut self) {
        et_rs_c::Tensor_destructor(self)
    }
}
impl<'a, D: Data> Storable for TensorBase<'a, D> {
    type Storage = et_c::Tensor;
}

/// An immutable tensor that does not own the underlying data.
pub type Tensor<'a> = TensorBase<'a, View>;
impl<'a> Tensor<'a> {
    /// Create a new [`Tensor`] from a [`TensorImpl`].
    ///
    /// The underlying Cpp object is allocated on the heap, which is preferred on systems in which allocations are
    /// available.
    /// For an identical version that allocates the object on the stack use:
    /// ```rust,ignore
    /// let storage = executorch::storage!(Tensor);
    /// let tensor: Tensor = storage.new(tensor_impl);
    /// ```
    /// See `executorch::util::Storage` for more information.
    #[cfg(feature = "alloc")]
    pub fn new(tensor_impl: &'a TensorImpl) -> Self {
        Self::new_boxed(tensor_impl)
    }

    pub(crate) fn from_inner_ref(tensor: &'a et_c::Tensor) -> Self {
        Self(NonTriviallyMovable::from_ref(tensor), PhantomData)
    }
}
impl Storage<Tensor<'_>> {
    /// Create a new [`Tensor`] from a [`TensorImpl`] in the given storage.
    ///
    /// This function is identical to `Tensor::new`, but it allows to create the tensor on the stack.
    /// See `executorch::util::Storage` for more information.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<'a>(self: Pin<&'a mut Self>, tensor_impl: &'a TensorImpl) -> Tensor<'a> {
        Tensor::new_in_storage(tensor_impl, self)
    }
}

/// A mutable tensor that does not own the underlying data.
pub type TensorMut<'a> = TensorBase<'a, ViewMut>;
impl<'a> TensorMut<'a> {
    /// Create a new [`TensorMut`] from a [`TensorImplMut`].
    ///
    /// The underlying Cpp object is allocated on the heap, which is preferred on systems in which allocations are
    /// available.
    /// For an identical version that allocates the object on the stack use:
    /// ```rust,ignore
    /// let storage = executorch::storage!(TensorMut);
    /// let tensor: TensorMut = storage.new(tensor_impl);
    /// ```
    /// See `executorch::util::Storage` for more information.
    #[cfg(feature = "alloc")]
    pub fn new(tensor_impl: &'a mut TensorImplMut) -> Self {
        Self::new_boxed(tensor_impl)
    }

    /// Returns a mutable pointer of type S to the underlying data blob.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the type `S`.
    pub fn as_mut_ptr<S: Scalar>(&self) -> *mut S {
        assert_eq!(self.scalar_type(), Some(S::TYPE), "Invalid type");
        (unsafe { et_rs_c::Tensor_mutable_data_ptr(self.tensor_ref()) }) as *mut S
    }

    // /// Returns an immutable tensor pointing to the same data of this tensor.
    // pub fn as_tensor(&self) -> Tensor {
    //     Tensor::from_inner_ref(self.tensor_ref())
    // }

    // /// Converts this tensor into an immutable tensor.
    // pub fn into_tensor(self) -> Tensor<'a> {
    //     Tensor(self.0, self.1)
    // }

    /// Get a mutable array view of the tensor.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the type `S`.
    /// If the number of dimensions of the tensor does not match the number of dimensions of the type `D`.
    pub fn as_array_mut<S: Scalar, Dim: Dimension>(&mut self) -> ArrayViewMut<'a, S, Dim> {
        let ndim = self.dim() as usize;
        let mut dim = Dim::zeros(ndim);
        let mut strides = Dim::zeros(ndim);
        let mut dim_order = Dim::zeros(ndim);
        for (i, d) in self.sizes().iter().enumerate() {
            dim[i] = *d as usize;
        }
        for (i, s) in TensorBase::strides(self).iter().enumerate() {
            strides[i] = *s as usize;
        }
        for (i, s) in self.dim_order().iter().enumerate() {
            dim_order[i] = *s as usize;
        }
        let ptr = self.as_mut_ptr::<S>();
        unsafe { ArrayViewMut::from_shape_ptr(dim.strides(strides), ptr) }.permuted_axes(dim_order)
    }

    /// Get a mutable array view of the tensor with dynamic number of dimensions.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the type `S`.
    #[cfg(feature = "alloc")]
    pub fn as_array_mut_dyn<S: Scalar>(&mut self) -> ArrayViewMut<'a, S, ndarray::IxDyn> {
        self.as_array_mut()
    }
}
impl Storage<TensorMut<'_>> {
    /// Create a new [`TensorMut`] from a [`TensorImplMut`] in the given storage.
    ///
    /// This function is identical to `TensorMut::new`, but it allows to create the tensor on the stack.
    /// See `executorch::util::Storage` for more information.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<'a>(self: Pin<&'a mut Self>, tensor_impl: &'a TensorImplMut) -> TensorMut<'a> {
        TensorMut::new_in_storage(tensor_impl, self)
    }
}

/// A tensor implementation that does not own the underlying data.
///
/// This is a base class for [`TensorImpl`] and [`TensorImplMut`] and is not meant to be
/// used directly. It is used to provide a common API for both of them.
pub struct TensorImplBase<'a, D: Data>(et_c::TensorImpl, PhantomData<(&'a (), D)>);
impl<'a, D: Data> TensorImplBase<'a, D> {
    unsafe fn new<S: Scalar>(
        dim: usize,
        sizes: *const SizesType,
        data: *mut S,
        dim_order: *const DimOrderType,
        strides: *const StridesType,
    ) -> Self {
        debug_assert!(!sizes.is_null());
        debug_assert!(!data.is_null());
        debug_assert!(!dim_order.is_null());
        debug_assert!(!strides.is_null());
        let impl_ = unsafe {
            et_c::TensorImpl::new(
                S::TYPE.into_c_scalar_type(),
                dim as isize,
                sizes as *mut SizesType,
                data as *mut _,
                dim_order as *mut DimOrderType,
                strides as *mut StridesType,
                et_c::TensorShapeDynamism::STATIC,
            )
        };
        Self(impl_, PhantomData)
    }

    unsafe fn from_ptr_impl<S: Scalar>(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        let dim = sizes.len();
        assert_eq!(dim, dim_order.len());
        assert_eq!(dim, strides.len());
        Self::new(
            dim,
            sizes.as_ptr(),
            data,
            dim_order.as_ptr(),
            strides.as_ptr(),
        )
    }
}

impl<D: Data> Debug for TensorBase<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut st = f.debug_struct("Tensor");
        st.field("scalar_type", &self.scalar_type());

        fn add_data_field<D: Data, S: Scalar + Debug>(
            this: &TensorBase<D>,
            st: &mut std::fmt::DebugStruct,
        ) {
            match this.dim() {
                0 => st.field("data", &this.as_array::<S, ndarray::Ix0>()),
                1 => st.field("data", &this.as_array::<S, ndarray::Ix1>()),
                2 => st.field("data", &this.as_array::<S, ndarray::Ix2>()),
                3 => st.field("data", &this.as_array::<S, ndarray::Ix3>()),
                4 => st.field("data", &this.as_array::<S, ndarray::Ix4>()),
                5 => st.field("data", &this.as_array::<S, ndarray::Ix5>()),
                6 => st.field("data", &this.as_array::<S, ndarray::Ix6>()),
                _ => {
                    cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                        st.field("data", &this.as_array_dyn::<S>())
                    } else {
                        st.field("data", &"unsupported (too many dimensions)")
                    } }
                }
            };
        }
        fn add_data_field_unsupported(st: &mut std::fmt::DebugStruct) {
            st.field("data", &"unsupported");
        }
        match self.scalar_type() {
            Some(ScalarType::Byte) => add_data_field::<_, u8>(self, &mut st),
            Some(ScalarType::Char) => add_data_field::<_, i8>(self, &mut st),
            Some(ScalarType::Short) => add_data_field::<_, i16>(self, &mut st),
            Some(ScalarType::Int) => add_data_field::<_, i32>(self, &mut st),
            Some(ScalarType::Long) => add_data_field::<_, i64>(self, &mut st),
            Some(ScalarType::Half) => {
                cfg_if::cfg_if! { if #[cfg(feature = "f16")] {
                    add_data_field::<_, half::f16>(self, &mut st);
                } else {
                    add_data_field_unsupported(&mut st);
                } }
            }
            Some(ScalarType::Float) => add_data_field::<_, f32>(self, &mut st),
            Some(ScalarType::Double) => add_data_field::<_, f64>(self, &mut st),
            Some(ScalarType::ComplexHalf) => {
                cfg_if::cfg_if! { if #[cfg(all(feature = "complex", feature = "f16"))] {
                    add_data_field::<_, num_complex::Complex<half::f16>>(self, &mut st);
                } else {
                    add_data_field_unsupported(&mut st);
                } }
            }
            Some(ScalarType::ComplexFloat) => {
                cfg_if::cfg_if! { if #[cfg(feature = "complex")] {
                    add_data_field::<_, num_complex::Complex32>(self, &mut st);
                } else {
                    add_data_field_unsupported(&mut st);
                } }
            }
            Some(ScalarType::ComplexDouble) => {
                cfg_if::cfg_if! { if #[cfg(feature = "complex")] {
                    add_data_field::<_, num_complex::Complex64>(self, &mut st);
                } else {
                    add_data_field_unsupported(&mut st);
                } }
            }
            Some(ScalarType::Bool) => add_data_field::<_, bool>(self, &mut st),
            Some(ScalarType::QInt8) => add_data_field_unsupported(&mut st),
            Some(ScalarType::QUInt8) => add_data_field_unsupported(&mut st),
            Some(ScalarType::QInt32) => add_data_field_unsupported(&mut st),
            Some(ScalarType::BFloat16) => {
                cfg_if::cfg_if! { if #[cfg(feature = "f16")] {
                    add_data_field::<_, half::bf16>(self, &mut st);
                } else {
                    add_data_field_unsupported(&mut st);
                } }
            }
            Some(ScalarType::QUInt4x2) => add_data_field_unsupported(&mut st),
            Some(ScalarType::QUInt2x4) => add_data_field_unsupported(&mut st),
            Some(ScalarType::Bits1x8) => add_data_field_unsupported(&mut st),
            Some(ScalarType::Bits2x4) => add_data_field_unsupported(&mut st),
            Some(ScalarType::Bits4x2) => add_data_field_unsupported(&mut st),
            Some(ScalarType::Bits8) => add_data_field_unsupported(&mut st),
            Some(ScalarType::Bits16) => add_data_field_unsupported(&mut st),
            None => {
                st.field("data", &"None");
            }
        };
        st.finish()
    }
}

/// An immutable tensor implementation that does not own the underlying data.
pub type TensorImpl<'a> = TensorImplBase<'a, View>;
impl<'a> TensorImpl<'a> {
    /// Create a new TensorImpl from a pointer to the data.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    /// the tensor. The slice must be valid for the lifetime of the TensorImpl.
    /// * `data` - A pointer to the data of the tensor. The caller must ensure that the data is valid for the
    /// lifetime of the TensorImpl.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Safety
    ///
    /// The caller must ensure elements in the data can be safely accessed according to the scalar type, sizes,
    /// dim_order and strides of the tensor.
    /// The caller must ensure that the data is valid for the lifetime of the TensorImpl.
    pub unsafe fn from_ptr<S: Scalar>(
        sizes: &'a [SizesType],
        data: *const S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        unsafe { Self::from_ptr_impl(sizes, data as *mut S, dim_order, strides) }
    }
}

/// A mutable tensor implementation that does not own the underlying data.
pub type TensorImplMut<'a> = TensorImplBase<'a, ViewMut>;
impl<'a> TensorImplMut<'a> {
    /// Create a new TensorImplMut from a pointer to the data.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    /// the tensor. The slice must be valid for the lifetime of the TensorImplMut.
    /// * `data` - A pointer to the data of the tensor. The caller must ensure that the data is valid for the
    /// lifetime of the TensorImplMut.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Safety
    ///
    /// The caller must ensure elements in the data can be safely accessed and mutated according to the scalar type,
    /// sizes, dim_order and strides of the tensor.
    pub unsafe fn from_ptr<S: Scalar>(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        unsafe { Self::from_ptr_impl(sizes, data, dim_order, strides) }
    }
}

/// A marker trait that provide information about the data type of a [`TensorBase`] and [`TensorImplBase`]
pub trait Data {}
/// A marker trait extending [`Data`] that indicate that the data is mutable.
#[allow(dead_code)]
pub trait DataMut: Data {}

/// A marker type of viewed data of a tensor.
pub struct View {}
#[allow(dead_code)]
impl View {
    fn new() -> Self {
        Self {}
    }
}
impl Data for View {}

/// A marker type of mutable viewed data of a tensor.
pub struct ViewMut {}

#[allow(dead_code)]
impl ViewMut {
    fn new() -> Self {
        Self {}
    }
}
impl Data for ViewMut {}
impl DataMut for ViewMut {}

/// A wrapper around `ndarray::ArrayBase` that can be converted to [`TensorImplBase`].
///
/// The [`TensorImplBase`] struct does not own any of the data it points to along side the dimensions and strides. This
/// struct owns any additional data in addition to the underlying `ndarray::ArrayBase`, allowing to create a
/// [`TensorImplBase`] that points to it.
///
/// Use `as_tensor_impl()` and `as_tensor_impl_mut` to obtain a [`TensorImplBase`] pointing to this array data.
pub struct Array<A: Scalar, S: ndarray::RawData<Elem = A>, D: Dimension> {
    array: ArrayBase<S, D>,
    sizes: D::Arr<SizesType>,
    dim_order: D::Arr<DimOrderType>,
    strides: D::Arr<StridesType>,
}
impl<A: Scalar, S: ndarray::RawData<Elem = A>, D: Dimension> Array<A, S, D> {
    /// Create a new [`Array`] from an ndarray.
    pub fn new(array: ArrayBase<S, D>) -> Array<A, S, D> {
        let ndim = array.ndim();
        let mut sizes = D::Arr::zeros(ndim);
        let mut dim_order = D::Arr::zeros(ndim);
        let mut strides = D::Arr::zeros(ndim);
        for (i, d) in array.shape().iter().enumerate() {
            sizes.as_mut()[i] = *d as SizesType;
        }
        for (i, s) in ndarray::ArrayBase::strides(&array).iter().enumerate() {
            strides.as_mut()[i] = *s as StridesType;
        }
        for (i, s) in (0..ndim).enumerate() {
            dim_order.as_mut()[i] = s as DimOrderType;
        }
        Array {
            array,
            sizes,
            dim_order,
            strides,
        }
    }

    /// Create a [`TensorImpl`] pointing to this struct's data.
    ///
    /// The [`TensorImpl`] does not own the data or the sizes, dim order and strides of the tensor. This struct
    /// must outlive the [`TensorImpl`] created from it.
    pub fn as_tensor_impl(&self) -> TensorImpl {
        let impl_ = unsafe {
            et_c::TensorImpl::new(
                A::TYPE.into_c_scalar_type(),
                self.sizes.as_ref().len() as isize,
                self.sizes.as_ref().as_ptr() as *mut SizesType,
                self.array.as_ptr() as *mut _,
                self.dim_order.as_ref().as_ptr() as *mut DimOrderType,
                self.strides.as_ref().as_ptr() as *mut StridesType,
                et_c::TensorShapeDynamism::STATIC,
            )
        };
        TensorImplBase(impl_, PhantomData)
    }
}
impl<A: Scalar, S: ndarray::RawDataMut<Elem = A>, D: Dimension> Array<A, S, D> {
    /// Create a [`TensorImplMut`] pointing to this struct's data.
    ///
    /// The [`TensorImplMut`] does not own the data or the sizes, dim order and strides of the tensor. This struct
    /// must outlive the [`TensorImplMut`] created from it.
    pub fn as_tensor_impl_mut<'a>(&'a mut self) -> TensorImplMut<'a> {
        let tensor = self.as_tensor_impl();
        // Safety: TensorImpl has the same memory layout as TensorImplBase
        unsafe { std::mem::transmute::<TensorImpl<'a>, TensorImplMut<'a>>(tensor) }
    }
}
impl<A: Scalar, S: ndarray::RawData<Elem = A>, D: Dimension> AsRef<ArrayBase<S, D>>
    for Array<A, S, D>
{
    fn as_ref(&self) -> &ArrayBase<S, D> {
        &self.array
    }
}

/// An extension to `ndarray::Dimension` for dimensions used to convert to/from Tensors.
pub trait Dimension: ndarray::Dimension {
    /// The array type that holds the sizes, dim order and strides of the tensor.
    ///
    /// Can be either a fixed size array (supported without alloc) or a dynamic array (vector).
    type Arr<T: Clone + Copy + Default>: DimArr<T>;
}
impl<D: FixedSizeDim> Dimension for D {
    type Arr<T: Clone + Copy + Default> = D::Arr<T>;
}
#[cfg(feature = "alloc")]
impl Dimension for ndarray::IxDyn {
    type Arr<T: Clone + Copy + Default> = et_alloc::Vec<T>;
}

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
impl Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("TensorInfo")
            .field("sizes", &self.sizes())
            .field("dim_order", &self.dim_order())
            .field("scalar_type", &self.scalar_type())
            .field("nbytes", &self.nbytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array3, Ix3};

    use super::*;

    #[test]
    fn test_tensor_from_ptr() {
        // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
        let sizes = [2, 3];
        let data = [1, 2, 3, 4, 5, 6];
        let dim_order = [0, 1];
        let strides = [3, 1];
        let tensor_impl =
            unsafe { TensorImpl::from_ptr::<i32>(&sizes, data.as_ptr(), &dim_order, &strides) };
        let tensor = Tensor::new(&tensor_impl);

        assert_eq!(tensor.nbytes(), 24);
        assert_eq!(tensor.size(0), 2);
        assert_eq!(tensor.size(1), 3);
        assert_eq!(tensor.dim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.scalar_type(), Some(ScalarType::Int));
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[2, 3]);
        assert_eq!(tensor.dim_order(), &[0, 1]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.as_ptr(), data.as_ptr());
    }

    #[test]
    fn test_tensor_mut_from_ptr() {
        // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
        let sizes = [2, 3];
        let mut data = [1, 2, 3, 4, 5, 6];
        let dim_order = [0, 1];
        let strides = [3, 1];
        let mut tensor_impl = unsafe {
            TensorImplMut::from_ptr::<i32>(&sizes, data.as_mut_ptr(), &dim_order, &strides)
        };
        let tensor = TensorMut::new(&mut tensor_impl);

        assert_eq!(tensor.nbytes(), 24);
        assert_eq!(tensor.size(0), 2);
        assert_eq!(tensor.size(1), 3);
        assert_eq!(tensor.dim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.scalar_type(), Some(ScalarType::Int));
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[2, 3]);
        assert_eq!(tensor.dim_order(), &[0, 1]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.as_ptr(), data.as_ptr());
    }

    #[test]
    fn test_array_as_tensor() {
        // Create a 1D array and convert it to a tensor
        let array = Array::<i32, _, _>::new(arr1(&[1, 2, 3]));
        let tensor_impl = array.as_tensor_impl();
        let tensor = Tensor::new(&tensor_impl);
        assert_eq!(tensor.nbytes(), 12);
        assert_eq!(tensor.size(0), 3);
        assert_eq!(tensor.dim(), 1);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.scalar_type(), Some(ScalarType::Int));
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[3]);
        assert_eq!(tensor.dim_order(), &[0]);
        assert_eq!(tensor.strides(), &[1]);
        assert_eq!(tensor.as_ptr(), array.as_ref().as_ptr());

        // Create a 2D array and convert it to a tensor
        let array = Array::<f64, _, _>::new(arr2(&[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]]));
        let tensor_impl = array.as_tensor_impl();
        let tensor = Tensor::new(&tensor_impl);
        assert_eq!(tensor.nbytes(), 48);
        assert_eq!(tensor.size(0), 2);
        assert_eq!(tensor.size(1), 3);
        assert_eq!(tensor.dim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.scalar_type(), Some(ScalarType::Double));
        assert_eq!(tensor.element_size(), 8);
        assert_eq!(tensor.sizes(), &[2, 3]);
        assert_eq!(tensor.dim_order(), &[0, 1]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.as_ptr(), array.as_ref().as_ptr());
    }

    #[test]
    fn test_array_as_tensor_mut() {
        // Create a 1D array and convert it to a tensor
        let mut array = Array::<i32, _, _>::new(arr1(&[1, 2, 3]));
        let arr_ptr = array.as_ref().as_ptr();
        let mut tensor_impl = array.as_tensor_impl_mut();
        let tensor = TensorMut::new(&mut tensor_impl);
        assert_eq!(tensor.nbytes(), 12);
        assert_eq!(tensor.size(0), 3);
        assert_eq!(tensor.dim(), 1);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.scalar_type(), Some(ScalarType::Int));
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[3]);
        assert_eq!(tensor.dim_order(), &[0]);
        assert_eq!(tensor.strides(), &[1]);
        assert_eq!(tensor.as_ptr(), arr_ptr);

        // Create a 2D array and convert it to a tensor
        let mut array = Array::<f64, _, _>::new(arr2(&[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]]));
        let arr_ptr = array.as_ref().as_ptr();
        let mut tensor_impl = array.as_tensor_impl_mut();
        let tensor = TensorMut::new(&mut tensor_impl);
        assert_eq!(tensor.nbytes(), 48);
        assert_eq!(tensor.size(0), 2);
        assert_eq!(tensor.size(1), 3);
        assert_eq!(tensor.dim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.scalar_type(), Some(ScalarType::Double));
        assert_eq!(tensor.element_size(), 8);
        assert_eq!(tensor.sizes(), &[2, 3]);
        assert_eq!(tensor.dim_order(), &[0, 1]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.as_ptr(), arr_ptr);
    }

    #[test]
    fn test_tensor_as_array() {
        let arr1 = Array::new(Array3::<f32>::zeros((3, 6, 4)));
        let tensor_impl = arr1.as_tensor_impl();
        let tensor = Tensor::new(&tensor_impl);
        let arr2 = tensor.as_array::<f32, Ix3>();
        assert_eq!(arr1.as_ref(), arr2);
        assert_eq!(arr1.as_ref().strides(), arr2.strides());

        cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
            let arr1 = Array::new(arr1.as_ref().view().into_dyn());
            let tensor_impl = arr1.as_tensor_impl();
            let tensor = Tensor::new(&tensor_impl);
            let arr2 = tensor.as_array::<f32, ndarray::IxDyn>().into_shape_with_order(vec![18, 4]).unwrap();
            assert_eq!(arr1.as_ref().view().into_shape_with_order(vec![18, 4]).unwrap(), arr2);
            assert_eq!(arr2.strides(), [4, 1]);
        } }
    }

    #[test]
    fn test_tensor_as_array_mut() {
        let mut arr1 = Array::new(Array3::<f32>::zeros((3, 6, 4)));
        let arr1_clone = arr1.as_ref().clone();
        let mut tensor_impl = arr1.as_tensor_impl_mut();
        let mut tensor = TensorMut::new(&mut tensor_impl);
        let arr2 = tensor.as_array_mut::<f32, Ix3>();
        assert_eq!(arr1_clone, arr2);
        assert_eq!(arr1_clone.strides(), arr2.strides());

        cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
            let mut arr1 = arr1_clone.into_dyn();
            let arr1_clone = arr1.clone();
            let mut arr1 = Array::new(arr1.view_mut().into_shape_with_order((18, 4)).unwrap());
            let mut tensor_impl = arr1.as_tensor_impl_mut();
            let mut tensor = TensorMut::new(&mut tensor_impl);
            let arr2 = tensor.as_array_mut::<f32, ndarray::IxDyn>();
            assert_eq!(arr1_clone.view().into_shape_with_order(vec![18, 4]).unwrap(), arr2);
            assert_eq!(arr2.strides(), [4, 1]);
        } }
    }

    #[test]
    fn test_tensor_with_scalar_type() {
        fn test_scalar_type<S: Scalar>(data_allocator: impl FnOnce(usize) -> et_alloc::Vec<S>) {
            let sizes = [2, 4, 17];
            let data = data_allocator(2 * 4 * 17);
            let dim_order = [0, 1, 2];
            let strides = [4 * 17, 17, 1];
            let tensor_impl =
                unsafe { TensorImpl::from_ptr::<S>(&sizes, data.as_ptr(), &dim_order, &strides) };
            let tensor = Tensor::new(&tensor_impl);
            assert_eq!(tensor.scalar_type(), Some(S::TYPE));
        }

        test_scalar_type::<u8>(|size| vec![0; size]);
        test_scalar_type::<i8>(|size| vec![0; size]);
        test_scalar_type::<i16>(|size| vec![0; size]);
        test_scalar_type::<i32>(|size| vec![0; size]);
        test_scalar_type::<i64>(|size| vec![0; size]);
        test_scalar_type::<f32>(|size| vec![0.0; size]);
        test_scalar_type::<f64>(|size| vec![0.0; size]);
        test_scalar_type::<bool>(|size| vec![false; size]);
    }
}
