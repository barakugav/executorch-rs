use executorch_sys::ET_ScalarType as CScalarType;

use crate::util::{IntoCpp, IntoRust};

/// Data types (dtypes) that can be used as element types in Tensors.
///
/// The enum contain all the scalar types supported by the Cpp ExecuTorch library.
/// Not all of these types are supported by the Rust library, see [`Scalar`].
#[repr(i8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarType {
    /// 8-bit unsigned integer, `u8`
    Byte = CScalarType::ET_ScalarType_Byte as i8,
    /// 8-bit signed, integer, `i8`
    Char = CScalarType::ET_ScalarType_Char as i8,
    /// 16-bit signed integer, `i16`
    Short = CScalarType::ET_ScalarType_Short as i8,
    /// 32-bit signed integer, `i32`
    Int = CScalarType::ET_ScalarType_Int as i8,
    /// 64-bit signed integer, `i64`
    Long = CScalarType::ET_ScalarType_Long as i8,
    /// 16-bit floating point, [`executorch::scalar::f16`](`crate::scalar::f16`).
    Half = CScalarType::ET_ScalarType_Half as i8,
    /// 32-bit floating point, `f32`
    Float = CScalarType::ET_ScalarType_Float as i8,
    /// 64-bit floating point, `f64`
    Double = CScalarType::ET_ScalarType_Double as i8,
    /// 16-bit complex floating point, [`executorch::scalar::Complex<executorch::scalar::f16>`](`crate::scalar::Complex`).
    ComplexHalf = CScalarType::ET_ScalarType_ComplexHalf as i8,
    /// 32-bit complex floating point, [`executorch::scalar::Complex<f32>`](`crate::scalar::Complex`).
    ComplexFloat = CScalarType::ET_ScalarType_ComplexFloat as i8,
    /// 64-bit complex floating point, [`executorch::scalar::Complex<f64>`](`crate::scalar::Complex`).
    ComplexDouble = CScalarType::ET_ScalarType_ComplexDouble as i8,
    /// Boolean, `bool`
    Bool = CScalarType::ET_ScalarType_Bool as i8,
    /// 8-bit quantized integer, [`executorch::scalar::QInt8`](`crate::scalar::QInt8`).
    QInt8 = CScalarType::ET_ScalarType_QInt8 as i8,
    /// 8-bit quantized unsigned integer, [`executorch::scalar::QUInt8`](`crate::scalar::QUInt8`).
    QUInt8 = CScalarType::ET_ScalarType_QUInt8 as i8,
    /// 32-bit quantized integer, [`executorch::scalar::QInt32`](`crate::scalar::QInt32`).
    QInt32 = CScalarType::ET_ScalarType_QInt32 as i8,
    /// 16-bit floating point using the bfloat16 format, [`executorch::scalar::bf16`](`crate::scalar::bf16`).
    BFloat16 = CScalarType::ET_ScalarType_BFloat16 as i8,
    /// Two 4-bit unsigned quantized integers packed into a byte. [`executorch::scalar::QUInt4x2`](`crate::scalar::QUInt4x2`).
    QUInt4x2 = CScalarType::ET_ScalarType_QUInt4x2 as i8,
    /// Four 2-bit unsigned quantized integers packed into a byte. [`executorch::scalar::QUInt2x4`](`crate::scalar::QUInt2x4`).
    QUInt2x4 = CScalarType::ET_ScalarType_QUInt2x4 as i8,
    /// Eight 1-bit values packed into a byte. [`executorch::scalar::Bits1x8`](`crate::scalar::Bits1x8`).
    Bits1x8 = CScalarType::ET_ScalarType_Bits1x8 as i8,
    /// Four 2-bit values packed into a byte. [`executorch::scalar::Bits2x4`](`crate::scalar::Bits2x4`).
    Bits2x4 = CScalarType::ET_ScalarType_Bits2x4 as i8,
    /// Two 4-bit values packed into a byte. [`executorch::scalar::Bits4x2`](`crate::scalar::Bits4x2`).
    Bits4x2 = CScalarType::ET_ScalarType_Bits4x2 as i8,
    /// 8-bit bitfield (1 byte). [`executorch::scalar::Bits8`](`crate::scalar::Bits8`).
    Bits8 = CScalarType::ET_ScalarType_Bits8 as i8,
    /// 16-bit bitfield (2 bytes). [`executorch::scalar::Bits16`](`crate::scalar::Bits16`).
    Bits16 = CScalarType::ET_ScalarType_Bits16 as i8,
    /// 8-bit floating-point with 1 bit for the sign, 5 bits for the exponents, 2 bits for the mantissa.
    /// [`executorch::scalar::Float8_e5m2`](`crate::scalar::Float8_e5m2`).
    #[allow(non_camel_case_types)]
    Float8_e5m2 = CScalarType::ET_ScalarType_Float8_e5m2 as i8,
    /// 8-bit floating-point with 1 bit for the sign, 4 bits for the exponents, 3 bits for the mantissa,
    /// only nan values and no infinite values (FN).
    /// [`executorch::scalar::Float8_e4m3fn`](`crate::scalar::Float8_e4m3fn`).
    #[allow(non_camel_case_types)]
    Float8_e4m3fn = CScalarType::ET_ScalarType_Float8_e4m3fn as i8,
    /// 8-bit floating-point with 1 bit for the sign, 5 bits for the exponents, 2 bits for the mantissa,
    /// only nan values and no infinite values (FN), no negative zero (UZ).
    /// [`executorch::scalar::Float8_e5m2fnuz`](`crate::scalar::Float8_e5m2fnuz`).
    #[allow(non_camel_case_types)]
    Float8_e5m2fnuz = CScalarType::ET_ScalarType_Float8_e5m2fnuz as i8,
    /// 8-bit floating-point with 1 bit for the sign, 4 bits for the exponents, 3 bits for the mantissa,
    /// only nan values and no infinite values (FN), no negative zero (UZ).
    /// [`executorch::scalar::Float8_e4m3fnuz`](`crate::scalar::Float8_e4m3fnuz`).
    #[allow(non_camel_case_types)]
    Float8_e4m3fnuz = CScalarType::ET_ScalarType_Float8_e4m3fnuz as i8,
    /// 16-bit unsigned integer, `u16`
    UInt16 = CScalarType::ET_ScalarType_UInt16 as i8,
    /// 32-bit unsigned integer, `u32`
    UInt32 = CScalarType::ET_ScalarType_UInt32 as i8,
    /// 64-bit unsigned integer, `u64`
    UInt64 = CScalarType::ET_ScalarType_UInt64 as i8,
}
impl IntoRust for CScalarType {
    type RsType = ScalarType;
    fn rs(self) -> Self::RsType {
        match self {
            CScalarType::ET_ScalarType_Byte => ScalarType::Byte,
            CScalarType::ET_ScalarType_Char => ScalarType::Char,
            CScalarType::ET_ScalarType_Short => ScalarType::Short,
            CScalarType::ET_ScalarType_Int => ScalarType::Int,
            CScalarType::ET_ScalarType_Long => ScalarType::Long,
            CScalarType::ET_ScalarType_Half => ScalarType::Half,
            CScalarType::ET_ScalarType_Float => ScalarType::Float,
            CScalarType::ET_ScalarType_Double => ScalarType::Double,
            CScalarType::ET_ScalarType_ComplexHalf => ScalarType::ComplexHalf,
            CScalarType::ET_ScalarType_ComplexFloat => ScalarType::ComplexFloat,
            CScalarType::ET_ScalarType_ComplexDouble => ScalarType::ComplexDouble,
            CScalarType::ET_ScalarType_Bool => ScalarType::Bool,
            CScalarType::ET_ScalarType_QInt8 => ScalarType::QInt8,
            CScalarType::ET_ScalarType_QUInt8 => ScalarType::QUInt8,
            CScalarType::ET_ScalarType_QInt32 => ScalarType::QInt32,
            CScalarType::ET_ScalarType_BFloat16 => ScalarType::BFloat16,
            CScalarType::ET_ScalarType_QUInt4x2 => ScalarType::QUInt4x2,
            CScalarType::ET_ScalarType_QUInt2x4 => ScalarType::QUInt2x4,
            CScalarType::ET_ScalarType_Bits1x8 => ScalarType::Bits1x8,
            CScalarType::ET_ScalarType_Bits2x4 => ScalarType::Bits2x4,
            CScalarType::ET_ScalarType_Bits4x2 => ScalarType::Bits4x2,
            CScalarType::ET_ScalarType_Bits8 => ScalarType::Bits8,
            CScalarType::ET_ScalarType_Bits16 => ScalarType::Bits16,
            CScalarType::ET_ScalarType_Float8_e5m2 => ScalarType::Float8_e5m2,
            CScalarType::ET_ScalarType_Float8_e4m3fn => ScalarType::Float8_e4m3fn,
            CScalarType::ET_ScalarType_Float8_e5m2fnuz => ScalarType::Float8_e5m2fnuz,
            CScalarType::ET_ScalarType_Float8_e4m3fnuz => ScalarType::Float8_e4m3fnuz,
            CScalarType::ET_ScalarType_UInt16 => ScalarType::UInt16,
            CScalarType::ET_ScalarType_UInt32 => ScalarType::UInt32,
            CScalarType::ET_ScalarType_UInt64 => ScalarType::UInt64,
        }
    }
}
impl IntoCpp for ScalarType {
    type CppType = CScalarType;

    fn cpp(self) -> Self::CppType {
        match self {
            ScalarType::Byte => CScalarType::ET_ScalarType_Byte,
            ScalarType::Char => CScalarType::ET_ScalarType_Char,
            ScalarType::Short => CScalarType::ET_ScalarType_Short,
            ScalarType::Int => CScalarType::ET_ScalarType_Int,
            ScalarType::Long => CScalarType::ET_ScalarType_Long,
            ScalarType::Half => CScalarType::ET_ScalarType_Half,
            ScalarType::Float => CScalarType::ET_ScalarType_Float,
            ScalarType::Double => CScalarType::ET_ScalarType_Double,
            ScalarType::ComplexHalf => CScalarType::ET_ScalarType_ComplexHalf,
            ScalarType::ComplexFloat => CScalarType::ET_ScalarType_ComplexFloat,
            ScalarType::ComplexDouble => CScalarType::ET_ScalarType_ComplexDouble,
            ScalarType::Bool => CScalarType::ET_ScalarType_Bool,
            ScalarType::QInt8 => CScalarType::ET_ScalarType_QInt8,
            ScalarType::QUInt8 => CScalarType::ET_ScalarType_QUInt8,
            ScalarType::QInt32 => CScalarType::ET_ScalarType_QInt32,
            ScalarType::BFloat16 => CScalarType::ET_ScalarType_BFloat16,
            ScalarType::QUInt4x2 => CScalarType::ET_ScalarType_QUInt4x2,
            ScalarType::QUInt2x4 => CScalarType::ET_ScalarType_QUInt2x4,
            ScalarType::Bits1x8 => CScalarType::ET_ScalarType_Bits1x8,
            ScalarType::Bits2x4 => CScalarType::ET_ScalarType_Bits2x4,
            ScalarType::Bits4x2 => CScalarType::ET_ScalarType_Bits4x2,
            ScalarType::Bits8 => CScalarType::ET_ScalarType_Bits8,
            ScalarType::Bits16 => CScalarType::ET_ScalarType_Bits16,
            ScalarType::Float8_e5m2 => CScalarType::ET_ScalarType_Float8_e5m2,
            ScalarType::Float8_e4m3fn => CScalarType::ET_ScalarType_Float8_e4m3fn,
            ScalarType::Float8_e5m2fnuz => CScalarType::ET_ScalarType_Float8_e5m2fnuz,
            ScalarType::Float8_e4m3fnuz => CScalarType::ET_ScalarType_Float8_e4m3fnuz,
            ScalarType::UInt16 => CScalarType::ET_ScalarType_UInt16,
            ScalarType::UInt32 => CScalarType::ET_ScalarType_UInt32,
            ScalarType::UInt64 => CScalarType::ET_ScalarType_UInt64,
        }
    }
}

/// A trait for types that can be used as scalar types in Tensors.
pub trait Scalar: 'static {
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
impl_scalar!(crate::scalar::f16, Half);
impl_scalar!(f32, Float);
impl_scalar!(f64, Double);
impl_scalar!(crate::scalar::Complex<crate::scalar::f16>, ComplexHalf);
impl_scalar!(crate::scalar::Complex<f32>, ComplexFloat);
impl_scalar!(crate::scalar::Complex<f64>, ComplexDouble);
impl_scalar!(bool, Bool);
impl_scalar!(crate::scalar::QInt8, QInt8);
impl_scalar!(crate::scalar::QUInt8, QUInt8);
impl_scalar!(crate::scalar::QInt32, QInt32);
impl_scalar!(crate::scalar::bf16, BFloat16);
impl_scalar!(crate::scalar::QUInt4x2, QUInt4x2);
impl_scalar!(crate::scalar::QUInt2x4, QUInt2x4);
impl_scalar!(crate::scalar::Bits1x8, Bits1x8);
impl_scalar!(crate::scalar::Bits2x4, Bits2x4);
impl_scalar!(crate::scalar::Bits4x2, Bits4x2);
impl_scalar!(crate::scalar::Bits8, Bits8);
impl_scalar!(crate::scalar::Bits16, Bits16);
impl_scalar!(crate::scalar::Float8_e5m2, Float8_e5m2);
impl_scalar!(crate::scalar::Float8_e4m3fn, Float8_e4m3fn);
impl_scalar!(crate::scalar::Float8_e5m2fnuz, Float8_e5m2fnuz);
impl_scalar!(crate::scalar::Float8_e4m3fnuz, Float8_e4m3fnuz);
impl_scalar!(u16, UInt16);
impl_scalar!(u32, UInt32);
impl_scalar!(u64, UInt64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_cpp_conversions() {
        type CType = CScalarType;
        type RType = ScalarType;
        let scalars = [
            (CType::ET_ScalarType_Byte, RType::Byte),
            (CType::ET_ScalarType_Char, RType::Char),
            (CType::ET_ScalarType_Short, RType::Short),
            (CType::ET_ScalarType_Int, RType::Int),
            (CType::ET_ScalarType_Long, RType::Long),
            (CType::ET_ScalarType_Half, RType::Half),
            (CType::ET_ScalarType_Float, RType::Float),
            (CType::ET_ScalarType_Double, RType::Double),
            (CType::ET_ScalarType_ComplexHalf, RType::ComplexHalf),
            (CType::ET_ScalarType_ComplexFloat, RType::ComplexFloat),
            (CType::ET_ScalarType_ComplexDouble, RType::ComplexDouble),
            (CType::ET_ScalarType_Bool, RType::Bool),
            (CType::ET_ScalarType_QInt8, RType::QInt8),
            (CType::ET_ScalarType_QUInt8, RType::QUInt8),
            (CType::ET_ScalarType_QInt32, RType::QInt32),
            (CType::ET_ScalarType_BFloat16, RType::BFloat16),
            (CType::ET_ScalarType_QUInt4x2, RType::QUInt4x2),
            (CType::ET_ScalarType_QUInt2x4, RType::QUInt2x4),
            (CType::ET_ScalarType_Bits1x8, RType::Bits1x8),
            (CType::ET_ScalarType_Bits2x4, RType::Bits2x4),
            (CType::ET_ScalarType_Bits4x2, RType::Bits4x2),
            (CType::ET_ScalarType_Bits8, RType::Bits8),
            (CType::ET_ScalarType_Bits16, RType::Bits16),
            (CType::ET_ScalarType_Float8_e5m2, RType::Float8_e5m2),
            (CType::ET_ScalarType_Float8_e4m3fn, RType::Float8_e4m3fn),
            (CType::ET_ScalarType_Float8_e5m2fnuz, RType::Float8_e5m2fnuz),
            (CType::ET_ScalarType_Float8_e4m3fnuz, RType::Float8_e4m3fnuz),
            (CType::ET_ScalarType_UInt16, RType::UInt16),
            (CType::ET_ScalarType_UInt32, RType::UInt32),
            (CType::ET_ScalarType_UInt64, RType::UInt64),
        ];
        for (cpp, rust) in scalars {
            assert_eq!(cpp.rs(), rust);
            assert_eq!(rust.cpp(), cpp);
        }
    }
}
