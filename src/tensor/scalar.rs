use executorch_sys::ScalarType as CScalarType;

use crate::scalar::{bf16, f16, Complex};
use crate::util::{IntoCpp, IntoRust};

/// Data types (dtypes) that can be used as element types in Tensors.
///
/// The enum contain all the scalar types supported by the Cpp ExecuTorch library.
/// Not all of these types are supported by the Rust library, see [`Scalar`].
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarType {
    /// 8-bit unsigned integer, `u8`
    Byte = CScalarType::ScalarType_Byte as u8,
    /// 8-bit signed, integer, `i8`
    Char = CScalarType::ScalarType_Char as u8,
    /// 16-bit signed integer, `i16`
    Short = CScalarType::ScalarType_Short as u8,
    /// 32-bit signed integer, `i32`
    Int = CScalarType::ScalarType_Int as u8,
    /// 64-bit signed integer, `i64`
    Long = CScalarType::ScalarType_Long as u8,
    /// 16-bit floating point, [`executorch::scalar::f16`](`crate::scalar::f16`).
    Half = CScalarType::ScalarType_Half as u8,
    /// 32-bit floating point, `f32`
    Float = CScalarType::ScalarType_Float as u8,
    /// 64-bit floating point, `f64`
    Double = CScalarType::ScalarType_Double as u8,
    /// 16-bit complex floating point, [`executorch::scalar::Complex<executorch::scalar::f16>`](`crate::scalar::Complex`).
    ComplexHalf = CScalarType::ScalarType_ComplexHalf as u8,
    /// 32-bit complex floating point, [`executorch::scalar::Complex<f32>`](`crate::scalar::Complex`).
    ComplexFloat = CScalarType::ScalarType_ComplexFloat as u8,
    /// 64-bit complex floating point, [`executorch::scalar::Complex<f64>`](`crate::scalar::Complex`).
    ComplexDouble = CScalarType::ScalarType_ComplexDouble as u8,
    /// Boolean, `bool`
    Bool = CScalarType::ScalarType_Bool as u8,
    /// **\[Unsupported\]** 8-bit quantized integer
    QInt8 = CScalarType::ScalarType_QInt8 as u8,
    /// **\[Unsupported\]** 8-bit quantized unsigned integer
    QUInt8 = CScalarType::ScalarType_QUInt8 as u8,
    /// **\[Unsupported\]** 32-bit quantized integer
    QInt32 = CScalarType::ScalarType_QInt32 as u8,
    /// 16-bit floating point using the bfloat16 format, [`executorch::scalar::bf16`](`crate::scalar::bf16`).
    BFloat16 = CScalarType::ScalarType_BFloat16 as u8,
    /// **\[Unsupported\]**
    QUInt4x2 = CScalarType::ScalarType_QUInt4x2 as u8,
    /// **\[Unsupported\]**
    QUInt2x4 = CScalarType::ScalarType_QUInt2x4 as u8,
    /// **\[Unsupported\]**
    Bits1x8 = CScalarType::ScalarType_Bits1x8 as u8,
    /// **\[Unsupported\]**
    Bits2x4 = CScalarType::ScalarType_Bits2x4 as u8,
    /// **\[Unsupported\]**
    Bits4x2 = CScalarType::ScalarType_Bits4x2 as u8,
    /// **\[Unsupported\]**
    Bits8 = CScalarType::ScalarType_Bits8 as u8,
    /// **\[Unsupported\]**
    Bits16 = CScalarType::ScalarType_Bits16 as u8,
    /// **\[Unsupported\]**
    #[allow(non_camel_case_types)]
    Float8_e5m2 = CScalarType::ScalarType_Float8_e5m2 as u8,
    /// **\[Unsupported\]**
    #[allow(non_camel_case_types)]
    Float8_e4m3fn = CScalarType::ScalarType_Float8_e4m3fn as u8,
    /// **\[Unsupported\]**
    #[allow(non_camel_case_types)]
    Float8_e5m2fnuz = CScalarType::ScalarType_Float8_e5m2fnuz as u8,
    /// **\[Unsupported\]**
    #[allow(non_camel_case_types)]
    Float8_e4m3fnuz = CScalarType::ScalarType_Float8_e4m3fnuz as u8,
    /// 16-bit unsigned integer, `u16`
    UInt16 = CScalarType::ScalarType_UInt16 as u8,
    /// 32-bit unsigned integer, `u32`
    UInt32 = CScalarType::ScalarType_UInt32 as u8,
    /// 64-bit unsigned integer, `u64`
    UInt64 = CScalarType::ScalarType_UInt64 as u8,
}
impl IntoRust for CScalarType {
    type RsType = ScalarType;
    fn rs(self) -> Self::RsType {
        match self {
            CScalarType::ScalarType_Byte => ScalarType::Byte,
            CScalarType::ScalarType_Char => ScalarType::Char,
            CScalarType::ScalarType_Short => ScalarType::Short,
            CScalarType::ScalarType_Int => ScalarType::Int,
            CScalarType::ScalarType_Long => ScalarType::Long,
            CScalarType::ScalarType_Half => ScalarType::Half,
            CScalarType::ScalarType_Float => ScalarType::Float,
            CScalarType::ScalarType_Double => ScalarType::Double,
            CScalarType::ScalarType_ComplexHalf => ScalarType::ComplexHalf,
            CScalarType::ScalarType_ComplexFloat => ScalarType::ComplexFloat,
            CScalarType::ScalarType_ComplexDouble => ScalarType::ComplexDouble,
            CScalarType::ScalarType_Bool => ScalarType::Bool,
            CScalarType::ScalarType_QInt8 => ScalarType::QInt8,
            CScalarType::ScalarType_QUInt8 => ScalarType::QUInt8,
            CScalarType::ScalarType_QInt32 => ScalarType::QInt32,
            CScalarType::ScalarType_BFloat16 => ScalarType::BFloat16,
            CScalarType::ScalarType_QUInt4x2 => ScalarType::QUInt4x2,
            CScalarType::ScalarType_QUInt2x4 => ScalarType::QUInt2x4,
            CScalarType::ScalarType_Bits1x8 => ScalarType::Bits1x8,
            CScalarType::ScalarType_Bits2x4 => ScalarType::Bits2x4,
            CScalarType::ScalarType_Bits4x2 => ScalarType::Bits4x2,
            CScalarType::ScalarType_Bits8 => ScalarType::Bits8,
            CScalarType::ScalarType_Bits16 => ScalarType::Bits16,
            CScalarType::ScalarType_Float8_e5m2 => ScalarType::Float8_e5m2,
            CScalarType::ScalarType_Float8_e4m3fn => ScalarType::Float8_e4m3fn,
            CScalarType::ScalarType_Float8_e5m2fnuz => ScalarType::Float8_e5m2fnuz,
            CScalarType::ScalarType_Float8_e4m3fnuz => ScalarType::Float8_e4m3fnuz,
            CScalarType::ScalarType_UInt16 => ScalarType::UInt16,
            CScalarType::ScalarType_UInt32 => ScalarType::UInt32,
            CScalarType::ScalarType_UInt64 => ScalarType::UInt64,
        }
    }
}
impl IntoCpp for ScalarType {
    type CppType = CScalarType;

    fn cpp(self) -> Self::CppType {
        match self {
            ScalarType::Byte => CScalarType::ScalarType_Byte,
            ScalarType::Char => CScalarType::ScalarType_Char,
            ScalarType::Short => CScalarType::ScalarType_Short,
            ScalarType::Int => CScalarType::ScalarType_Int,
            ScalarType::Long => CScalarType::ScalarType_Long,
            ScalarType::Half => CScalarType::ScalarType_Half,
            ScalarType::Float => CScalarType::ScalarType_Float,
            ScalarType::Double => CScalarType::ScalarType_Double,
            ScalarType::ComplexHalf => CScalarType::ScalarType_ComplexHalf,
            ScalarType::ComplexFloat => CScalarType::ScalarType_ComplexFloat,
            ScalarType::ComplexDouble => CScalarType::ScalarType_ComplexDouble,
            ScalarType::Bool => CScalarType::ScalarType_Bool,
            ScalarType::QInt8 => CScalarType::ScalarType_QInt8,
            ScalarType::QUInt8 => CScalarType::ScalarType_QUInt8,
            ScalarType::QInt32 => CScalarType::ScalarType_QInt32,
            ScalarType::BFloat16 => CScalarType::ScalarType_BFloat16,
            ScalarType::QUInt4x2 => CScalarType::ScalarType_QUInt4x2,
            ScalarType::QUInt2x4 => CScalarType::ScalarType_QUInt2x4,
            ScalarType::Bits1x8 => CScalarType::ScalarType_Bits1x8,
            ScalarType::Bits2x4 => CScalarType::ScalarType_Bits2x4,
            ScalarType::Bits4x2 => CScalarType::ScalarType_Bits4x2,
            ScalarType::Bits8 => CScalarType::ScalarType_Bits8,
            ScalarType::Bits16 => CScalarType::ScalarType_Bits16,
            ScalarType::Float8_e5m2 => CScalarType::ScalarType_Float8_e5m2,
            ScalarType::Float8_e4m3fn => CScalarType::ScalarType_Float8_e4m3fn,
            ScalarType::Float8_e5m2fnuz => CScalarType::ScalarType_Float8_e5m2fnuz,
            ScalarType::Float8_e4m3fnuz => CScalarType::ScalarType_Float8_e4m3fnuz,
            ScalarType::UInt16 => CScalarType::ScalarType_UInt16,
            ScalarType::UInt32 => CScalarType::ScalarType_UInt32,
            ScalarType::UInt64 => CScalarType::ScalarType_UInt64,
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
impl_scalar!(f16, Half);
impl_scalar!(f32, Float);
impl_scalar!(f64, Double);
impl_scalar!(Complex<f16>, ComplexHalf);
impl_scalar!(Complex<f32>, ComplexFloat);
impl_scalar!(Complex<f64>, ComplexDouble);
impl_scalar!(bool, Bool);
impl_scalar!(bf16, BFloat16);
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
            (CType::ScalarType_Byte, RType::Byte),
            (CType::ScalarType_Char, RType::Char),
            (CType::ScalarType_Short, RType::Short),
            (CType::ScalarType_Int, RType::Int),
            (CType::ScalarType_Long, RType::Long),
            (CType::ScalarType_Half, RType::Half),
            (CType::ScalarType_Float, RType::Float),
            (CType::ScalarType_Double, RType::Double),
            (CType::ScalarType_ComplexHalf, RType::ComplexHalf),
            (CType::ScalarType_ComplexFloat, RType::ComplexFloat),
            (CType::ScalarType_ComplexDouble, RType::ComplexDouble),
            (CType::ScalarType_Bool, RType::Bool),
            (CType::ScalarType_QInt8, RType::QInt8),
            (CType::ScalarType_QUInt8, RType::QUInt8),
            (CType::ScalarType_QInt32, RType::QInt32),
            (CType::ScalarType_BFloat16, RType::BFloat16),
            (CType::ScalarType_QUInt4x2, RType::QUInt4x2),
            (CType::ScalarType_QUInt2x4, RType::QUInt2x4),
            (CType::ScalarType_Bits1x8, RType::Bits1x8),
            (CType::ScalarType_Bits2x4, RType::Bits2x4),
            (CType::ScalarType_Bits4x2, RType::Bits4x2),
            (CType::ScalarType_Bits8, RType::Bits8),
            (CType::ScalarType_Bits16, RType::Bits16),
            (CType::ScalarType_Float8_e5m2, RType::Float8_e5m2),
            (CType::ScalarType_Float8_e4m3fn, RType::Float8_e4m3fn),
            (CType::ScalarType_Float8_e5m2fnuz, RType::Float8_e5m2fnuz),
            (CType::ScalarType_Float8_e4m3fnuz, RType::Float8_e4m3fnuz),
            (CType::ScalarType_UInt16, RType::UInt16),
            (CType::ScalarType_UInt32, RType::UInt32),
            (CType::ScalarType_UInt64, RType::UInt64),
        ];
        for (cpp, rust) in scalars {
            assert_eq!(cpp.rs(), rust);
            assert_eq!(rust.cpp(), cpp);
        }
    }
}
