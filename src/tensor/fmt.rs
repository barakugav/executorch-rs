#[cfg(feature = "ndarray")]
mod array_based {
    use crate::tensor::{Data, Scalar, ScalarType, TensorBase};

    impl<D: Data> std::fmt::Debug for TensorBase<'_, D> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            fn add_data_field<S: Scalar + std::fmt::Debug, D: Data>(
                this: &TensorBase<D>,
                f: &mut std::fmt::Formatter,
            ) -> std::fmt::Result {
                let this = this.as_typed::<S>();
                match this.dim() {
                    0 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix0>(), f),
                    1 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix1>(), f),
                    2 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix2>(), f),
                    3 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix3>(), f),
                    4 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix4>(), f),
                    5 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix5>(), f),
                    6 => std::fmt::Debug::fmt(&this.as_array::<ndarray::Ix6>(), f),
                    _ => {
                        cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                            std::fmt::Debug::fmt( &this.as_array_dyn(), f)
                        } else {
                            write!(
                                f,
                                "[unsupported (too many dimensions) ...], shape={:?}, strides={:?}",
                                this.sizes(),
                                this.strides(),
                            )
                        } }
                    }
                }
            }

            match self.scalar_type() {
                ScalarType::Byte => add_data_field::<u8, _>(self, f),
                ScalarType::Char => add_data_field::<i8, _>(self, f),
                ScalarType::Short => add_data_field::<i16, _>(self, f),
                ScalarType::Int => add_data_field::<i32, _>(self, f),
                ScalarType::Long => add_data_field::<i64, _>(self, f),
                ScalarType::Half => add_data_field::<crate::scalar::f16, _>(self, f),
                ScalarType::Float => add_data_field::<f32, _>(self, f),
                ScalarType::Double => add_data_field::<f64, _>(self, f),
                ScalarType::ComplexHalf => {
                    add_data_field::<crate::scalar::Complex<crate::scalar::f16>, _>(self, f)
                }
                ScalarType::ComplexFloat => {
                    add_data_field::<crate::scalar::Complex<f32>, _>(self, f)
                }
                ScalarType::ComplexDouble => {
                    add_data_field::<crate::scalar::Complex<f64>, _>(self, f)
                }
                ScalarType::Bool => add_data_field::<bool, _>(self, f),
                ScalarType::QInt8 => add_data_field::<crate::scalar::QInt8, _>(self, f),
                ScalarType::QUInt8 => add_data_field::<crate::scalar::QUInt8, _>(self, f),
                ScalarType::QInt32 => add_data_field::<crate::scalar::QInt32, _>(self, f),
                ScalarType::BFloat16 => add_data_field::<crate::scalar::bf16, _>(self, f),
                ScalarType::QUInt4x2 => add_data_field::<crate::scalar::QUInt4x2, _>(self, f),
                ScalarType::QUInt2x4 => add_data_field::<crate::scalar::QUInt2x4, _>(self, f),
                ScalarType::Bits1x8 => add_data_field::<crate::scalar::Bits1x8, _>(self, f),
                ScalarType::Bits2x4 => add_data_field::<crate::scalar::Bits2x4, _>(self, f),
                ScalarType::Bits4x2 => add_data_field::<crate::scalar::Bits4x2, _>(self, f),
                ScalarType::Bits8 => add_data_field::<crate::scalar::Bits8, _>(self, f),
                ScalarType::Bits16 => add_data_field::<crate::scalar::Bits16, _>(self, f),
                ScalarType::Float8_e5m2 => add_data_field::<crate::scalar::Float8_e5m2, _>(self, f),
                ScalarType::Float8_e4m3fn => {
                    add_data_field::<crate::scalar::Float8_e4m3fn, _>(self, f)
                }
                ScalarType::Float8_e5m2fnuz => {
                    add_data_field::<crate::scalar::Float8_e5m2fnuz, _>(self, f)
                }
                ScalarType::Float8_e4m3fnuz => {
                    add_data_field::<crate::scalar::Float8_e4m3fnuz, _>(self, f)
                }
                ScalarType::UInt16 => add_data_field::<u16, _>(self, f),
                ScalarType::UInt32 => add_data_field::<u32, _>(self, f),
                ScalarType::UInt64 => add_data_field::<u64, _>(self, f),
            }
        }
    }
}
