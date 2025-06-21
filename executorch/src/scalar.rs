//! Custom scalar types that can be used in tensors.
//!
//! Half precision floating point types are provided by the `half` feature if enabled, otherwise a simple wrappers
//! around `u16` are provided without any arithmetic operations.
//!
//! Complex numbers are provided by the `num-complex` feature if enabled, otherwise a simple struct with real and
//! imaginary parts is provided without any arithmetic operations.

macro_rules! scalar_type {
    ($(#[$outer:meta])* $name:ident, $repr:ty) => {
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(transparent)]
        $(#[$outer])*
        pub struct $name($repr);
        impl $name {
            #[doc = concat!("Creates a new `", stringify!($name), "` from its raw bit representation.")]
            pub fn from_bits(bits: $repr) -> Self {
                Self(bits)
            }
            #[doc = concat!("Get the raw bit representation of the `", stringify!($name), "`.")]
            pub fn bits(&self) -> $repr {
                self.0
            }
        }
    };
}

cfg_if::cfg_if! { if #[cfg(feature = "half")] {
    pub use half::f16;
    pub use half::bf16;
} else {
    scalar_type!(
        /// A 16-bit floating point type implementing the IEEE 754-2008 standard [`binary16`] a.k.a "half"
        /// format.
        ///
        /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
        /// Enable the `half` feature to get a fully functional `f16` type.
        #[allow(non_camel_case_types)]
        f16, u16
    );

    scalar_type!(
        /// A 16-bit floating point type implementing the [`bfloat16`] format.
        ///
        /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
        /// Enable the `half` feature to get a fully functional `bf16` type.
        #[allow(non_camel_case_types)]
        bf16, u16
    );
} }

cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
    pub use num_complex::Complex;
} else {
    /// A complex number in Cartesian form.
    ///
    /// Doesn't provide any arithmetic operations, but expose the real and imaginary parts.
    /// Enable the `num-complex` feature to get a fully functional `Complex` type.
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(C)]
    pub struct Complex<T> {
        /// Real portion of the complex number
        pub re: T,
        /// Imaginary portion of the complex number
        pub im: T,
    }
} }

scalar_type!(
    /// 8-bit quantized integer.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    QInt8, u8
);

scalar_type!(
    /// 8-bit unsigned quantized integer.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    QUInt8, u8
);

scalar_type!(
    /// 32-bit quantized integer.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    QInt32, u32
);

scalar_type!(
    /// Two 4-bit unsigned quantized integers packed into a byte.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    QUInt4x2, u8
);

scalar_type!(
    /// Four 2-bit unsigned quantized integers packed into a byte.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    QUInt2x4, u8
);

scalar_type!(
    /// Eight 1-bit values packed into a byte.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    Bits1x8, u8
);

scalar_type!(
    /// Four 2-bit values packed into a byte.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    Bits2x4, u8
);

scalar_type!(
    /// Two 4-bit values packed into a byte.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    Bits4x2, u8
);

scalar_type!(
    /// 8-bit bitfield (1 byte).
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    Bits8, u8
);

scalar_type!(
    /// 16-bit bitfield (2 bytes).
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    Bits16, u16
);

scalar_type!(
    /// 8-bit floating-point with 1 bit for the sign, 5 bits for the exponents, 2 bits for the mantissa.
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    #[allow(non_camel_case_types)]
    Float8_e5m2, u8
);

scalar_type!(
    /// 8-bit floating-point with 1 bit for the sign, 4 bits for the exponents, 3 bits for the mantissa,
    /// only nan values and no infinite values (FN).
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    #[allow(non_camel_case_types)]
    Float8_e4m3fn, u8
);

scalar_type!(
    /// 8-bit floating-point with 1 bit for the sign, 5 bits for the exponents, 2 bits for the mantissa,
    /// only nan values and no infinite values (FN), no negative zero (UZ).
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    #[allow(non_camel_case_types)]
    Float8_e5m2fnuz, u8
);

scalar_type!(
    /// 8-bit floating-point with 1 bit for the sign, 4 bits for the exponents, 3 bits for the mantissa,
    /// only nan values and no infinite values (FN), no negative zero (UZ).
    ///
    /// Does not provide any arithmetic operations, but can be converted to/from bits representation.
    #[allow(non_camel_case_types)]
    Float8_e4m3fnuz, u8
);
