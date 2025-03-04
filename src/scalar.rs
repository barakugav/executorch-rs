//! Custom scalar types that can be used in tensors.
//!
//! Half precision floating point types are provided by the `half` feature if enabled, otherwise a simple wrappers
//! around `u16` are provided without any arithmetic operations.
//!
//! Complex numbers are provided by the `num-complex` feature if enabled, otherwise a simple struct with real and
//! imaginary parts is provided without any arithmetic operations.

cfg_if::cfg_if! { if #[cfg(feature = "half")] {
    pub use half::f16;
    pub use half::bf16;
} else {
    /// A 16-bit floating point type implementing the IEEE 754-2008 standard [`binary16`] a.k.a "half"
    /// format.
    ///
    /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
    /// Enable the `half` feature to get a fully functional `f16` type.
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    pub struct f16(u16);
    impl f16 {
        /// Creates a new `f16` from its raw bit representation.
        pub fn from_bits(bits: u16) -> Self {
            Self(bits)
        }
        /// Get the raw bit representation of the `f16`.
        pub fn bits(&self) -> u16 {
            self.0
        }
    }

    /// A 16-bit floating point type implementing the [`bfloat16`] format.
    ///
    /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
    /// Enable the `half` feature to get a fully functional `bf16` type.
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    pub struct bf16(u16);
    impl bf16 {
        /// Creates a new `bf16` from its raw bit representation.
        pub fn from_bits(bits: u16) -> Self {
            Self(bits)
        }
        /// Get the raw bit representation of the `bf16`.
        pub fn bits(&self) -> u16 {
            self.0
        }
    }
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
