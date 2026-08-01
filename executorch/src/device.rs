//! Device specification.

use executorch_sys as sys;

use crate::util::IntoRust;

/// An index representing a specific device; e.g. GPU 0 vs GPU 1.
pub type DeviceIndex = i8;

/// Represents the type of compute device.
/// Note: ExecuTorch Device is distinct from PyTorch Device.
#[repr(i8)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DeviceType {
    /// The host CPU.
    Cpu = sys::ET_DeviceType::ET_DeviceType_CPU as i8,
    /// A CUDA device.
    Cuda = sys::ET_DeviceType::ET_DeviceType_CUDA as i8,
}
impl IntoRust for sys::ET_DeviceType {
    type RsType = DeviceType;
    fn rs(self) -> DeviceType {
        match self {
            sys::ET_DeviceType::ET_DeviceType_CPU => DeviceType::Cpu,
            sys::ET_DeviceType::ET_DeviceType_CUDA => DeviceType::Cuda,
        }
    }
}

/// An abstraction for the compute device on which a tensor is located.
///
/// Tensors carry a Device to express where their underlying data resides
/// (e.g. CPU host memory vs CUDA device memory). The runtime uses this to
/// dispatch memory allocation to the appropriate device allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    type_: DeviceType,
    index: DeviceIndex,
}
impl Device {
    /// Construct a Device from its type and index.
    pub fn new(type_: DeviceType, index: DeviceIndex) -> Self {
        Self { type_, index }
    }

    /// Returns the type of device the tensor data resides on.
    pub fn type_(&self) -> DeviceType {
        self.type_
    }

    /// Returns the device index.
    pub fn index(&self) -> DeviceIndex {
        self.index
    }

    /// Returns true if the device is of CPU type.
    pub fn is_cpu(&self) -> bool {
        self.type_ == DeviceType::Cpu
    }
}
impl IntoRust for sys::ET_Device {
    type RsType = Device;
    fn rs(self) -> Device {
        Device {
            type_: self.type_.rs(),
            index: self.index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_roundtrip() {
        let dev = Device::new(DeviceType::Cpu, 0);
        assert_eq!(dev.type_(), DeviceType::Cpu);
        assert_eq!(dev.index(), 0);
        assert!(dev.is_cpu());
    }
}
