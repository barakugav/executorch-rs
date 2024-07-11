use executorch::et_c::util::FileDataLoader;
use executorch::util::IntoRust;
use executorch::{c_link, et_c, et_rs_c};
use std::{ffi::CString, mem::ManuallyDrop, ptr, vec};

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        .init();

    unsafe { c_link::executorch_c::root::et_pal_init() };

    let model_path = CString::new("model.pte").unwrap();
    let mut file_data_loader = unsafe { FileDataLoader::from(model_path.as_ptr(), 16) }
        .rs()
        .unwrap();
    let file_data_loader_ptr: *mut et_c::util::FileDataLoader = &mut file_data_loader as *mut _;
    let data_loader_ptr: *mut et_c::DataLoader = file_data_loader_ptr as *mut _;

    let program = unsafe {
        et_c::Program::load(
            data_loader_ptr,
            et_c::Program_Verification_InternalConsistency,
        )
    }
    .rs()
    .unwrap();

    let method_name = CString::new("forward").unwrap();
    let method_meta = unsafe { et_rs_c::Program_method_meta(&program, method_name.as_ptr()) }
        .rs()
        .unwrap();

    let num_memory_planned_buffers = unsafe { method_meta.num_memory_planned_buffers() };
    let mut planned_buffers: Vec<Vec<u8>> = vec![];
    let mut planned_arenas: Vec<et_c::Span<u8>> = vec![];
    for idx in 0..num_memory_planned_buffers {
        let buffer_size =
            unsafe { et_rs_c::MethodMeta_memory_planned_buffer_size(&method_meta, idx) }
                .rs()
                .unwrap() as usize;
        planned_buffers.push(vec![0; buffer_size]);
        planned_arenas.push(et_c::Span {
            data_: planned_buffers.last_mut().unwrap().as_mut_ptr(),
            length_: planned_buffers.last().unwrap().len(),
            _phantom_0: std::marker::PhantomData,
        });
    }

    let mut planned_memory = et_c::HierarchicalAllocator {
        span_array_: (0..16)
            .map(|_| et_c::Span {
                data_: ptr::null_mut(),
                length_: 0,
                _phantom_0: std::marker::PhantomData,
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        buffers_: et_c::Span {
            data_: planned_arenas.as_mut_ptr(),
            length_: planned_arenas.len(),
            _phantom_0: std::marker::PhantomData,
        },
    };

    let mut method_allocator = unsafe { et_rs_c::MallocMemoryAllocator_new() };
    let method_allocator_ptr: *mut et_c::util::MallocMemoryAllocator =
        &mut method_allocator as *mut _;
    let method_allocator_ptr: *mut et_c::MemoryAllocator = method_allocator_ptr as *mut _;

    let mut memory_manager = et_c::MemoryManager {
        method_allocator_: method_allocator_ptr,
        planned_memory_: &mut planned_memory,
        temp_allocator_: ptr::null_mut(),
    };

    let mut method =
        unsafe { program.load_method(method_name.as_ptr(), &mut memory_manager, ptr::null_mut()) }
            .rs()
            .unwrap();

    let mut data1 = vec![1.0_f32; 1];
    let mut sizes1 = [1];
    let mut data_order1 = [0];
    let mut input_tensor1 = unsafe {
        et_c::TensorImpl::new(
            et_c::ScalarType_Float,
            sizes1.len() as isize,
            sizes1.as_mut_ptr(),
            data1.as_mut_ptr() as *mut _,
            data_order1.as_mut_ptr(),
            ptr::null_mut(),
            et_c::TensorShapeDynamism_STATIC,
        )
    };
    let input_evalue1 = et_c::EValue {
        payload: et_c::EValue_Payload {
            as_tensor: ManuallyDrop::new(et_c::Tensor {
                impl_: &mut input_tensor1,
            }),
        },
        tag: et_c::Tag_Tensor,
    };
    let mut data2 = vec![1.0_f32; 1];
    let mut sizes2 = [1];
    let mut data_order2 = [0];
    let mut input_tensor2 = unsafe {
        et_c::TensorImpl::new(
            et_c::ScalarType_Float,
            sizes2.len() as isize,
            sizes2.as_mut_ptr(),
            data2.as_mut_ptr() as *mut _,
            data_order2.as_mut_ptr(),
            ptr::null_mut(),
            et_c::TensorShapeDynamism_STATIC,
        )
    };
    let input_evalue2 = et_c::EValue {
        payload: et_c::EValue_Payload {
            as_tensor: ManuallyDrop::new(et_c::Tensor {
                impl_: &mut input_tensor2,
            }),
        },
        tag: et_c::Tag_Tensor,
    };
    unsafe { method.set_input(&input_evalue1, 0) }.rs().unwrap();
    unsafe { method.set_input(&input_evalue2, 1) }.rs().unwrap();

    unsafe { method.execute() }.rs().unwrap();
    let output = unsafe { method.get_output(0) };
    assert!(unsafe { (*output).tag } == et_c::Tag_Tensor);
}
