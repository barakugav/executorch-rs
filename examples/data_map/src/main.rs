#![deny(warnings)]

use std::path::{Path, PathBuf};

use executorch::data_loader::FileDataLoader;
use executorch::data_map::FlatTensorDataMap;
use executorch::evalue::{EValue, IntoEValue};
use executorch::memory::{BufferMemoryAllocator, HierarchicalAllocator, MemoryManager};
use executorch::module::Module;
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::TensorPtr;
use executorch::util::Span;
use ndarray::array;

fn main() {
    unsafe { executorch::platform::pal_init() };

    println!("Running the model using a Module...");
    main_module();

    println!("Running the model using a Program...");
    main_program();
}

fn main_module() {
    let (model_path, data_file) = model_files();
    let mut module = Module::new(&model_path, &[&data_file], None, None);

    let data = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let input = TensorPtr::from_array(data).unwrap();
    let inputs = [input.into_evalue()];

    let outputs = module.forward(&inputs).unwrap();
    let [output]: [EValue; 1] = outputs.try_into().expect("not a single output");
    let output = output.as_tensor().into_typed::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(array![[5.0, 8.0], [11.0, 14.0]], output.as_array());
}

fn main_program() {
    let (model_path, data_file) = model_files();

    let mut buffer = [0_u8; 4096];
    let allocator = BufferMemoryAllocator::new(&mut buffer);

    let data_loader = FileDataLoader::from_path(&model_path, None).unwrap();
    let program =
        Program::load(&data_loader, Some(ProgramVerification::InternalConsistency)).unwrap();

    let method_meta = program.method_meta(c"forward").unwrap();

    let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
    let planned_arenas = allocator
        .allocate_arr_fn(num_memory_planned_buffers, |idx| {
            let buf_size = method_meta.memory_planned_buffer_size(idx).unwrap();
            let buf = allocator
                .allocate_arr::<u8>(buf_size)
                .expect("Failed to allocate buffer");
            Span::from_slice(buf)
        })
        .unwrap();

    let mut planned_memory = HierarchicalAllocator::new(planned_arenas);

    let memory_manager = MemoryManager::new(&allocator, Some(&mut planned_memory), None);

    let data_map_loader = FileDataLoader::from_path(&data_file, None).unwrap();
    let data_map = FlatTensorDataMap::load(&data_map_loader).unwrap();
    let mut method = program
        .load_method(c"forward", &memory_manager, None, Some(&data_map))
        .unwrap();

    let data = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let input = TensorPtr::from_array(data).unwrap();
    let input = input.into_evalue();

    let mut method_exe = method.start_execution();

    method_exe.set_input(&input, 0).unwrap();

    let outputs = method_exe.execute().unwrap();
    let output = outputs.get(0);
    let output = output.as_tensor().into_typed::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(array![[5.0, 8.0], [11.0, 14.0]], output.as_array());
}

fn model_files() -> (PathBuf, PathBuf) {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("model");
    let model_path = model_dir.join("model.pte");
    let data_file = model_dir.join("_default_external_constant.ptd");
    (model_path, data_file)
}
