// This header is generated by build.rs during compilation, specifying which bindings should be
#include "executorch_rs_defines.h"

#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/memory_manager.h"
#include "executorch/runtime/core/hierarchical_allocator.h"

#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"
#if defined(EXECUTORCH_RS_EXTENSION_DATA_LOADER)
#include "executorch/extension/data_loader/file_data_loader.h"
#include "executorch/extension/data_loader/mmap_data_loader.h"
#include "executorch/extension/data_loader/buffer_data_loader.h"
#endif

#if defined(EXECUTORCH_RS_EXTENSION_MODULE)
#include "executorch/extension/module/module.h"
#endif

#include "executorch_rs_ext/api_utils.hpp"
