import shutil
from pathlib import Path

import torch
from torch.export import export

from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower


class ModuleAddMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 3 * torch.ones(2, 2, dtype=torch.float)
        self.b = 2 * torch.ones(2, 2, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        out_1 = torch.mul(self.a, x)
        out_2 = torch.add(out_1, self.b)
        return out_2


model = ModuleAddMul()

exported_program = export(model, (torch.ones(2, 2, dtype=torch.float),))
executorch_program = to_edge_transform_and_lower(exported_program).to_executorch(
    ExecutorchBackendConfig(external_constants=True)
)

model_dir = Path(__file__).parent.resolve() / "model"
if model_dir.exists():
    shutil.rmtree(model_dir)
model_dir.mkdir(parents=True)
with open(model_dir / "model.pte", "wb") as file:
    executorch_program.write_to_file(file)
executorch_program.write_tensor_data_to_file(model_dir)
