from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


# A simple PyTorch model that adds two input tensors
class Add(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


model = Add()

exported_program = export(model, (torch.ones(1), torch.ones(1)))
executorch_program = to_edge_transform_and_lower(exported_program).to_executorch()

model_path = Path(__file__).parent.parent / "models" / "add.pte"
with open(model_path, "wb") as file:
    file.write(executorch_program.buffer)
