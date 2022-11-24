import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, (3, 3), bias = False)
        self.conv.weight.data.zero_()

    def forward(self, x):
        y = self.conv(x)
        return y

input = torch.rand(1, 1, 3, 3)
model = ConvModel()
# out = model(torch.Tensor([[[[1,1,1], [1,1,1], [1,1,1]]]]))
# print(out.detach().numpy())
traced_model = torch.jit.trace(model, input, check_trace=True)
torch.jit.save(traced_model, "conv.pt")

