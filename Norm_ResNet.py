import torch
import torch.nn as nn
# from DFTND.robustness_lib.robustness import helpers

class  Norm_ResNet18(nn.Module):
    def __init__(self, model, dataset):
        super(Norm_ResNet18, self).__init__()
        # self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
    def forward (self, inp):
        # normalized_inp = self.normalizer(inp)
        # output = self.model(normalized_inp)
        output = self.model(inp)
        return output