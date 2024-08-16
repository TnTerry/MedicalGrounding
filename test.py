import torch
import torch.nn as nn

from EMCAD.lib.networks import EMCADNet

class VKInject(nn.Module):
    def __init__(
            self, 
            num_classes=1, 
            kernel_sizes=[1,3,5], 
            expansion_factor=2, 
            dw_parallel=True, 
            add=True, 
            lgag_ks=3, 
            activation='relu', 
            encoder='pvt_v2_b2', 
            pretrain=True
    ):
        super(VKInject, self).__init__()
        self.EMCAD = EMCADNet(num_classes, kernel_sizes, expansion_factor, dw_parallel, add, lgag_ks, activation, encoder, pretrain)


if __name__ == '__main__':
    model = EMCADNet()
