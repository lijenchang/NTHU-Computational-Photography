import torch
import torch.nn as nn

'''
    Example model construction in pytorch
'''
class WDSRblock_typeA(nn.Module):
    def __init__(self, nFeat, ExpandRatio=2):
        super(WDSRblock_typeA, self).__init__()
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat*ExpandRatio, 3, padding=1, bias=True))
        modules.append(nn.ReLU(True))
        modules.append(nn.Conv2d(nFeat*ExpandRatio, nFeat, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        out += x
        return out

class WDSRblock_typeB(nn.Module):
    def __init__(self, nFeat, ExpandRatio=2, bias=True):
        super(WDSRblock_typeB, self).__init__()
        #===== write your model definition here =====#
        self.body = nn.Sequential(
            nn.Conv2d(nFeat, nFeat * ExpandRatio, kernel_size = 1, padding = 0, bias = True),
            nn.ReLU(True),
            nn.Conv2d(nFeat * ExpandRatio, int(0.8 * nFeat + 0.5), kernel_size = 1, padding = 0, bias = True),
            nn.Conv2d(int(0.8 * nFeat + 0.5), nFeat, kernel_size = 3, padding = 1, bias = True)
        )
    
    def forward(self, x):
        #===== write your dataflow here =====#
        out = self.body(x)
        out += x
        return out

class upsampler(nn.Module):
    def __init__(self, nFeat, scale=2):
        super(upsampler, self).__init__()
        #===== write your model definition here =====#
        self.body = nn.Sequential(
            nn.Conv2d(nFeat, (scale**2) * nFeat, kernel_size = 3, padding = 1, bias = True),
            nn.PixelShuffle(upscale_factor = scale),
            nn.ReLU(True)
        )
 
    def forward(self, x):
        #===== write your dataflow here =====#
        out = self.body(x)
        return out

class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, ExpandRatio=4, nResBlock=8, imgChannel=3):
        super(ZebraSRNet, self).__init__()
        #===== write your model definition here using 'WDSRblock_typeB' and 'upsampler' as the building blocks =====#
        self.conv0 = nn.Conv2d(imgChannel, nFeat, kernel_size = 3, padding = 1, bias = True)

        module = [WDSRblock_typeB(nFeat, ExpandRatio) for i in range(nResBlock)]
        self.wdsr_blocks = nn.Sequential(*module)
        
        self.upsamplers = nn.Sequential(
            upsampler(nFeat, scale = 2),
            upsampler(nFeat, scale = 2)
        )
        self.conv1 = nn.Conv2d(nFeat, imgChannel, kernel_size = 3, padding = 1, bias = True)

    def forward(self, x):
        #===== write your dataflow here =====#
        x = self.conv0(x)
        out = self.wdsr_blocks(x)
        out += x
        out = self.upsamplers(out)
        out = self.conv1(out)

        return out