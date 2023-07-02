from operations import *
from genotypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


operations_call = {
    "ConvNext": ConvNextBlock,
    "InvBottleneck": InvBottleNeckBlock,
    "ResNet" : ResNetBlock, 
}

def decode(genotype : list ) -> list :
        return [ [gene[0:-1]] * gene[-1] for gene in genotype ]

class Network(nn.Module):

    def __init__(self, genotype, n_classes = 2):
        super().__init__()
        self.genotype = genotype

        # CHANNEL PER STAGE 
        self.in_channel = [16,32,64,96]

        # NETWORK
        self.layers = []
        
        # STEM CONVOLUTION
        self.stem_conv = conv3x3(3, self.in_channel[0], stride=2)

        #Decode the genotype
        self.network_description = decode(genotype)

        for stage_number, stage in enumerate(self.network_description):
            for i, block_gene in enumerate(stage):
                operation = operations_call[block_gene[0]](self.in_channel[stage_number], *block_gene[1:])
                self.layers.append(operation)
            
            # Exit of the stage: then downsample
            if stage_number < ( len(self.network_description) - 1) : 
                downsample = conv3x3(self.in_channel[stage_number],self.in_channel[stage_number+1],stride = 2)
                self.layers.append(downsample)

               
        # BUILD THE NET
        self.layers = nn.Sequential(*self.layers)
        self.last_conv = conv1x1(self.in_channel[-1], 1280)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, n_classes)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # change device if possible.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    
    def forward(self, x):
      x = self.stem_conv(x)
      x = self.layers(x)
      x = self.last_conv(x)
      x = self.avg_pool(x).view(-1, 1280)
      x = self.classifier(x)
      return x
    
    def print_architecture(self):
        for stage_number, stage in enumerate(self.genotype):
            print(f"*****STAGE NUMBER {stage_number}*****")
            for i, _ in enumerate(stage):
                print (self.layers[i])
