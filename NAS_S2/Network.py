from modalities import * 
from genotypes import get_connection_dictionary
from torch import nn
from torchvision import transforms
import torch
from operations import ConvNextBlock, InvBottleNeckBlock, conv3x3, conv1x1

from PIL import Image

operations_call = {
    "ConvNext": ConvNextBlock,
    "InvBottleNeck": InvBottleNeckBlock,
}

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x

class Network( nn.Module ):

    def __init__(self, genotype, n_classes = 2):
        super().__init__()
        self.genotype = genotype

        # CHANNEL PER STAGE 
        self.in_channel = [16,32,64,96] 

        # NETWORK
        self.blocks = nn.ModuleList()
        self.connections = []

        # STEM CONVOLUTION
        self.stem_conv = conv3x3(3, self.in_channel[0], stride=2)


        # For each stage
        for block_number, block in enumerate(self.genotype):
            cell_1,cell_2,connection_info = block
            cell_1_op = nn.Sequential()
            cell_2_op = nn.Sequential()
            connection_info = get_connection_dictionary(connection_info)
            self.connections.append( connection_info )
            if (one_operation_mode(connection_info)) :
                for i in range(4) : 
                    cell_1_op.add_module(f"{cell_1[0]}-{block_number}-{i}",operations_call[cell_1[0]](self.in_channel[block_number],cell_1[1]))
            else :
                for i in range(2):
                    cell_1_op.add_module(f"{cell_1[0]}-{block_number}-{i}",operations_call[cell_1[0]](self.in_channel[block_number],cell_1[1]))
                    cell_2_op.add_module(f"{cell_2[0]}-{block_number}-{i}",operations_call[cell_2[0]](self.in_channel[block_number],cell_2[1]))
                

            if block_number < ( len(self.genotype) - 1) : 
                downsample = conv3x3(self.in_channel[block_number], self.in_channel[block_number+1], stride = 2)
            else :
                downsample = IdentityModule()
            self.blocks.append(nn.ModuleList([ cell_1_op, cell_2_op, downsample ]) )
                
            
        # BUILD THE NET
        # self.layers = nn.Sequential(*self.layers)
        self.last_conv = conv1x1(self.in_channel[-1], 1280)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, n_classes)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # change device if possible.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self,X):
        X = self.stem_conv(X)
        X = self.block_forward(X)
        X = self.last_conv(X)
        X = self.avg_pool(X).view(-1, 1280)
        X = self.classifier(X)
        return X

    def block_forward(self,X):
        for block, connection in zip( self.blocks, self.connections ) : 
            cell_1, cell_2 , downsample = block

            if one_operation_mode(connection):
                X = downsample ( cell_1(X) ) 

            if two_branch_mode(connection):
                X = downsample( cell_1(X) + cell_2(X) )

            if sequential_mode(connection):
                if(connection["skip"]):
                    X = downsample( cell_2(cell_1(X)) + X )
                else :
                    X = downsample( cell_2( cell_1(X) ) )

            if complete_mode(connection) :
                if(connection["skip"]):
                    out_1 = cell_1(X)
                    X = downsample( cell_2(out_1 + X) + out_1)
                else :
                    X = downsample ( cell_2(cell_1(X) + X) )
                
        return X
                

    def probabilities(self, image : Image.Image ):
        self.eval()
        return torch.sigmoid(self(self.preprocess(image)).data)
        

    def predict(self, image : Image.Image ) :
        self.eval()
        prediction = torch.argmax(self.probabilities(image))
        if ( prediction == 0) : 
            return "NOT PERSON"
        else : 
            return "PERSON"
        
    def preprocess(self, image : Image.Image):
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype( dtype = torch.float ),
            transforms.Resize(128,antialias=True),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] )
        ])
        X = transform(image).unsqueeze(0).to(self.device)
        return X