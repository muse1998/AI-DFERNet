import torch
from torch import nn
from models.modules import backbone,Transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = backbone()
        self.fc = nn.Linear(256*4*4, 7)

    def forward(self, x):

        x,x_local,map= self.s_former(x,epoch)
        ##feature before fc layer
        x_FER=x[:,0]
        ## snippet-based feature
        x_local=x_local[:,0]
        ##output
        x=self.fc(x_FER)


        return x,x_FER,x_local


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)
