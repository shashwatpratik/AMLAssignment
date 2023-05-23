import torch
from torch import nn
from models.CNN import net3D,ResNet,resnet50_fc512
from models.transformer import transformer

class M3T(nn.Module):
    def __init__(self, num_classes,in_channel,out_channel_3D,out_dim_2D,pretrained=True,use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu
        self.net3D = net3D(in_channel,out_channel_3D)
        self.resnet = resnet50_fc512(out_channel_3D,out_dim_2D,pretrained)
        self.transformer = transformer(out_dim_2D,num_classes)

    def forward(self,x):
        x = self.net3D(x)
        B, C, L, W, H = x.shape
        x, BN = self.pre_resnet(x)
        x = self.resnet(x)
        x = self.post_resnet(x, BN, B)
        #P,B,N,embed_dim = x.shape
        x = self.transformer(x)
        return x


    def pre_resnet(self,x):
        # For slicing in N frames
        for i in range(x.size(-1)):
            if i == 0:
                pl = x[:, :, i, :, :]
            else:
                pl = torch.cat((pl, x[:, :, i, :, :]), dim=0)

        BN, C, L, L = pl.shape
        all_plane = pl
        # For slicing in N frames
        for i in range(x.size(-1)):
            if i == 0:
                pl = x[:, :, :, i, :]
            else:
                pl = torch.cat((pl, x[:, :, :, i, :]), dim=0)

        all_plane = torch.cat((all_plane,pl),dim=0)
        # For slicing in N frames
        for i in range(x.size(-1)):
            if i == 0:
                pl = x[:, :, :, :, i]
            else:
                pl = torch.cat((pl, x[:, :, :, :, i]), dim=0)

        x = torch.cat((all_plane,pl),dim=0)
        return x, BN

    def post_resnet(self,x, BN, B):
        N = int(BN / B)
        x = torch.split(x, N, dim=0)
        x = torch.stack(x, dim=0)
        x = torch.split(x, B, dim=0)
        x = torch.stack(x, dim=0)
        return x


