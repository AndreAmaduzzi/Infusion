import torch.nn as nn
import torch
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d
import random

__all__ = ['PVConv', 'SelfAttention', 'CrossAttention', 'Swish', 'PVConvReLU']

class ContextSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, context=None, text=None, epoch=0, save_matrices=False):
        for layer in self:
            if isinstance(layer, CrossAttention):
                x = layer(x, context, text, epoch, save_matrices)
            else:
                x = layer(x)
        return x


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)


class SelfAttention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super(SelfAttention, self).__init__()
        assert in_ch % num_groups == 0
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1) #in_ch=64
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)
        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1) #in_ch=512
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)

        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)


    def forward(self, x):
        B, C = x.shape[:2]
        h = x

        q = self.q(h)           # B,64,16,16,16
        q = q.reshape(B,C,-1)   # B,64,4096
        k = self.k(h)           # B,64,16,16,16
        k = k.reshape(B,C,-1)   # B,64,4096
        v = self.v(h)
        v = v.reshape(B,C,-1)

        qk = torch.matmul(q.permute(0, 2, 1), k) #* (int(C) ** (-0.5))

        w = self.sm(qk)

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B,C,*x.shape[2:])

        h = self.out(h)

        x = h + x

        x = self.nonlin(self.norm(x))   

        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, num_groups, context_dim, D=3):
        super(CrossAttention, self).__init__()
        assert query_dim % num_groups == 0
        self.q = nn.Conv3d(query_dim, query_dim, 1)     # 64,64 =>      Bx64x16x16x16 => Bx77x4096
        self.k = nn.Conv1d(context_dim, query_dim, 1)   # 1024,64 =>    Bx77x1024   => Bx77x64
        self.v = nn.Conv1d(context_dim, query_dim, 1)   # 1024,64 =>    Bx77x1024   => Bx77x64

        self.out = nn.Conv3d(query_dim, query_dim, 1)   # 64,64

        self.norm = nn.GroupNorm(num_groups, query_dim)
        self.nonlin = Swish()
        self.context_dim = context_dim

        self.sm = nn.Softmax(-1)


    def forward(self, x, context=None, text=None, epoch=0, save_matrices=False):
        B, C = x.shape[:2]                  # x=B,64,16,16,16  # context=B,77,1024
        
        random_n = random.randrange(0, 10000)
        if save_matrices:
            torch.save(x, f"./exps/exp_14/matrices/x_{epoch}_{random_n}.pt")
            torch.save(context, f"./exps/exp_14/matrices/context_{epoch}_{random_n}.pt")
            with open(f"./exps/exp_14/matrices/text_{epoch}_{random_n}.txt", "w") as f:
                for text_pt in text:
                    f.write(text_pt + "\n")


        q = self.q(x)                               #Bx64x16x16x16
        q = q.reshape(B,C,-1)                       #Bx64x4096
        k = self.k(context.permute(0,2,1))          #Bx77x1024 => Bx1024x77 => Bx64x77
        v = self.v(context.permute(0,2,1))          #Bx77x1024 => Bx77x64

        qk = torch.matmul(q.permute(0,2,1), k)* (int(C) ** (-0.5))

        w = self.sm(qk)

        h = torch.matmul(v, w.permute(0, 2, 1))
        if save_matrices:
            torch.save(h, f"./exps/exp_14/matrices/v*w_{epoch}_{random_n}.pt")

        h = h.reshape(B,C,*x.shape[2:])

        h = self.out(h) # equivalent to to_out of StableDiffusion

        x = h + x
 
        x = self.nonlin(self.norm(x))

        if save_matrices:
            torch.save(qk,  f"./exps/exp_12/matrices/qk_{epoch}_{random_n}.pt")
            torch.save(h,   f"./exps/exp_12/matrices/h_{epoch}_{random_n}.pt")
            torch.save(x,   f"./exps/exp_12/matrices/out_{epoch}_{random_n}.pt")

        return x


class FeedForward(nn.Module):
    def __init__(self, query_dim, num_groups, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        assert query_dim % num_groups == 0
        inner_dim = query_dim * mult

        self.net = nn.Sequential(
                                    nn.Conv3d(query_dim, inner_dim, 1),
                                    Swish(),
                                    nn.Dropout(dropout),
                                    nn.Conv3d(inner_dim, query_dim, 1)
                                )

        self.norm = nn.GroupNorm(num_groups, query_dim)
        self.nonlin = Swish()

    def forward(self, x, context=None, text=None, epoch=0, save_matrices=False):
        B, C = x.shape[:2]                  
        
        x = self.net(x) + x
        x = self.nonlin(self.norm(x))

        return x


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = 1024 # length of text embeddings
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            Swish()
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            SelfAttention(out_channels, 8) if attention else Swish(),
        ]

        if attention:
            voxel_layers += [
                CrossAttention(out_channels, 8, self.context_dim),                
                FeedForward(out_channels, 8, dropout=dropout),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)                
                ]

        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))

        self.voxel_layers = ContextSequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs, context=None, text=None, epoch=0, save_matrices=False):
        features, coords, temb = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features, context, text, epoch, save_matrices)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, temb

class PVConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False, leak=0.2,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(leak, True)
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            SelfAttention(out_channels, 8) if attention else nn.LeakyReLU(leak, True)
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, temb = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, temb
