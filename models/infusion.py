from typing import Mapping
import torch
from torch import nn, Tensor
from models.mapping_net import MappingNet
from models.pvd import *

class Infusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.mapping_net = MappingNet(opt.dmodel, opt.nhead, opt.nlayers, opt.out_flatshape) # TODO: modify mapping_net to get desired output from it
        betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
        self.pvd = PVD(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    def get_loss(self,
                cloud: Tensor,
                noises_batch: Tensor,
                text_embed: Tensor,
                src_key_padding_mask: Tensor
                ):
                
        condition = self.mapping_net(text_embed, src_key_padding_mask)
        
        loss = self.pvd.get_loss_iter(data=cloud, noises=noises_batch, condition=condition)

        return loss
    
    def get_clouds(self,
                text_embed: Tensor,             # a batch of text embeds
                src_key_padding_mask: Tensor,   # a batch of masks  
                test_shapes: Tensor):           # a batch of shapes
        
        condition = self.mapping_net(text_embed, src_key_padding_mask)
        clouds = self.pvd.gen_samples(self.new_x_chain(test_shapes, condition.shape[0]).shape, test_shapes.device, clip_denoised=False, condition=condition)                    # we get clouds for a batch of texts
        return clouds

    def get_cloud_traj(self,
                text_embed: Tensor,             # a batch of text embeds
                src_key_padding_mask: Tensor,   # a batch of masks  
                test_shapes: Tensor):           # a batch of shapes
        
        condition = self.mapping_net(text_embed, src_key_padding_mask)
        first_condition = condition[0].unsqueeze(0)
        cloud_traj = self.pvd.gen_sample_traj(self.new_x_chain(test_shapes, 1).shape, test_shapes.device, freq=40, clip_denoised=False, condition=first_condition) # we get trajectory only for 1 shape
        return cloud_traj
        

    def multi_gpu_wrapper(self, f):
        self.pvd.model = f(self.pvd.model)

    def new_x_chain(self, x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)
