import torch
from torch import nn, Tensor
from models.mapping_net import MappingNet
from models.pvd import *

class Infusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mapping_net = MappingNet(args.dmodel, args.nhead, args.nlayers, args.out_flatshape)
        betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
        self.pvd = PVD(args, betas, args.loss_type, args.model_mean_type, args.model_var_type)

    def forward(self,
                x: Tensor,
                src_key_padding_mask: Tensor):
            
        pred_noise = self.mapping_net(x, src_key_padding_mask)

        self.pvd.eval()
        with torch.no_grad():
            pred_shape = self.pvd.gen_samples(pred_noise.shape,
                                                'cuda',
                                                pred_noise=pred_noise,
                                                clip_denoised=False)

        return pred_shape, pred_noise

    def multi_gpu_wrapper(self, f):
        self.pvd.model = f(self.pvd.model)


