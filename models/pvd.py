import numpy as np
import torch
from torch import nn
from models.pvcnn import PVCNN2Base
import random

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.eps_scale = self.betas / self.sqrt_one_minus_alphas_cumprod

        '''
        #DDIM PARAMETERS
        self.ddim_timesteps = self.make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        '''
    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)   # noise = epsilon
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool, condition=None, epoch=0, save_matrices=False,):

        # data = x_t
        if condition is not None:
            model_output = denoise_fn(data, t, condition, epoch=epoch, save_matrices=save_matrices)     # this is eps_theta
        else:
            raise Exception('condition is None')

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False, condition=None, epoch=0, save_matrices=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True, condition=condition, epoch=epoch, save_matrices=save_matrices)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    
    ## DDIM sampling from https://github.com/CompVis/latent-diffusion
    @torch.no_grad()
    def p_sample_ddim(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False, condition=None, epoch=0, save_matrices=False):
    #(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1., unconditional_conditioning=None):
        # data = x_t
        if condition is not None:
            e_t = denoise_fn(data, t, condition, epoch=epoch, save_matrices=save_matrices)     # eps_theta
        else:
            raise Exception('condition is None')

        b, *_, device = *x.shape, x.device


        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def p_sample_loop(self, denoise_fn, shape, device, condition=None,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, condition=condition,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t
    
    def test_p_sample_loop(self, denoise_fn, shape, device, condition=None,
                      noise_fn=torch.randn, constrain_fn=lambda x, t:x,
                      clip_denoised=True, max_timestep=None, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """
        if max_timestep is None:
            final_time = self.num_timesteps
        else:
            final_time = max_timestep

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, final_time if not keep_running else len(self.betas))):
            img_t = constrain_fn(img_t, t)
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, condition=condition,
                                  clip_denoised=clip_denoised, return_pred_xstart=False).detach()


        assert img_t.shape == shape
        return img_t


    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, condition=None,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, condition=condition,
                                  return_pred_xstart=False).detach()
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool, condition=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, condition=condition, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None, condition=None, text=None, epoch=0, save_matrices=False):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t, condition, text, epoch, save_matrices)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == torch.Size([B, D, N])
            assert eps_recon.shape == data_start.shape
            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        elif self.loss_type == 'scaled_mse':
            # predict x_t instead of x_start. It becomes a scaled version of mse loss
            eps_recon = denoise_fn(data_t, t, condition, text, epoch, save_matrices)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == torch.Size([B, D, N])
            assert eps_recon.shape == data_start.shape
            eps_scale = self._extract(self.eps_scale.to(data_start.device), t, data_start.shape)
            losses = ((eps_scale * noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, condition=None, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, condition=condition, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()

class PVCNN2(PVCNN2Base):
    
    sa_blocks = [                                                   # PVConv                                        # PointNetSAModule
        ((32, 2, 24), (1024, 0.1, 32, (32, 64))),  # ((out_channels, num_blocks, voxel_resolution), (num_centers, radius, num_neighbors, out_channels),
        ((64, 3, 12), (256, 0.2, 32, (64, 128))),
        ((128, 3, 6), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]   
    fp_blocks = [                #  PointNetFPModule                # PVConv  
        ((256, 256), (256, 3, 6)), # (out_channels), (out_channels, num_blocks, voxel_resolution)
        ((256, 256), (256, 3, 6)),
        ((256, 128), (128, 2, 12)),
        ((128, 128, 64), (64, 2, 24)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )

class PVD(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(PVD, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        self.model = PVCNN2(num_classes=args.nc, embed_dim=args.embed_dim, use_att=args.attention,
                            dropout=args.dropout, extra_feature_channels=0)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, condition=None, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, condition, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, condition=None, text=None, epoch=0, save_matrices=False):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, condition, text, epoch, save_matrices)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, c=None, noises=None, condition=None, text=None, epoch=0, save_matrices=False):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)  # modify epsilon for elements with t!=0  

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, condition=condition, text=text, epoch=epoch, save_matrices=save_matrices)
        assert losses.shape == t.shape == torch.Size([B])
        return losses, t

    def gen_samples(self, shape, device, noise_fn=torch.randn, condition=None, constrain_fn=lambda x, t:x,
                    clip_denoised=True, max_timestep=None,
                    keep_running=False):

        return self.diffusion.test_p_sample_loop(self._denoise, shape=shape, device=device, condition=condition, noise_fn=noise_fn,
                                            constrain_fn=constrain_fn, clip_denoised=clip_denoised, max_timestep=None,
                                            keep_running=keep_running)

        #return self.diffusion.test_p_sample_loop(self._denoise, shape=shape, device=device, condition=condition, noise_fn=noise_fn,
        #                                    clip_denoised=clip_denoised, keep_running=keep_running)

    def gen_sample_traj(self, shape, device, freq, noise_fn=torch.randn, condition=None,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, noise_fn=noise_fn, freq=freq,
                                                       condition=condition, clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)