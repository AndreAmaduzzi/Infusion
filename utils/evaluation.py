### FUNCTION USED FOR VALIDATION DURING TRAINING
from tqdm import tqdm
import torch
import math
from utils.misc import *
sys.path.append('../diffusion-text-shape/')
from evaluation import evaluation_metrics

def chamfer_cond_uncond(model,
                        batch_size,
                        dataloader,
                        epoch,
                        exp_path,
                        unconditional_path='../PVD/output/get_clouds_and_seed/2022-11-02-18-14-40/syn/'):

    # save current seed
    curr_torch_seed = torch.get_rng_state()
    curr_np_seed = np.random.get_state()

    # take saved seed from unconditional PVD inference
    pvd_torch_seed = torch.load(os.path.join(unconditional_path, 'torch_seed.pt'))
    pvd_np_seed_arr = np.load(os.path.join(unconditional_path, 'np_seed_array.npy'))
    pvd_pos = np.load(os.path.join(unconditional_path, 'np_pos.npy'))
    pvd_has_gauss = np.load(os.path.join(unconditional_path, 'np_has_gauss.npy'))
    pvd_cached_gauss = np.load(os.path.join(unconditional_path, 'np_cached_gauss.npy'))

    torch.backends.cudnn.deterministic = True   # If I repeat this command before calling torch.randn, I get the same result every time

    # set seed
    torch.set_rng_state(pvd_torch_seed)
    rand_state = ('MT19937', pvd_np_seed_arr, pvd_pos, pvd_has_gauss, pvd_cached_gauss)
    np.random.set_state(rand_state)

    # generate 1000 clouds with this seed
    x_pvd = torch.load('../PVD/output/get_clouds_and_seed/2022-11-02-18-14-40/syn/samples.pth')
    comp_size = x_pvd.shape[0]
    random_1 = torch.randn((1), dtype=torch.float, device='cuda:0')
    random_2 = torch.randn((1), dtype=torch.float, device='cuda:0')
    gen_pcs = []
    for i in tqdm(range(0, math.ceil(comp_size / batch_size)), 'Generate'):
        model.pvd.eval()
        with torch.no_grad():
            val_batch = next(iter(dataloader))
            text_embed_val = val_batch["text_embed"].cuda()
            text_embed_val = maxlen_padding(text_embed_val)
            x_val = val_batch['pointcloud'].transpose(1,2).cuda() 
            x_gen_eval = model.get_clouds(text_embed_val, x_val)
            # transpose shapes because metrics want (2048, 3) instead of (3, 2048)
            x_gen_eval = x_gen_eval.transpose(1,2)
            gen_pcs.append(x_gen_eval.detach().cpu())

    gen_pcs = torch.cat(gen_pcs, dim=0)[:comp_size]

    visualize_pointcloud_batch(os.path.join(exp_path, f'pvd_clouds_{epoch}.png'),
                                x_pvd[:20], None, None,
                                None)

    visualize_pointcloud_batch(os.path.join(exp_path, f'gen_clouds_{epoch}.png'),
                                gen_pcs[:20], None, None,
                                None)

    print('Generated clouds: ', gen_pcs.shape)
    print('PVD clouds: ', x_pvd.shape)

    x_gen = normalize_clouds_for_validation(gen_pcs, mode='shape_bbox', logger=logger)
    x_pvd = normalize_clouds_for_validation(x_pvd, mode='shape_bbox', logger=logger)

    # compute mean chamfer loss between corresponding clouds from unconditional and conditional PVD
    chamfer_dist = chamfer(x_gen, x_pvd)
    chamfer_dist = chamfer_dist[0]
    mean_chamfer = torch.mean(chamfer_dist)

    # reset the seed to previous value
    torch.set_rng_state(curr_torch_seed)
    np.random.set_state(curr_np_seed)
    torch.backends.cudnn.deterministic = False

    return chamfer_dist, mean_chamfer

def run_validation(model,
                ref_histogram,
                output_dir,
                epoch,
                val_size,
                batch_size,
                dataloader):
                
    gen_pcs=[]
    ref_pcs=[]
    gen_pcs_denorm=[]
    ref_pcs_denorm=[]
    texts=[]
    model_ids=[]
    for val_batch in tqdm(dataloader, desc='Generate'):     # we iterate over the WHOLE dataloader
        with torch.no_grad():
            text_embed_val = val_batch["text_embed"].cuda()
            text_embed_val = maxlen_padding(text_embed_val)
            x_val = val_batch['pointcloud'].transpose(1,2).cuda() 
            x_gen_eval = model.get_clouds(text_embed_val, x_val).detach().cpu()
            
            for text in val_batch["text"]:
                texts.append(text)
            for model_id in val_batch["model_id"]:
                model_ids.append(model_id)
            # transpose shapes because metrics want (2048, 3) instead of (3, 2048)
            x_gen_eval = x_gen_eval.transpose(1,2)
            x_val = x_val.transpose(1,2)
            
            # de-normalize clouds
            mean, std = val_batch['mean'].float(), val_batch['std'].float()
            x_gen_eval_denorm = x_gen_eval * std + mean
            x_val_denorm = x_val.detach().cpu() * std + mean

            gen_pcs_denorm.append(x_gen_eval_denorm)
            ref_pcs_denorm.append(x_val_denorm)

            gen_pcs.append(x_gen_eval)
            ref_pcs.append(x_val.detach().cpu())

    gen_pcs = torch.cat(gen_pcs, dim=0)[:val_size]
    ref_pcs = torch.cat(ref_pcs, dim=0)[:val_size]

    gen_pcs_denorm = torch.cat(gen_pcs_denorm, dim=0)[:val_size]
    ref_pcs_denorm = torch.cat(ref_pcs_denorm, dim=0)[:val_size]

    texts = texts[:val_size]
    model_ids = model_ids[:val_size]

    logger.info('Saving point clouds and text...')
    np.save(os.path.join(output_dir, f'out_{epoch}_denorm.npy'), gen_pcs_denorm.numpy())
    np.save(os.path.join(output_dir, f'ref_{epoch}_denorm.npy'), ref_pcs_denorm.numpy())
    np.save(os.path.join(output_dir, f'out_{epoch}.npy'), gen_pcs.numpy())
    np.save(os.path.join(output_dir, f'ref_{epoch}.npy'), ref_pcs.numpy())

    chamfer_dist = chamfer(ref_pcs, gen_pcs)
    chamfer_dist = chamfer_dist[0]
    print('chamfer: ', chamfer_dist.shape)
    np.save(os.path.join(output_dir, 'chamfer.npy'), chamfer_dist.cpu().numpy())
    mean_chamfer = torch.mean(chamfer_dist)
    print('mean chamfer: ', mean_chamfer)

    print('size of ref pcs: ', ref_pcs.shape)
    print('size of gen pcs: ', gen_pcs.shape)
    print('len of gen texts: ', len(texts))

    ## Compute metrics

    # Normalize clouds before computing metrics

    #gen_pcs = normalize_clouds_for_validation(gen_pcs, mode='shape_bbox', logger=logger)
    #ref_pcs = normalize_clouds_for_validation(ref_pcs, mode='shape_bbox', logger=logger)

    with torch.no_grad():
        results = evaluation_metrics.compute_all_metrics(gen_pcs, ref_pcs, batch_size, model_ids=model_ids, texts=texts, save_dir=output_dir, ref_hist=ref_histogram)
        results = {k:v.item() for k, v in results.items()}
        jsd = evaluation_metrics.jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results['jsd'] = jsd

    for k, v in results.items():
        logger.info('%s: %.12f' % (k, v))

    return results, mean_chamfer

def visualize_shape_grids(model, 
                        train_batch,
                        val_dl, 
                        output_dir, 
                        epoch, 
                        logger,
                        concat=False,
                        context_dim=77):

    with torch.no_grad():       
        text_embed_train = train_batch["text_embed"].cuda()
        text_embed_train = maxlen_padding(text_embed_train)
        x_train = train_batch["pointcloud"].transpose(1,2).cuda() 
        x_gen_eval = model.get_clouds(text_embed_train, x_train, concat=concat, context_dim=context_dim).detach().cpu()
        x_gen_list = model.get_cloud_traj(text_embed_train[0].unsqueeze(0), x_train, concat=concat, context_dim=context_dim)
        x_gen_all = torch.cat(x_gen_list, dim=0).detach().cpu()

        gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]

        gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

        logger.info('eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                    'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
            .format(
            *gen_eval_range, *gen_stats,
        ))

        # denormalize all the clouds, except the ones of the trajectory
        mean, std = train_batch['mean'].float(), train_batch['std'].float()

        x_gen_all = x_gen_all.transpose(1,2).contiguous()
        x_gen_eval = x_gen_eval.transpose(1,2).contiguous()
        x_train = x_train.transpose(1,2).contiguous()

        gen_eval_denorm = x_gen_eval * std + mean
        x_train_denorm = x_train.detach().cpu() * std + mean

        visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_train.png' % (output_dir, epoch),
                            gen_eval_denorm[:32], None, None,
                            None)

        visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all_train.png' % (output_dir, epoch),
                            x_gen_all[:32], None,
                            None,
                            None)

        visualize_pointcloud_batch('%s/epoch_%03d_x_train.png' % (output_dir, epoch), x_train_denorm[:32], None,
                            None,
                            None)

        text_embed_train = text_embed_train.detach().cpu()

    with torch.no_grad():
        val_batch = next(iter(val_dl))  # we pick always the first batch of the validation set
        text_embed_val = val_batch["text_embed"].cuda()
        text_embed_val = maxlen_padding(text_embed_val)
        x_val = val_batch['pointcloud'].transpose(1,2).cuda()
        x_gen_eval = model.get_clouds(text_embed_val, x_val).detach().cpu()
        x_gen_list = model.get_cloud_traj(text_embed_val[0].unsqueeze(0), x_val)
        x_gen_all = torch.cat(x_gen_list, dim=0).detach().cpu()

        gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]

        gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

        logger.info('eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                    'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
            .format(
            *gen_eval_range, *gen_stats,
        ))

        # denormalize all the clouds, except the ones of the trajectory
        mean, std = val_batch['mean'].float(), val_batch['std'].float()

        x_gen_all = x_gen_all.transpose(1,2).contiguous()
        x_gen_eval = x_gen_eval.transpose(1,2).contiguous()
        x_val = x_val.transpose(1,2).contiguous()

        gen_eval_denorm = x_gen_eval * std + mean
        x_val_denorm = x_val.detach().cpu() * std + mean


        visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_valid.png' % (output_dir, epoch),
                            gen_eval_denorm[:32], None, None,
                            None)

        visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all_valid.png' % (output_dir, epoch),
                            x_gen_all[:32], None,
                            None,
                            None)

        visualize_pointcloud_batch('%s/epoch_%03d_x_valid.png' % (output_dir, epoch), x_val_denorm[:32], None,
                            None,
                            None)
        
        text_embed_val = text_embed_val.detach().cpu()
