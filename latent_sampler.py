import torch
import random
import pickle as pkl


def read_avg_latent(chosen_label, type):
    """
    type = global or local
    """
    return pkl.load(open(f"/local/omp/infinityGAN/logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset/avg_latents/avg_{type}_latent_{chosen_label}.pkl", "rb"))

def read_saved_testing_vars(n):
    return pkl.load(open(f"/local/omp/infinityGAN/logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset/latents/{n}.pkl", "rb"))

def center_crop_local_latent(latent, dim):
    orig_dim = latent[0].shape[0] # 53
    start = (orig_dim - dim) // 2
    end = start + dim
    return torch.stack([l[start:end, start:end] for l in latent])

class LatentSampler():
    def __init__(self, generator, config):
        self.config = config
        self.generator = generator

    @torch.no_grad()
    def sample_global_latent(self, batch_size, device, requires_grad=False, mixing=True, saved_img_number=None):
        global_latent_dim = self.config.train_params.global_latent_dim
        is_mixing = random.random() < self.config.train_params.mixing if mixing else False

        latent_1 = torch.randn(batch_size, global_latent_dim, device=device)
        # Use same global latent for all created images
        # latent_1 = latent_1[0].repeat(batch_size, 1)
        if saved_img_number:
            index = int(saved_img_number) % 8
            saved_latent = read_saved_testing_vars(saved_img_number).global_latent[index][0]
            sunset_latent = read_saved_testing_vars("000131").global_latent[index][0]
            mixed_latent = saved_latent*0 + sunset_latent*1
            latent_1[0] = mixed_latent
        # latent_1 = read_avg_latent("Mountain_95", "global")[0]
        # latent_1 = latent_1[0].repeat(batch_size, 1)
        
        latent_2 = torch.randn(batch_size, global_latent_dim, device=device)
        # Try mixing global latents
        # if False:
        #     is_mixing = True
        #     index = int("000131") % 8
        #     latent_2 = read_saved_testing_vars("000131").global_latent[index][0]
        #     latent_2 = latent_2.repeat(batch_size, 1)

        latent = torch.stack([
            latent_1,
            latent_2 if is_mixing else latent_1,
        ], 1) # shape: (B, 2, D) # batch-first for dataparallel

        latent.requires_grad = requires_grad
        return latent

    def sample_local_latent(self, batch_size, device, requires_grad=False,
                            spatial_size_enlarge=1, specific_shape=None, exclude_padding=False, saved_img_number=None):

        local_latent_dim = self.config.train_params.local_latent_dim   

        if specific_shape is not None:
            spatial_shape = specific_shape
        elif spatial_size_enlarge != 1:
            if hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline:
                size = self.config.train_params.ts_input_size * spatial_size_enlarge
                spatial_shape = (size, size)
            else:
                base = self.config.train_params.ts_input_size // 2
                size = (int(round(base * spatial_size_enlarge)) * 2) + 1
                spatial_shape = (size, size)
        else:
            size = self.config.train_params.ts_input_size
            spatial_shape = (size, size)
        
        if self.config.train_params.use_ss and self.config.train_params.ss_unfold_radius > 0:
            if self.config.train_params.ss_n_layers > 0:
                ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
            else:
                ss_unfold_size = 0
            if exclude_padding:
                spatial_shape_ext = spatial_shape
            else:
                spatial_shape_ext = [
                    spatial_shape[0] + 2 * ss_unfold_size,
                    spatial_shape[1] + 2 * ss_unfold_size]
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape_ext[0], spatial_shape_ext[1], device=device)
        else:
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape[0], spatial_shape[1], device=device)

        # Use same local latents for each image
        # z_local = z_local[0].repeat(batch_size,1,1,1)
        # a = torch.randn(spatial_shape_ext, device=device)
        # a = torch.randn(spatial_shape_ext, device=device)
        # for i in range(batch_size):
        #     for j in range(local_latent_dim):
        #         z_local[i][j][:spatial_shape_ext[0] //2 ] += a[:spatial_shape_ext[0] //2] * i*0.5
        # z_local = z_local[0][0].repeat(256,1,1).repeat(8,1,1,1)


        # Saved latent
        if saved_img_number:
            index = int(saved_img_number) % 8
            z_local = read_saved_testing_vars(saved_img_number).local_latent[index]
            # center crop to 35x35 for fused_seq_connecting_generation
            # z_local = center_crop_local_latent(z_local, 35)
            # Repeat for each image in batch
            z_local = z_local.repeat(batch_size,1,1,1)
        
        # Label
        # z_local = read_avg_latent("Mountain_95", "local")

        # Repeat for each image in batch
        # z_local = z_local.repeat(batch_size,1,1,1)
        
        # for i in range(batch_size):
        #     for channel in range(local_latent_dim):
        #         for dimx in range(saved_latents.local_latent.shape[2]):
        #             for dimy in range(saved_latents.local_latent.shape[2]):
        #                 z_local[i][channel][dimx][dimy] = saved_latents.local_latent[0][channel][dimx][dimy]
                        # z_local[i][channel][dimx][dimy*2] = saved_latents.local_latent[0][channel][dimx][dimy]

                # for dimx in range(saved_latents.local_latent.shape[2]):
                #     for dimy in range(saved_latents.local_latent.shape[2],z_local.shape[3]):
                #         z_local[i][channel][dimx][dimy] = saved_latents.local_latent[0][channel][dimx][dimy-saved_latents.local_latent.shape[2]]

        z_local.requires_grad = requires_grad
        return z_local

