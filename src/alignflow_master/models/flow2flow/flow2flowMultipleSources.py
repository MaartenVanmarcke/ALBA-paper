import os
import torch
import torch.nn as nn
import util

from itertools import chain
from models.patch_gan import PatchGAN
from models.real_nvp import RealNVP, RealNVPLoss


    
import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src","seed.txt")
file = open(p)
seed = int(file.read())
file.close()
torch.manual_seed(seed)

class Flow2Flow(nn.Module):
    """Flow2Flow Model

    Normalizing flows for unpaired image-to-image translation.
    Uses two normalizing flow models (RealNVP) for the generators,
    and two PatchGAN discriminators. The generators map to a shared
    intermediate latent space `Z` with simple prior `p_Z`, and the
    whole system optimizes a hybrid GAN-MLE objective.
    """
    def __init__(self, args):
        """
        Args:
            args: Configuration args passed in via the command line.
        """
        super(Flow2Flow, self).__init__()
        self.device = 'cuda' if len(args.gpu_ids) > 0 else 'cpu'
        self.gpu_ids = args.gpu_ids
        self.is_training = args.is_training
        self.num_sources = args.num_sources


        # Set up RealNVP generators (g_src: X <-> Z, g_tgt: Y <-> Z)
        self.g_src = nn.ModuleList([RealNVP(args.features, un_normalize_x=True, no_latent=False) for _ in range(self.num_sources)])
        for model in self.g_src:
            util.init_model(model, init_method=args.initializer)

        if self.is_training:
            # Set up discriminators
            self.d_tgt =nn.ModuleList( [PatchGAN(args) for _ in range(self.num_sources)])

            self._data_parallel()

            # Set up loss functions
            self.max_grad_norm = args.clip_gradient
            self.lambda_mle = args.lambda_mle
            self.mle_loss_fn = RealNVPLoss(n_features=args.features)
            self.gan_loss_fn = util.GANLoss(device=self.device, use_least_squares=True)

            self.clamp_jacobian = args.clamp_jacobian
            self.jc_loss_fn = util.JacobianClampingLoss(args.jc_lambda_min, args.jc_lambda_max)

            # Set up optimizers
            g_src_params = [util.get_param_groups(model, args.weight_norm_l2, norm_suffix='weight_g') for model in self.g_src]
            self.opt_g = torch.optim.Adam(chain.from_iterable(g_src_params),
                                          lr=args.rnvp_lr,
                                          betas=(args.rnvp_beta_1, args.rnvp_beta_2))
            self.opt_d = torch.optim.Adam(chain.from_iterable([model.parameters() for model in self.d_tgt]),
                                          lr=args.lr,
                                          betas=(args.beta_1, args.beta_2))
            self.optimizers = [self.opt_g, self.opt_d]
            self.schedulers = [util.get_lr_scheduler(opt, args) for opt in self.optimizers]

            # Setup image mixers
            buffer_capacity = 50 if args.use_mixer else 0
            self.src2tgt_buffer = [[util.ImageBuffer(buffer_capacity) for _ in range(self.num_sources)] for _ in range(self.num_sources)] # Buffer of generated tgt images
        else:
            self._data_parallel()

        # Images in flow src -> lat -> tgt
        self.src = [0]*self.num_sources
        self.src2lat = [0]*self.num_sources
        self.src2tgt = [[0]*self.num_sources for _ in range(self.num_sources)]

        # Jacobian clamping tensors
        self.src_jc = [0]*self.num_sources
        self.src2tgt_jc = [[0]*self.num_sources for _ in range(self.num_sources)]

        # Discriminator loss
        self.loss_d_tgt = [0]*self.num_sources
        self.loss_d = None

        # Generator GAN loss
        self.loss_gan_src = [0]*self.num_sources
        self.loss_gan = None

        # Generator MLE loss
        self.loss_mle_src = [0]*self.num_sources
        self.loss_mle = None

        # Jacobian Clamping loss
        self.loss_jc_src = [0]*self.num_sources
        self.loss_jc_tgt = [0]*self.num_sources
        self.loss_jc = None

        # Generator total loss
        self.loss_g = None

    def set_inputs(self, x, weights = None):
        """Set the inputs prior to a forward pass through the network.

        Args:
            src_input: Tensor with src input
            tgt_input: Tensor with tgt input
        """
        self.src = [src_input.to(self.device) for src_input in x]
        if weights == None:
            self.weights = [None for _ in range(self.num_sources)]
        else:
            self.weights = [w for w in weights]
        

    def forward(self):
        """No-op. We do the forward pass in `backward_g`."""
        pass

    def test(self):
        """Run a forward pass through the generator for test inference.
        Used during test inference only, as this throws away forward-pass values,
        which would be needed for backprop.

        Important: Call `set_inputs` prior to each successive call to `test`.
        """
        # Disable auto-grad because we will not backprop
        with torch.no_grad():
            src2lat = [0]*self.num_sources
            src2lat2tgt = [[0]*self.num_sources for _ in range(self.num_sources)]
            for i in range(self.num_sources):
                src2lat[i], _ = self.g_src[i](self.src[i], reverse=False)
                for j in range(self.num_sources):
                    if i != j:
                        src2lat2tgt[i][j],_ = self.g_src[j](src2lat[i], reverse=True)
                        self.src2tgt[i][j] = torch.tanh(src2lat2tgt[i][j]) # TODO: ???
        return src2lat, src2lat2tgt

    def _forward_d(self, d, real_img, fake_img):
        """Forward  pass for one discriminator."""

        # Forward on real and fake images (detach fake to avoid backprop into generators)
        loss_real = self.gan_loss_fn(d(real_img), is_tgt_real=True)
        loss_fake = self.gan_loss_fn(d(fake_img.detach()), is_tgt_real=False)
        loss_d = 0.5 * (loss_real + loss_fake)

        return loss_d

    def backward_d(self):
        self.loss_d_tgt = [0]*self.num_sources
        for i in range(self.num_sources):
            for j in range(self.num_sources):
                if i != j:
                    # Forward tgt discriminator
                    src2tgt = self.src2tgt_buffer[i][j].sample(self.src2tgt[i][j])
                    self.loss_d_tgt[j] += self._forward_d(self.d_tgt[j], self.src[j], src2tgt)

        # Backprop
        self.loss_d = sum(self.loss_d_tgt)
        if not isinstance(self.loss_d, int):
            self.loss_d.backward()
        else:
            # No discriminator loss, so there should only be one bag (else there is a bug)
            if self.num_sources != 1:
                raise Exception("Implementation error")

    def backward_g(self):
        if self.clamp_jacobian:
            # Double batch size with perturbed inputs for Jacobian Clamping
            self._jc_preprocess()

        # Forward src -> lat: Get MLE loss
        for i in range(self.num_sources):
            self.src2lat[i], sldj_src2lat = self.g_src[i](self.src[i], reverse=False)
            self.loss_mle_src[i] = self.lambda_mle * self.mle_loss_fn(self.src2lat[i], sldj_src2lat, self.weights[i])

            # Finish src -> lat -> tgt: Say target is real to invert loss
            for j in range(self.num_sources):
                if i != j:
                    src2tgt, _ = self.g_src[j](self.src2lat[i], reverse=True)
                    self.src2tgt[i][j] = torch.tanh(src2tgt)


        # Jacobian Clamping loss
        if self.clamp_jacobian:
            # Split inputs and outputs from Jacobian Clamping
            self.loss_jc_src = [0]*self.num_sources
            self._jc_postprocess()
            for i in range(self.num_sources):
                for j in range(self.num_sources):
                    if i != j:
                        self.loss_jc_src[i] += self.jc_loss_fn(self.src2tgt[i][j], self.src2tgt_jc[i][j], self.src[i], self.src_jc[i])
            self.loss_jc = sum(self.loss_jc_src)
        else:
            for i in range(self.num_sources):
                self.loss_jc_src[i] = 0.
            self.loss_jc = 0.

        # GAN loss
        self.loss_gan_src = [0]*self.num_sources
        for i in range(self.num_sources):
            for j in range(self.num_sources):
                if i != j:
                    self.loss_gan_src[i] += self.gan_loss_fn(self.d_tgt[j](self.src2tgt[i][j]), is_tgt_real=True) 

        # Total losses
        self.loss_gan = sum(self.loss_gan_src)
        self.loss_mle = sum(self.loss_mle_src)

        # Backprop
        self.loss_g = self.loss_gan + self.loss_mle + self.loss_jc
        self.loss_g.backward()

    def train_iter(self):
        """Run a training iteration (forward/backward) on a single batch.
        Important: Call `set_inputs` prior to each call to this function.
        """
        # Forward
        self.forward()

        # Backprop the generators
        self.opt_g.zero_grad()
        self.backward_g()
        util.clip_grad_norm(self.opt_g, self.max_grad_norm)
        self.opt_g.step()

        # Backprop the discriminators
        self.opt_d.zero_grad()
        self.backward_d()
        util.clip_grad_norm(self.opt_d, self.max_grad_norm)
        self.opt_d.step()

    def get_loss_dict(self):
        """Get a dictionary of current errors for the model."""
        loss_dict = {
            # Generator loss
            'loss_gan': self.loss_gan,
            'loss_jc': self.loss_jc,
            'loss_mle': self.loss_mle,
            'loss_g': self.loss_g,

            # Discriminator loss
            'loss_d': self.loss_d
        }

        # Map scalars to floats for interpretation outside of the model
        loss_dict = {k: v.item() for k, v in loss_dict.items()
                     if isinstance(v, torch.Tensor)}

        return loss_dict

    def get_image_dict(self):
        """Get a dictionary of current images (src, tgt_real, tgt_fake) for the model.

        Returns: Dictionary containing numpy arrays of shape (batch_size, num_channels, height, width).
        Keys: {src, src2tgt, tgt2src}.
        """
        raise NotImplementedError("")
        image_tensor_dict = {'src': self.src,
                             'src2tgt': self.src2tgt}

        if self.is_training:
            # When training, include full cycles
            image_tensor_dict.update({
                'tgt': self.tgt,
                'tgt2src': self.tgt2src
            })

        image_dict = {k: util.un_normalize(v) for k, v in image_tensor_dict.items()}

        return image_dict

    def on_epoch_end(self):
        """Callback for end of epoch.

        Update the learning rate by stepping the LR schedulers.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        """Get the current learning rate"""
        return self.optimizers[0].param_groups[0]['lr']

    def _data_parallel(self):
        for i in range(self.num_sources):
            self.g_src[i] = nn.DataParallel(self.g_src[i], self.gpu_ids).to(self.device)
            if self.is_training:
                self.d_tgt[i] = nn.DataParallel(self.d_tgt[i], self.gpu_ids).to(self.device)

    def _jc_preprocess(self):
        """Pre-process inputs for Jacobian Clamping. Doubles batch size.

        See Also:
            Algorithm 1 from https://arxiv.org/1802.08768v2
        """
        for i in range(self.num_sources):
            delta = torch.randn_like(self.src[i])
            src_jc = self.src[i] + delta / delta.norm()
            src_jc.clamp_(-1, 1)
            self.src[i] = torch.cat((self.src[i], src_jc), dim=0)

    def _jc_postprocess(self):
        """Post-process outputs after Jacobian Clamping.

        Chunks `self.src` into `self.src` and `self.src_jc`,
        `self.src2tgt` into `self.src2tgt` and `self.src2tgt_jc`,
        and similarly for `self.tgt` and `self.tgt2src`.

        See Also:
            Algorithm 1 from https://arxiv.org/1802.08768v2
        """
        for i in range(self.num_sources):
            self.src[i], self.src_jc[i] = self.src[i].chunk(2, dim=0)

            for j in range(self.num_sources):
                if i != j:
                    self.src2tgt[i][j], self.src2tgt_jc[i][j] = self.src2tgt[i][j].chunk(2, dim=0)
