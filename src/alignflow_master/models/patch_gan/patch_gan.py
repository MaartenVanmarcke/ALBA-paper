import torch.nn as nn

from util import init_model, get_norm_layer


class PatchGAN(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, args):
        """Constructs a basic PatchGAN convolutional discriminator.

        Each position in the output is a score of discriminator confidence that
        a 70x70 patch of the input is real.

        Args:
            args: Arguments passed in via the command line.
        """
        super(PatchGAN, self).__init__()

        #norm_layer = get_norm_layer(args.norm_type)

        layers = []

        # Double channels for conditional GAN (concatenated src and tgt images)

        layers += [nn.Linear(2,3),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Linear(3,2),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Linear(2,2),
                   nn.LeakyReLU(0.2, True)]

        self.model = nn.Sequential(*layers)
        init_model(self.model, init_method=args.initializer)

    def forward(self, input_):
        return self.model(input_)
