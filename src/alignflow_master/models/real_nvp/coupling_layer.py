import torch
import torch.nn as nn

from enum import IntEnum
from util import checkerboard_like

import math

class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, n_features, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        self.st_net = NewNet(n_features)

        # Learnable scale and shift for s
        self.s_scale = nn.Parameter(torch.ones(1))
        self.s_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x, sldj=None, reverse=True):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_like(x, reverse=self.reverse_mask)
            x_b = x * b
            #x_b = 2. * self.st_norm(x_b)
            #b = b.expand(x.size(0), -1, -1, -1)
            #x_b = F.relu(torch.cat((x_b, -x_b, b), dim=1))
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift
            s = s * (1. - b)
            t = t * (1. - b)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                if (sldj == None):
                    sldj = s.view(s.size(0), -1).sum(-1)
                else:
                    sldj += s.view(s.size(0), -1).sum(-1)
                
        '''else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_norm(x_id)
            st = F.relu(torch.cat((st, -st), dim=1))
            st = self.st_net(st)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)'''

        return x, sldj


class NewNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features,20*math.ceil(math.log2(n_features)))
        self.fc5 = nn.Linear(20*math.ceil(math.log2(n_features)),2*math.ceil(math.log2(n_features)))
        self.fc6 = nn.Linear(2*math.ceil(math.log2(n_features)),2)
        self._init_weights(self.fc1)
        self._init_weights(self.fc5)
        self._init_weights(self.fc6)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.tanh(self.fc6(x))
        return x
    
    """def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,3)
        self.fc2 = nn.Linear(3,2)
        self.fc3 = nn.Linear(2,2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x"""
