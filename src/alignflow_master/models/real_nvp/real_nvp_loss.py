import numpy as np
import torch.nn as nn
import torch


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k
        self.volume = self._volumeNball(2, 3) # sigma = 1; TODO: generalize 2 to num_dimension !!! 

    def forward(self, z, sldj, weights = None):
        if weights == None:
            return self.lossNormals(z, sldj, weights)
        else:
            return self.lossNormals(z, sldj, weights) + self.lossAnomalies(z, sldj, weights)#*10
    
    def lossNormals(self, z, sldj, weights = None):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1)
        ll = prior_ll + sldj
        if weights == None:
            nll = -ll.mean()
        else:
            weights = 1.-weights
            ll = ll*weights
            nll = -ll.sum()
            nrm = weights.sum()
            if nrm == 0:
                return 0*nll
            else:
                return nll/nrm
            
        return nll
    
    def lossAnomalies(self, z, sldj, weights = None):
        V = self.volume
        prior_ll = z**2
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1)
        eps = 1e-7
        prior_ll = torch.log(1+eps-torch.exp(-0.5 * prior_ll))
        prior_ll = prior_ll - np.log(2*np.pi) - np.log(V/(2*np.pi)-1)

        ll = prior_ll + sldj
        if weights == None:
            nll = 0*ll.sum()
        else:
            ll = ll*weights
            nll = -ll.sum()
            nrm = weights.sum()
            if nrm == 0:
                return 0*nll
            else:
                return nll/nrm
            
        return nll

    def _volumeNball(self, n, R):
        if n == 0:
            return 1
        elif n==1:
            return 2*R
        else:
            return 2*np.pi*R*R*self._volumeNball(n-2, R)/n