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
    def __init__(self, n_features, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k
        self.n_features = n_features
        self.volume = self._volumeNball(self.n_features, 5) # sigma = 1; TODO: generalize 2 to num_dimension !!! 

    def forward(self, z, sldj, weights = None):
        if weights == None:
            return self.lossNormals(z, sldj, weights)
        else:
            return self.lossNormals(z, sldj, weights) + self.lossAnomalies(z, sldj, weights)#*10
    
    def _unc(self, weights):
        unc = -weights*torch.log2(weights)-(1-weights)*torch.log2(1-weights)
        unc[weights == 0] = 0
        unc[weights == 1] = 1
        return unc[weights<.5]/weights[weights<.5].size(0), unc[weights>.5]/weights[weights>.5].size(0)

    def lossNormals(self, z, sldj, weights = None):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1)
        ll = prior_ll + sldj

        """weights = 1-weights
        ll = ll*weights
        return -ll.sum() """      
     
        if weights == None:
            nll = -ll.mean()
        else:
            #newWeights = torch.zeros_like(weights)
            #newWeights[weights<=.5] = weights[weights<= .5]
            newWeights = weights
            newWeights = 1.-newWeights
            ll = ll*newWeights
            nll = -ll.sum()
            nrm = newWeights.sum()
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
        
        """if weights == None:
            return -ll.mean()"""
        Uneg, Upos = self._unc(weights)
        """ll = ll*weights
        res = -ll.sum()"""
        Uneg = Uneg.sum()
        Upos = Upos.sum()    
    
        if weights == None:
            nll = 0*ll.sum()
        else:
            """newWeights = torch.zeros_like(weights)
            newWeights[weights>=.5] = weights[weights>= .5]"""
            newWeights = weights
            ll = ll*newWeights
            nll = -ll.sum()
            nrm = newWeights.sum()
            if nrm == 0:
                return 0*nll
            else:
                res = nll/nrm
                res = res*Uneg
                res = res/(Uneg+Upos)
                return res
                
        return nll

    def _volumeNball(self, n, R):
        if n == 0:
            return 1
        elif n==1:
            return 2*R
        else:
            return 2*np.pi*R*R*self._volumeNball(n-2, R)/n