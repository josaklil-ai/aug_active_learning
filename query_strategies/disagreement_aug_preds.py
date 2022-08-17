import numpy as np
import torch
from torchvision import transforms
from .strategy import Strategy


class DAPSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
        super(DAPSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

        self.transform = transforms.AugMix()

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        
        print(self.X)
        X_aug = self.X[idxs_unlabeled]
        Y_aug = self.Y[idxs_unlabeled]
        
        probs, embeddings = self.predict_prob_embed(X_aug, Y_aug)
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        selected = U.sort()[1][:n]
        
        return idxs_unlabeled[selected], embeddings, probs.max(1)[1], probs, selected, None
