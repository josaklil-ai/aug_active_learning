import numpy as np
import torch
from .strategy import Strategy

from torchvision import transforms
import copy

class DAPSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
        super(DAPSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

        self.transform = transforms.AugMix()
        self.A = 5

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        
        X_orig_unlb = self.X[idxs_unlabeled].unsqueeze(1)
        Y_orig_unlb = self.Y[idxs_unlabeled].unsqueeze(1)
        
        X_unlb = []
        Y_unlb = []
        
        for i in range(self.A):
            X_aug = self.transform(X_orig_unlb)
            Y_aug = copy.deepcopy(Y_orig_unlb)
            X_unlb.append(X_aug)
            Y_unlb.append(Y_aug)
            
        X_unlb = torch.cat(X_unlb)
        Y_unlb = torch.cat(Y_unlb)

        probs, embeddings = self.predict_prob_embed(X_orig_unlb, Y_orig_unlb)
        probs_aug, embeddings_aug = self.predict_prob_embed(X_unlb, Y_unlb)
   
        loss = torch.nn.CrossEntropyLoss()
    
        cross_entropies = torch.zeros(len(probs))
        for i in range(len(probs)):
            ce_avg = 0
            for j in range(self.A):
                ce = loss(probs_aug, probs[i])
                ce_avg += ce     
            cross_entropies[i] = ce_avg / self.A

        selected = cross_entropies.sort()[1][:n]
        
        return idxs_unlabeled[selected], embeddings, probs.max(1)[1], probs, selected, None
