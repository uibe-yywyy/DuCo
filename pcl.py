import numpy as np
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class PCL(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        
        self.proto_weight = args.proto_m # memontum update proto

        self.model = encoder() # student encoder

        self.register_buffer("prototypes_t", torch.zeros(10, 128)) # save proto_t 

        self.register_buffer("com_prototypes", torch.zeros(10, 128)) # save com_prototypes

    
    def set_prototype_update_weight(self, epoch, args):
        start = float(args.pro_weight_range[0])
        end = float(args.pro_weight_range[1])
        self.proto_weight = 1. * epoch / args.epochs * (end - start) + start # update weight with current epoch

    def save_prototypes(self,a):
        """
        Save prototypes to a file.
        """
        torch.save({
            'prototypes_t': self.prototypes_t,
            'com_prototypes': self.com_prototypes
        }, 'prototype')
        print(a)


    def forward(self, img1, img2, y): # y - com
        y = y.long()
        output, feat1 = self.model(img1) # logit and feat of teacher encoder
        output2, feat2 = self.model(img2)
        # output_t, feat_t = output_t.detach(), feat_t.detach()
        # feat = feat1.clone().detach()
        # output_s_mix, feat_s_mix = self.model_s(img_mix) # logit and feat of student encoder (mix-up)
        feat_1 = feat1.clone().detach()
        feat_2 = feat2.clone().detach()
        max_scores, pseudo_labels = torch.max(output, dim=1) # calculate pseudo label for image of teacher encoder
        # max_scores_s, pseudo_labels_s = torch.max(output_s, dim=1) # calculate pseudo label for image of student encoder
        
        # update proto_t
        for feat, label in zip(feat_1, pseudo_labels):
            self.prototypes_t[label] = self.proto_weight * self.prototypes_t[label] + (1 - self.proto_weight) * feat
        
        # update com_proto
        for feat, label in zip(feat_1, y):
            self.com_prototypes[label] = self.proto_weight * self.com_prototypes[label] + (1 - self.proto_weight) * feat

        # for feat, label in zip(feat_2, pseudo_labels):
        #     self.prototypes_t[label] = self.proto_weight * self.prototypes_t[label] + (1 - self.proto_weight) * feat

        self.prototypes_t = F.normalize(self.prototypes_t, p=2, dim=1)
        self.com_prototypes = F.normalize(self.com_prototypes, p=2, dim=1)

        # update proto_s
        # for feat, label in zip(feat_s, pseudo_labels_s):
        #     self.prototypes_s[label] = self.proto_weight * self.prototypes_s[label] + (1 - self.proto_weight) * label

        prototypes = self.prototypes_t.clone().detach() # get proto
        prototypes_com = self.com_prototypes.clone().detach() # get proto
        # print(prototypes.shape)
        # print(feat_s.shape)
        # print(output_s.shape)
        
        logits_prot = torch.mm(feat2, prototypes.t()) # calculate sim(feat_s, proto)
        logits_prot2 = torch.mm(feat1, prototypes.t()) 
        logits_com = torch.mm(feat2, prototypes_com.t())
        # logits_prot_s_mix = torch.mm(feat_s_mix, prototypes.t()) # calculate sim(feat_s_mix, proto)

        # self.prototypes_s = F.normalize(self.prototypes_s, p=2, dim=1)
        
        return output, output2, logits_prot, logits_prot2, pseudo_labels, logits_com, feat1

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    
    return output


@torch.no_grad()
def get_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-7) # avoid NaN
    return -(probs*log_probs).sum(1)