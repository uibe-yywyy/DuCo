import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# different complementary loss
# mcl: output, partialY(Y/y_bar)
# scl: output, target(y_bar)

# let partial label prediction be larger
# log (mcl and scl)
def log_loss(outputs, partialY):
    k = partialY.shape[1] # num of classes
    can_num = partialY.sum(dim=1).float()  # n
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY
    average_loss = - ((k-1)/(k-can_num) * torch.log(final_outputs.sum(dim=1))).mean()
    return average_loss

def p_sum_loss(outputs, partialY):
    k = partialY.shape[1] # num of classes
    can_num = partialY.sum(dim=1).float()  # n
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss

def p_sum_loss_uw(outputs, partialY):
    K = partialY.shape[1] # num of classes
    can_num = partialY.sum(dim=1).float()  # n
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log((final_outputs*(1/partialY.sum(1).view(-1, 1).repeat(1, K))).sum(dim=1)).mean()
    return average_loss

def p_loss(outputs, partialY): # w/o weight
    pred = F.softmax(outputs, dim = 1)
    # confidence = softmax_p(output, partial_Y)
    average_loss = -((torch.sum(pred.log() * partialY, dim=1)) / partialY.sum(1)).mean()
    return average_loss

def p_loss_w(outputs, partialY): # w weight
    pred = F.softmax(outputs, dim = 1)
    confidence = softmax_p(outputs, partialY)
    average_loss = -(torch.sum(pred.log() * confidence, dim=1)) .mean()
    return average_loss

def softmax_p(output, partial_Y):
    z_exp = torch.exp(output)
    z_exp = z_exp * partial_Y
    return z_exp * (1/z_exp.sum(1).reshape(-1,1))

# let complementary label prediction be smaller


# weighted loss
def non_k_softmax_loss(f, K, labels):
    Q_1 = 1 - F.softmax(f, 1)
    Q_1 = F.softmax(Q_1, 1)
    labels = labels.long()
    return F.nll_loss(Q_1.log(), labels.long())  # Equation(5) in paper

def w_loss(f, K, labels):
    loss_class = non_k_softmax_loss(f=f, K=K, labels=labels)
    loss_w = w_loss_p(f=f, K=K, labels=labels)
    final_loss = loss_class + loss_w  # Equation(12) in paper
    return final_loss

def w_loss_p(f, K, labels):
    Q_1 = 1-F.softmax(f, 1)
    Q = F.softmax(Q_1, 1)
    q = torch.tensor(1.0) / torch.sum(Q_1, dim=1)
    q = q.view(-1, 1).repeat(1, K)
    w = torch.mul(Q_1, q)  # weight
    w_1 = torch.mul(w, Q.log())
    return F.nll_loss(w_1, labels.long())  # Equation(14) in paper

# combine loss
def combine_loss(f, K, labels, partialY, alpha=0.5): # log+l_w
    loss1 = log_loss(f, partialY)
    loss2 = w_loss(f, K, labels)
    return alpha*loss1 + loss2

def combine_loss2(f, K, labels, partialY, alpha=0.5): # p_l+l_w
    pred = F.softmax(f, dim=1)
    loss1 = -(((pred.log()*partialY).sum(1)) / (partialY.sum(1))).mean()
    loss2 = w_loss(f,K,labels)
    return alpha*loss1 + loss2

def combine_loss3(f, K, labels, partialY, alpha=0.5): # log+l_uw
    loss1 = log_loss(f, partialY)
    loss2 = non_k_softmax_loss(f, K, labels)
    return alpha*loss1 + loss2

def combine_loss4(f, K, labels, partialY, alpha=0.5): # l_w+p_lw
    pred = F.softmax(f, dim=1)
    q = torch.tensor(1.0) / torch.sum(pred, dim=1) 
    q = q.view(-1, 1).repeat(1, K)
    q1 = torch.mul(partialY, q)
    q2 = torch.tensor(1.0) / torch.sum(torch.mul(q1,pred), dim=1)
    q2 = q2.view(-1, 1).repeat(1, K)
    w = torch.mul(q2,torch.mul(q1,pred))
    w_1 = torch.mul(w, pred.log())
    loss1 = -w_1.sum(1).mean()
    loss2 = w_loss(f,K,labels)
    return alpha*loss1 + loss2

# free
def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)
# ga
def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)
# nn
def non_negative_loss(f, K, labels, ccp, beta): # logits, 10, mincom, ccp, beta=0
    ccp = torch.from_numpy(ccp).float().to(device)
    neglog = -F.log_softmax(f, dim=1)  # (bs, K)
    loss_vector = torch.zeros(K, requires_grad=True).to(device)
    temp_loss_vector = torch.zeros(K).to(device)
    for k in range(K):
        idx = (labels == k)
        if torch.sum(idx).item() > 0:
            idxs = idx.view(-1, 1).repeat(1, K)  # (bs, K)
            neglog_k = torch.masked_select(neglog, idxs).view(-1, K)
            temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            # ccp[k] or ccp, refer to https://github.com/takashiishida/comp/issues/2
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0)  # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1, 1), torch.zeros(K, requires_grad=True).view(-1, 1).to(device)-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss #, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)

# forward
def forward_loss(f, K, labels, reduction='mean'): # logits, 10, mincom
    Q = torch.ones(K, K) * 1/(K-1)  # uniform assumption
    Q = Q.to(device)
    for k in range(K):
        Q[k, k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long(), reduction=reduction)

# pc
def pc_loss(f, K, labels): # logits, 10, mincom
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = sigmoid(-1. * (f - fbar))  # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
    return pc_loss


# loss function for complementary label
def phi_loss(phi, logits, target, reduction='mean'):
    """
        Official implementation of "Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels"
        by Yu-Ting Chou et al.
    """
    if phi == 'lin':
        activated_prob = F.softmax(logits, dim=1)
    elif phi == 'quad':
        activated_prob = torch.pow(F.softmax(logits, dim=1), 2)
    elif phi == 'exp':
        activated_prob = torch.exp(F.softmax(logits, dim=1))
        # activated_prob = torch.exp(alpha * F.softmax(logits, dim=1) - (1 - alpha) * pred_outputs)
    elif phi == 'log':
        activated_prob = torch.log(F.softmax(logits, dim=1))
    elif phi == 'nl':
        activated_prob = -torch.log(1 - F.softmax(logits, dim=1) + 1e-5)
        # activated_prob = -torch.log(alpha * (1 - F.softmax(logits, dim=1) + 1e-5) + (1 - alpha) * pred_outputs)
    elif phi == 'hinge':
        activated_prob = F.softmax(logits, dim=1) - (1 / 10)
        activated_prob[activated_prob < 0] = 0
    else:
        raise ValueError('Invalid phi function')

    loss = -F.nll_loss(activated_prob, target, reduction=reduction)
    return loss

# scl_nl
def scl_nl(logits, target, reduction='mean'):
    activated_prob = -torch.log(1 - F.softmax(logits, dim=1) + 1e-5)
    loss = -F.nll_loss(activated_prob, target, reduction=reduction)
    return loss
# scl_log
def scl_log(logits, target, reduction='mean'):
    activated_prob = torch.log(F.softmax(logits, dim=1))
    loss = -F.nll_loss(activated_prob, target, reduction=reduction)
    return loss
# scl_exp
def scl_exp(logits, target, reduction='mean'):
    activated_prob = torch.exp(F.softmax(logits, dim=1))
    loss = -F.nll_loss(activated_prob, target, reduction=reduction)
    return loss
# scl_hinge
def scl_hinge(logits, target, reduction='mean'):    
    activated_prob = F.softmax(logits, dim=1) - (1 / 10)
    activated_prob[activated_prob < 0] = 0
    loss = -F.nll_loss(activated_prob, target, reduction=reduction)
    return loss


def relative_loss(f, K, labels): # logits, 10, mincom
    sigmoid = nn.Sigmoid()
    # print(f.gather(1, labels.long().view(-1, 1)))
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = -1.*torch.log(sigmoid((f - fbar)))  # multiply -1 for "complementary"
    # M1, M2 = K*(K-1)/2, K-1
    # print(loss_matrix)
    loss_matrix = torch.einsum('nk,nk->nk', loss_matrix, 1.-F.one_hot(labels)) # only preserve non-comp
    loss = torch.sum(loss_matrix)/((K-1)*len(labels))
    return loss    

def relative_loss_a(f, K, labels, a): # logits, 10, mincom
    sigmoid = nn.Sigmoid()
    # print(f.gather(1, labels.long().view(-1, 1)))
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = -1.*torch.log(sigmoid(a*(f - fbar)))  # multiply -1 for "complementary"
    # M1, M2 = K*(K-1)/2, K-1
    # print(loss_matrix)
    loss_matrix = torch.einsum('nk,nk->nk', loss_matrix, 1.-F.one_hot(labels)) # only preserve non-comp
    loss = torch.sum(loss_matrix)/((K-1)*len(labels))
    return loss

def relative_loss_exp(f, K, labels):
    sigmoid = torch.nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = torch.exp(-1*(f - fbar))  # multiply -1 for "complementary"
    loss_matrix = torch.einsum('nk,nk->nk', loss_matrix, 1.-F.one_hot(labels))
    loss = torch.sum(loss_matrix)/((K-1)*len(labels))
    return loss 


def relative_loss_log(f, K, labels):
    sigmoid = torch.nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = torch.log(1+torch.exp(-1*(f - fbar))) # multiply -1 for "complementary"
    loss_matrix = torch.einsum('nk,nk->nk', loss_matrix, 1.-F.one_hot(labels))
    # print(loss_matrix.sum(1)/w)   
    loss = torch.sum(loss_matrix)/((K-1)*len(labels))
    return loss 

def ce_loss(f, labels):
    f = F.softmax(f, dim=1)
    f = torch.log(f)
    l = F.one_hot(labels)*f
    return -l.sum(1).mean()

# ovr
def ovr_loss(f, labels): # logits, 10, mincom
    y_bar = F.one_hot(labels)
    y = 1. - F.one_hot(labels)
    fy_bar = (torch.log(1+torch.exp(f))*y_bar).sum(1)
    fy = (torch.log(1+torch.exp(-1.*f))*y).mean(1)
    loss = (fy_bar + fy).sum()/len(labels)
    return loss

def log_ce_loss(outputs, partialY, pseudo_labels, alpha): # 1-one_hot_comp_y
    # print(pseudo_labels)
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY  # \sum_{j\notin \bar{Y}} [p(j|x)]

    pred_outputs = sm_outputs[torch.arange(sm_outputs.size(0)), pseudo_labels]  # p(pl|x)
    # pred_outputs, _ = torch.max(final_outputs, dim=1)  # \max \sum_{j\notin \bar{Y}} [p(j|x)]

    average_loss = - ((k - 1) / (k - can_num) * torch.log(alpha * final_outputs.sum(dim=1) + (1 - alpha) * pred_outputs + 1e-10)).mean()

    return average_loss

def boost_loss(logits, s, y, alpha):
    f = F.softmax(logits, dim=1)
    f_bar = torch.log(1.-f+1e-5)
    f1 = f_bar*(1.-s)
    f2 = f_bar*(1.-F.one_hot(y, num_classes=10))
    l = -(alpha*(f1.sum(1))+(1-alpha)*(f2.sum(1)))
    return l.mean()

def plloss_ce(logits, s, y, alpha):
    f = F.log_softmax(logits, dim=1)
    f1 = f*s
    f2 = f*F.one_hot(y, num_classes=10)
    l = -(alpha*(f1.sum(1))+(1-alpha)*(f2.sum(1)))
    # l = l/s.sum(1)
    return l.mean()