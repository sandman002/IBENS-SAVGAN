import torch
import torch.nn as nn
import math
from einops import rearrange, reduce

class NystromAttention(nn.Module):
    def __init__(self, head_dim, num_landmarks, seq_len):
        super(NystromAttention, self).__init__()

        self.head_dim = head_dim
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len
        self.softmax = nn.Softmax(dim=-1)
        self.init_option = "original"
    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0. 
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, Q, K):
        l = math.ceil(self.seq_len / self.num_landmarks)
        # landmark_einops_eq = '... (n l) d -> ... n d'
        # Q_tilde = reduce(Q, landmark_einops_eq, 'sum', l = l)
        # K_tilde = reduce(K, landmark_einops_eq, 'sum', l = l)
        # divisor = l
        # Q_tilde /= divisor
        # K_tilde /= divisor


        # Q_tilde = Q.reshape(-1, self.num_landmarks, width*height*T // self.num_landmarks, self.channel_in).mean(dim = -2)
        # K_tilde = K.reshape(-1, self.num_landmarks, width*height*T // self.num_landmarks, self.channel_in).mean(dim = -2)
        # Q_tilde = Q[:, 1000:1000+self.num_landmarks, : ]
        # K_tilde = K[:, 1000:1000+self.num_landmarks, : ]


        # QQ = Q.clone().detach()
        # KK = K.clone().detach()
        # with torch.no_grad():
        #     energy =  torch.bmm(QQ,KK.permute(0,2,1)) # transpose check
        #     full_attention = self.softmax(energy)


        perm = torch.randperm(Q.size(1))
        idx = perm[:self.num_landmarks]
        Q_tilde = Q[:, idx, :]
        
        perm = torch.randperm(Q.size(1))
        idx = perm[:self.num_landmarks]
        K_tilde = K[:, idx, :]

        kernel1 = self.softmax(torch.bmm(Q,K_tilde.permute(0,2,1)))
        kernel2 = self.softmax(torch.bmm(Q_tilde,K_tilde.permute(0,2,1)))
        kernel3 = self.softmax(torch.bmm(Q_tilde, K.permute(0,2,1)))

        pinv_kernel2 = self.iterative_inv(kernel2)
        
        A_tilde = torch.bmm(torch.bmm(kernel1, pinv_kernel2),kernel3)

        att = (A_tilde)
       
        return att#, full_attention
