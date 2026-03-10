import torch
from torch.nn.parameter import Parameter
from torch.nn import Module, Linear
import math

# ===== Tree-LRU 层 =====
class TreeLRU(Module):
    def __init__(self, in_features, out_features, state_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        self.input_proj = Linear(in_features, in_features)

        # LRU 参数初始化
        self.D = Parameter(torch.randn(out_features, in_features)/math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = Parameter(torch.log(-0.5*torch.log(u1 + 1e-6)))
        self.theta_log = Parameter(torch.log(2*math.pi*u2))
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod)-torch.square(Lambda_mod))))
        B_re = torch.randn(state_features, in_features)/math.sqrt(2*in_features)
        B_im = torch.randn(state_features, in_features)/math.sqrt(2*in_features)
        self.B = Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn(out_features, state_features)/math.sqrt(state_features)
        C_im = torch.randn(out_features, state_features)/math.sqrt(state_features)
        self.C = Parameter(torch.complex(C_re, C_im))
    
    def _forward_single_tree(self, x, idx):
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im)
        gamma = torch.exp(self.gamma_log)

        def recur(i):
            x_node = x[i]  # 直接取 [token_feature]，因为 x 是 (N, F)
            x_node_proj = self.input_proj(x_node)
            x_node_proj = x_node_proj.to(self.B.device).view(-1)

            left_state, left_outputs = recur(idx[i][0]) if idx[i][0] is not None else (None, None)
            right_state, right_outputs = recur(idx[i][1]) if idx[i][1] is not None else (None, None)

            device = self.B.device
            h_children = torch.zeros(self.state_features, dtype=torch.cfloat, device=device)
            Lambda_device = Lambda.to(device)
            gamma_device = gamma.to(device)
            if left_state is not None:
                h_children += left_state.to(device)
            if right_state is not None:
                h_children += right_state.to(device)

            h_node = Lambda_device * h_children + gamma_device * (self.B @ x_node_proj.to(dtype=self.B.dtype))
            y_node = (self.C @ h_node).real.unsqueeze(0)

            tree_output = y_node
            if left_outputs is not None:
                tree_output = torch.cat((tree_output, left_outputs), dim=0)
            if right_outputs is not None:
                tree_output = torch.cat((tree_output, right_outputs), dim=0)
            return h_node, tree_output

        return recur(0)  # 从根节点 index 0 开始

    def forward(self, batch):
        x_batch, idx_batch = batch  # x_batch.shape = (B, N, F), idx_batch.shape = (B, N, 2)
        all_outputs = []
        all_idxes = []

        for b in range(x_batch.shape[0]):  # 遍历每个样本
            x = x_batch[b]
            idx = idx_batch[b]
            _, node_outputs = self._forward_single_tree(x, idx)
            all_outputs.append(node_outputs)
            all_idxes.append(idx)

        # 找到 batch 内最大节点数
        max_nodes = max(o.shape[0] for o in all_outputs)
        feat_dim = all_outputs[0].shape[1]

        # 堆叠成 batch 维度
        # Padding outputs
        padded_outputs = []
        for o in all_outputs:
            pad = torch.zeros(max_nodes, feat_dim, dtype=o.dtype, device=o.device)
            pad[:o.shape[0]] = o
            padded_outputs.append(pad)
        all_outputs = torch.stack(padded_outputs, dim=0)  # [B, max_nodes, out_features]

        # Padding idxes
        padded_idxes = []
        for idx in all_idxes:
            pad = [(None, None)] * max_nodes
            pad[:len(idx)] = idx
            padded_idxes.append(pad)
        all_idxes = padded_idxes  # 保留为 list of list of tuples     # [B, N, 2]

        return all_outputs, all_idxes

class TreeActivation(Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x[0]), x[1]


class TreeLayerNorm(Module):
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)


class DynamicPooling(Module):
    def forward(self, x):
        # dim=1 是节点维度（7）
        pooled = torch.max(x[0], dim=1, keepdim=True).values  # shape = (2, 1, 64)
        pooled = pooled.squeeze(1)  # shape = (2, 64)
        return pooled