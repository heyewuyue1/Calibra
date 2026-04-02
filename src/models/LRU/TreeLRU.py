import math

import torch
from torch.nn import Linear, Module
from torch.nn.parameter import Parameter


class TreeLRU(Module):
    def __init__(self, in_features, out_features, state_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        self.input_proj = Linear(in_features, in_features)

        self.D = Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = Parameter(torch.log(-0.5 * torch.log(u1 + 1e-6)))
        self.theta_log = Parameter(torch.log(2 * math.pi * u2))
        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = Parameter(torch.log(torch.sqrt(torch.ones_like(lambda_mod) - torch.square(lambda_mod))))
        b_re = torch.randn(state_features, in_features) / math.sqrt(2 * in_features)
        b_im = torch.randn(state_features, in_features) / math.sqrt(2 * in_features)
        self.B = Parameter(torch.complex(b_re, b_im))
        c_re = torch.randn(out_features, state_features) / math.sqrt(state_features)
        c_im = torch.randn(out_features, state_features) / math.sqrt(state_features)
        self.C = Parameter(torch.complex(c_re, c_im))

        self._device_schedule_cache = {}

    def _compute_constants(self):
        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(self.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(self.theta_log))
        return torch.complex(lambda_re, lambda_im), torch.exp(self.gamma_log)

    def _device_schedule(self, schedule, device):
        cache_key = (schedule["cache_key"], device.type, device.index)
        cached = self._device_schedule_cache.get(cache_key)
        if cached is None:
            cached = [
                {
                    "batch": level["batch"].to(device, non_blocking=True),
                    "node": level["node"].to(device, non_blocking=True),
                    "left": level["left"].to(device, non_blocking=True),
                    "right": level["right"].to(device, non_blocking=True),
                }
                for level in schedule["levels"]
            ]
            self._device_schedule_cache[cache_key] = cached
        return cached

    def forward(self, batch):
        x_batch, idx_batch, schedule = batch

        device = self.B.device
        x_batch = x_batch.to(device)
        level_schedule = self._device_schedule(schedule, device)

        lambda_value, gamma = self._compute_constants()
        lambda_value = lambda_value.to(device)
        gamma = gamma.to(device)

        projected = self.input_proj(x_batch)
        projected_complex = projected.to(dtype=self.B.dtype)

        batch_size, max_nodes, _ = projected.shape
        states = torch.zeros(batch_size, max_nodes, self.state_features, dtype=torch.cfloat, device=device)
        outputs = torch.zeros(batch_size, max_nodes, self.out_features, dtype=projected.dtype, device=device)

        b_transposed = self.B.transpose(0, 1)
        c_transposed = self.C.transpose(0, 1)
        lambda_vector = lambda_value.unsqueeze(0)
        gamma_vector = gamma.unsqueeze(0)

        for schedule_level in level_schedule:
            if schedule_level["batch"].numel() == 0:
                continue

            batch_tensor = schedule_level["batch"]
            node_tensor = schedule_level["node"]
            left_tensor = schedule_level["left"]
            right_tensor = schedule_level["right"]

            child_states = torch.zeros(batch_tensor.shape[0], self.state_features, dtype=torch.cfloat, device=device)
            left_mask = left_tensor >= 0
            right_mask = right_tensor >= 0
            if torch.any(left_mask):
                child_states[left_mask] += states[batch_tensor[left_mask], left_tensor[left_mask]]
            if torch.any(right_mask):
                child_states[right_mask] += states[batch_tensor[right_mask], right_tensor[right_mask]]

            inputs = projected_complex[batch_tensor, node_tensor]
            input_term = torch.matmul(inputs, b_transposed)
            h_nodes = lambda_vector * child_states + gamma_vector * input_term
            y_nodes = torch.matmul(h_nodes, c_transposed).real

            states[batch_tensor, node_tensor] = h_nodes
            outputs[batch_tensor, node_tensor] = y_nodes

        return outputs, idx_batch, schedule


class TreeActivation(Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x[0]), x[1], x[2]


class TreeLayerNorm(Module):
    def forward(self, x):
        data, idxes, schedule = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return normd, idxes, schedule


class DynamicPooling(Module):
    def forward(self, x):
        pooled = torch.max(x[0], dim=1, keepdim=True).values
        return pooled.squeeze(1)
