from torch import nn
from LRU.TreeLRU import _flatten_tree_batch, TreeLRU, TreeLayerNorm, TreeActivation, DynamicPooling
import torch
from torch import optim

# ===== 堆叠多层 TreeLRU 网络 =====
class TreeLRUNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.model = nn.Sequential(
            TreeLRU(in_features, 128, 64),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            TreeLRU(128, 64, 64),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, tree):
        x = tree
        for layer in self.model:
            x = layer(x)
        return x

# ===== 构造一个随机二叉树 =====
tree1 = (
    (0, 1),
    ((1, 2), ((0, 1),), ((-1, 0),)),
    ((-3, 0), ((2, 3),), ((1, 2),))
)

tree2 = (
    (16, 3),
    ((0, 1), ((5, 3),), ((2, 6),)),
    ((2, 9),)
)


trees = [tree1, tree2]

# ===== 训练 Demo =====
model = TreeLRUNet(in_features=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 随机目标
target = torch.tensor([1.0, 100.0]).view(-1, 1)

for step in range(200):
    flattened_trees = _flatten_tree_batch(trees)
    pred = model(flattened_trees)
    loss = loss_fn(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step+1) % 50 == 0:
        print(f"Step {step+1}, Loss={loss.item():.6f}")

# 测试输出
out_test = model(flattened_trees)
print("TreeLRUNet 输出:", out_test)