import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from collections import deque
import matplotlib.pyplot as plt

# -------------------- 层次树结构定义 --------------------
class HierarchicalTree:
    def __init__(self, node_names, parent_idx):
        self.N = len(node_names)
        self.node_names = node_names
        self.parent = parent_idx
        self.children = [[] for _ in range(self.N)]
        for i, p in enumerate(parent_idx):
            if p != -1:
                self.children[p].append(i)
        self.root = parent_idx.index(-1) if -1 in parent_idx else 0

    def get_edge_index(self):
        edges = []
        for i, p in enumerate(self.parent):
            if p != -1:
                edges.append([p, i])
                edges.append([i, p])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

# -------------------- GNN层次分类模型 --------------------
class HierarchicalGNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, tree):
        super().__init__()
        self.tree = tree
        self.num_nodes = tree.N
        self.node_emb = nn.Embedding(self.num_nodes, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.node_bias = nn.Parameter(torch.zeros(self.num_nodes))
        self.register_buffer('edge_index', tree.get_edge_index())

    def forward(self, x):
        node_feat = self.node_emb.weight
        node_feat = self.conv1(node_feat, self.edge_index)
        node_feat = F.relu(node_feat)
        node_feat = self.conv2(node_feat, self.edge_index)
        sample_feat = self.fc(x)
        logits_node = torch.mm(sample_feat, node_feat.t()) + self.node_bias.unsqueeze(0)
        return logits_node

    def hierarchical_loss(self, logits, target, leaf_nodes):
        batch_size = logits.size(0)
        device = logits.device
        tree = self.tree

        # 构建从根到每个叶子的路径
        leaf_paths = []
        for leaf in leaf_nodes:
            path = []
            node = leaf
            while node != tree.root:
                parent = tree.parent[node]
                path.append((parent, node))
                node = parent
            path.reverse()
            leaf_paths.append(path)

        leaf_probs = []
        for path in leaf_paths:
            prob = torch.ones(batch_size, device=device)
            for parent, child in path:
                siblings = tree.children[parent]
                child_logits = logits[:, siblings]
                child_cond = F.softmax(child_logits, dim=1)
                child_pos = siblings.index(child)
                prob = prob * child_cond[:, child_pos]
            leaf_probs.append(prob)

        leaf_probs = torch.stack(leaf_probs, dim=1)
        loss = F.nll_loss(torch.log(leaf_probs + 1e-10), target)
        return loss


class NoGNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, tree):
        super().__init__()
        self.tree = tree
        self.num_nodes = tree.N
        self.node_emb = nn.Embedding(self.num_nodes, hidden_dim)
        #self.conv1 = GCNConv(hidden_dim, hidden_dim)
        #self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.node_bias = nn.Parameter(torch.zeros(self.num_nodes))
        self.register_buffer('edge_index', tree.get_edge_index())

    def forward(self, x):
        node_feat = self.node_emb.weight
        #node_feat = self.conv1(node_feat, self.edge_index)
        #node_feat = F.relu(node_feat)
        #node_feat = self.conv2(node_feat, self.edge_index)
        sample_feat = self.fc(x)
        logits_node = torch.mm(sample_feat, node_feat.t()) + self.node_bias.unsqueeze(0)
        return logits_node

    def hierarchical_loss(self, logits, target, leaf_nodes):
        batch_size = logits.size(0)
        device = logits.device
        tree = self.tree

        # 构建从根到每个叶子的路径
        leaf_paths = []
        for leaf in leaf_nodes:
            path = []
            node = leaf
            while node != tree.root:
                parent = tree.parent[node]
                path.append((parent, node))
                node = parent
            path.reverse()
            leaf_paths.append(path)

        leaf_probs = []
        for path in leaf_paths:
            prob = torch.ones(batch_size, device=device)
            for parent, child in path:
                siblings = tree.children[parent]
                child_logits = logits[:, siblings]
                child_cond = F.softmax(child_logits, dim=1)
                child_pos = siblings.index(child)
                prob = prob * child_cond[:, child_pos]
            leaf_probs.append(prob)

        leaf_probs = torch.stack(leaf_probs, dim=1)
        loss = F.nll_loss(torch.log(leaf_probs + 1e-10), target)
        return loss

# -------------------- 数据生成 --------------------
def generate_data(num_samples, input_dim, leaf_nodes):
    class_intervals = {
        0: (0, 25),    # 狗
        1: (25, 50),   # 猫
        2: (50, 75),   # 花
        3: (75, 100)   # 树
    }
    X = torch.zeros(num_samples, input_dim)
    y = torch.randint(0, len(leaf_nodes), (num_samples,))

    for i in range(num_samples):
        cls = y[i].item()
        start, end = class_intervals[cls]
        # 在对应区间填充较大的值
        X[i, start:end] = torch.randn(end - start) * 1.0 + 2.0
        # 添加全局噪声
        X[i, :] += torch.randn(input_dim) * 0.1
    return X, y

def shuffle_tensor(tensor, dim=0):
    """
    沿着指定维度打乱张量的顺序。
    
    参数：
        tensor (torch.Tensor): 输入张量。
        dim (int): 需要打乱的维度。
    
    返回：
        torch.Tensor: 打乱后的张量。
    """
    # 生成随机索引，确保与输入张量在同一设备上
    indices = torch.randperm(tensor.size(dim), device=tensor.device)
    
    # 使用 index_select 沿着指定维度重新排列
    return tensor.index_select(dim, indices)



def gen_shuffle_data(num_samples, input_dim, leaf_nodes):
    class_intervals = {
        0: (0, 25),    # 狗
        1: (25, 50),   # 猫
        2: (50, 75),   # 花
        3: (75, 100)   # 树
    }
    X = torch.zeros(num_samples, input_dim)
    y = torch.randint(0, len(leaf_nodes), (num_samples,))

    for i in range(num_samples):
        cls = y[i].item()
        start, end = class_intervals[cls]
        # 在对应区间填充较大的值
        X[i, start:end] = (torch.randn(end - start) * 0.5+2)*torch.randint(high=2,size=(end-start,))#1.0 + 2.0
        # 添加全局噪声

        X[i, :] += torch.randn(input_dim) * 0.1

    X=shuffle_tensor(X,1)
    
    return X, y

# -------------------- 训练 --------------------
if __name__ == "__main__":
    # 构建层次树
    node_names = ['根', '动物', '植物', '狗', '猫', '花', '树']
    parent_idx = [-1, 0, 0, 1, 1, 2, 2]
    tree = HierarchicalTree(node_names, parent_idx)
    leaf_nodes = [3, 4, 5, 6]

    input_dim = 100
    hidden_dim = 64
    model = HierarchicalGNNClassifier(input_dim, hidden_dim, tree)#NoGNNClassifier(input_dim, hidden_dim, tree)#

    # 生成数据
    X, y = gen_shuffle_data(200, input_dim, leaf_nodes)#generate_data(200, input_dim, leaf_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model(X)
        loss = model.hierarchical_loss(logits, y, leaf_nodes)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                # 计算准确率（复用路径概率计算）
                leaf_probs = []
                batch_size = X.size(0)
                device = X.device
                leaf_paths = []
                for leaf in leaf_nodes:
                    path = []
                    node = leaf
                    while node != tree.root:
                        parent = tree.parent[node]
                        path.append((parent, node))
                        node = parent
                    path.reverse()
                    leaf_paths.append(path)
                leaf_probs = []
                for path in leaf_paths:
                    prob = torch.ones(batch_size, device=device)
                    for parent, child in path:
                        siblings = tree.children[parent]
                        child_logits = logits[:, siblings]
                        child_cond = F.softmax(child_logits, dim=1)
                        child_pos = siblings.index(child)
                        prob = prob * child_cond[:, child_pos]
                    leaf_probs.append(prob)
                leaf_probs = torch.stack(leaf_probs, dim=1)
                pred = leaf_probs.argmax(dim=1)
                acc = (pred == y).float().mean()
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
