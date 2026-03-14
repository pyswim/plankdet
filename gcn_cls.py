import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class HierarchicalGCNPyG(nn.Module):
    """
    基于PyG GCN的层次分类器。
    输入特征 x (batch_size, input_dim)，输出每个节点的路径概率 (batch_size, num_nodes)。
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16, 8], num_layers=5, dropout=0.2):
        super().__init__()
        # 构建树结构（节点数、父子关系、子节点列表）
        self.num_nodes, self.parent, self.children = self._build_tree()
        # 构建无向边索引（不包含自环，GCNConv内部会自动添加）
        self.register_buffer('edge_index', self._build_edge_index())

        # 构建GCN层（输入通道 → 隐藏层 → 输出1维）
        layers = []
        in_ch = input_dim
        for i in range(num_layers - 1):
            out_ch = hidden_dims[i] if i < len(hidden_dims) else 1
            layers.append(GCNConv(in_ch, out_ch))
            in_ch = out_ch
        layers.append(GCNConv(in_ch, 1))   # 最后一层输出每个节点的1维logits
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        返回: (batch_size, num_nodes) 每个节点的路径概率
        """
        batch_size = x.size(0)
        N = self.num_nodes
        device = x.device

        # 将每个样本的特征复制到所有节点: (batch*N, input_dim)
        node_feat = x.unsqueeze(1).expand(-1, N, -1).reshape(-1, x.size(-1))

        # 为每个样本复制边索引，并添加节点偏移量以实现批量处理
        # edge_index 原始形状 (2, E)，需要扩展为 (2, E * batch_size)
        edge_index = self.edge_index                # (2, E)
        E = edge_index.size(1)
        # 每个样本的节点偏移量 = 样本索引 * N
        offsets = torch.arange(batch_size, device=device) * N   # (batch_size,)
        # 将偏移应用到每条边上
        offset_per_edge = offsets.repeat_interleave(E)          # (E*batch_size,)
        edge_index_batch = edge_index.repeat(1, batch_size) + offset_per_edge.unsqueeze(0)

        # 通过GCN层（除最后一层外都加ReLU和Dropout）
        for i, layer in enumerate(self.layers):
            node_feat = layer(node_feat, edge_index_batch)
            if i < len(self.layers) - 1:   # 中间层
                node_feat = F.relu(node_feat)
                node_feat = self.dropout(node_feat)

        # node_feat: (batch*N, 1) → 重塑为 (batch, N)
        logits = node_feat.squeeze(-1).view(batch_size, N)   # (batch, N)

        # 计算路径概率（层次softmax）
        path_probs = self._compute_path_probs(logits)
        return path_probs

    def _build_tree(self):
        """定义分类树的节点索引及父子关系（与之前相同）"""
        nodes = [
            "哺乳动物纲", "灵长目", "食肉目", "长鼻目", "啮齿目",
            "人科", "猩猩科", "猫科", "象科", "鼠科", "松鼠科",
            "人属", "黑猩猩属", "大猩猩属", "豹属", "象属", "小鼠属", "松鼠属",
            "人类", "黑猩猩", "大猩猩", "狮子", "老虎", "豹", "亚洲象", "非洲象", "家鼠", "松鼠"
        ]
        num_nodes = len(nodes)
        parent = [-1, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 15, 15, 16, 17]
        children = [[] for _ in range(num_nodes)]
        for child, p in enumerate(parent):
            if p != -1:
                children[p].append(child)
        return num_nodes, parent, children

    def _build_edge_index(self):
        """根据父子关系构建无向边索引（用于GCN消息传递）"""
        edges = []
        for child, p in enumerate(self.parent):
            if p != -1:
                edges.append([p, child])
                edges.append([child, p])   # 无向边
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index

    def _compute_path_probs(self, logits):
        """
        将节点logits转换为层次化的路径概率。
        对每个内部节点的子节点logits做softmax，然后从根节点向下累积。
        logits: (batch_size, num_nodes)
        返回: (batch_size, num_nodes) 路径概率（满足叶子节点概率和为1）
        """
        batch_size = logits.size(0)
        N = self.num_nodes
        device = logits.device

        path_probs = torch.zeros(batch_size, N, device=device)
        path_probs[:, 0] = 1.0   # 根节点路径概率为1

        from collections import deque
        q = deque([0])
        visited = {0}
        while q:
            p = q.popleft()
            children = self.children[p]
            if not children:
                continue
            child_logits = logits[:, children]                 # (batch, c)
            cond_probs = F.softmax(child_logits, dim=-1)       # (batch, c)
            parent_prob = path_probs[:, p].unsqueeze(1)        # (batch, 1)
            child_probs = parent_prob * cond_probs             # (batch, c)
            for i, child in enumerate(children):
                path_probs[:, child] = child_probs[:, i]
                if child not in visited:
                    q.append(child)
                    visited.add(child)
        return path_probs


# 示例用法
if __name__ == "__main__":
    input_dim = 6   # 体长、体重、脑容量、寿命、奔跑速度、食性
    model = HierarchicalGCNPyG(input_dim, hidden_dims=[64, 32, 16, 8], num_layers=5)

    # 生成随机输入（batch_size=4）
    x = torch.randn(4, input_dim)
    probs = model(x)

    print("输出路径概率形状:", probs.shape)   # (4, 28)
    # 验证根节点概率为1
    print("根节点概率:", probs[0, 0].item())
    # 验证叶子节点概率之和为1
    leaf_indices = [i for i in range(model.num_nodes) if len(model.children[i]) == 0]
    leaf_sum = probs[0, leaf_indices].sum().item()
    print("第一个样本的叶子节点概率之和:", leaf_sum)