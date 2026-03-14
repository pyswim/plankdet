import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------- 数据生成 --------------------
def generate_biological_dataset(n_samples_per_species=10, noise_scale=0.01, random_state=42):
    np.random.seed(random_state)
    species_data = [
        ("人类",       "哺乳动物纲", "灵长目", "人科",   "人属",       170,  70,  1300, 80, 20, 3),
        ("黑猩猩",     "哺乳动物纲", "灵长目", "人科",   "黑猩猩属",   140,  50,   400, 50, 30, 3),
        ("大猩猩",     "哺乳动物纲", "灵长目", "猩猩科", "大猩猩属",   160, 150,   500, 40, 25, 2),
        ("狮子",       "哺乳动物纲", "食肉目", "猫科",   "豹属",       200, 200,   250, 15, 50, 1),
        ("老虎",       "哺乳动物纲", "食肉目", "猫科",   "豹属",       250, 200,   300, 20, 40, 1),
        ("豹",         "哺乳动物纲", "食肉目", "猫科",   "豹属",       120,  70,   200, 12, 60, 1),
        ("亚洲象",     "哺乳动物纲", "长鼻目", "象科",   "象属",       600,4500,  5000, 60, 20, 2),
        ("非洲象",     "哺乳动物纲", "长鼻目", "象科",   "象属",       650,5500,  6000, 70, 25, 2),
        ("家鼠",       "哺乳动物纲", "啮齿目", "鼠科",   "小鼠属",       8, 0.03,    1,  2,  8, 3),
        ("松鼠",       "哺乳动物纲", "啮齿目", "松鼠科", "松鼠属",      25,  0.7,    5, 10, 15, 2),
    ]
    rows = []
    for item in species_data:
        species, class_, order, family, genus, length, weight, brain, lifespan, speed, diet = item
        for _ in range(n_samples_per_species):
            length_noise = np.random.normal(0, length * noise_scale)
            weight_noise = np.random.normal(0, weight * noise_scale)
            brain_noise = np.random.normal(0, brain * noise_scale)
            lifespan_noise = np.random.normal(0, lifespan * noise_scale)
            speed_noise = np.random.normal(0, speed * noise_scale)
            row = {
                "纲": class_, "目": order, "科": family, "属": genus, "种": species,
                "体长(cm)": max(0, length + length_noise),
                "体重(kg)": max(0, weight + weight_noise),
                "脑容量(cm³)": max(0, brain + brain_noise),
                "寿命(年)": max(0, lifespan + lifespan_noise),
                "奔跑速度(km/h)": max(0, speed + speed_noise),
                "食性": diet,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = ["体长(cm)", "体重(kg)", "脑容量(cm³)", "寿命(年)", "奔跑速度(km/h)", "食性"]
    return df, feature_cols

# -------------------- 节点名称映射 --------------------
NODE_NAMES = [
    "哺乳动物纲",  # 0
    "灵长目",      # 1
    "食肉目",      # 2
    "长鼻目",      # 3
    "啮齿目",      # 4
    "人科",        # 5
    "猩猩科",      # 6
    "猫科",        # 7
    "象科",        # 8
    "鼠科",        # 9
    "松鼠科",      # 10
    "人属",        # 11
    "黑猩猩属",    # 12
    "大猩猩属",    # 13
    "豹属",        # 14
    "象属",        # 15
    "小鼠属",      # 16
    "松鼠属",      # 17
    "人类",        # 18
    "黑猩猩",      # 19
    "大猩猩",      # 20
    "狮子",        # 21
    "老虎",        # 22
    "豹",          # 23
    "亚洲象",      # 24
    "非洲象",      # 25
    "家鼠",        # 26
    "松鼠"         # 27
]
species_to_idx = {name: idx for idx, name in enumerate(NODE_NAMES) if name in ["人类","黑猩猩","大猩猩","狮子","老虎","豹","亚洲象","非洲象","家鼠","松鼠"]}

# -------------------- Dataset --------------------
class BioDataset(Dataset):
    def __init__(self, df, feature_cols, species_to_idx):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor([species_to_idx[name] for name in df["种"]], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# -------------------- 模型定义（修正版，使用可微的路径概率计算）--------------------
class HierarchicalGCNPyG(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16, 8], num_layers=5, dropout=0.2):
        super().__init__()
        self.num_nodes, self.parent, self.child_list, self.node_names = self._build_tree()
        # 预先计算每个节点的路径信息
        self.node_paths = self._compute_node_paths()
        self.edge_index = self._build_edge_index()
        self.register_buffer('edge_index_buffer', self.edge_index)

        layers = []
        in_ch = input_dim
        for i in range(num_layers - 1):
            out_ch = hidden_dims[i] if i < len(hidden_dims) else 1
            layers.append(GCNConv(in_ch, out_ch))
            in_ch = out_ch
        layers.append(GCNConv(in_ch, 1))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _build_tree(self):
        nodes = NODE_NAMES
        num_nodes = len(nodes)
        parent = [-1, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 15, 15, 16, 17]
        child_list = [[] for _ in range(num_nodes)]
        for child, p in enumerate(parent):
            if p != -1:
                child_list[p].append(child)
        return num_nodes, parent, child_list, nodes

    def _compute_node_paths(self):
        """为每个节点计算从根到该节点的路径（父节点和子节点位置）"""
        N = self.num_nodes
        paths = [[] for _ in range(N)]
        for i in range(1, N):
            node = i
            path = []
            while node != 0:
                parent = self.parent[node]
                child_idx = self.child_list[parent].index(node)
                path.append((parent, child_idx))
                node = parent
            paths[i] = path[::-1]  # 反转，使从根到节点
        return paths

    def _build_edge_index(self):
        edges = []
        for child, p in enumerate(self.parent):
            if p != -1:
                edges.append([p, child])
                edges.append([child, p])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index

    def forward(self, x):
        batch_size = x.size(0)
        N = self.num_nodes
        device = x.device

        node_feat = x.unsqueeze(1).expand(-1, N, -1).reshape(-1, x.size(-1))
        edge_index = self.edge_index_buffer
        E = edge_index.size(1)
        offsets = torch.arange(batch_size, device=device) * N
        offset_per_edge = offsets.repeat_interleave(E)
        edge_index_batch = edge_index.repeat(1, batch_size) + offset_per_edge.unsqueeze(0)

        for i, layer in enumerate(self.layers):
            node_feat = layer(node_feat, edge_index_batch)
            if i < len(self.layers) - 1:
                node_feat = F.relu(node_feat)
                node_feat = self.dropout(node_feat)

        logits = node_feat.squeeze(-1).view(batch_size, N)   # (batch, N)
        path_probs = self._compute_path_probs(logits)
        return path_probs, logits

    def _compute_path_probs(self, logits):
        """可微的路径概率计算，无就地修改"""
        batch_size = logits.size(0)
        N = self.num_nodes
        device = logits.device

        # 预先计算所有内部节点的条件概率
        internal_nodes = [i for i in range(N) if self.child_list[i]]
        cond_probs = {}
        for p in internal_nodes:
            children = self.child_list[p]
            child_logits = logits[:, children]  # (batch, c)
            cond = F.softmax(child_logits, dim=-1)  # (batch, c)
            cond_probs[p] = cond

        # 计算每个节点的路径概率
        node_probs = []
        # 根节点
        node_probs.append(torch.ones(batch_size, 1, device=device))
        # 其他节点
        for i in range(1, N):
            path = self.node_paths[i]
            prob = torch.ones(batch_size, device=device)
            for parent, child_idx in path:
                cond = cond_probs[parent]  # (batch, num_children)
                child_prob = cond[:, child_idx]  # (batch,)
                prob = prob * child_prob
            node_probs.append(prob.unsqueeze(1))

        path_probs = torch.cat(node_probs, dim=1)  # (batch, N)
        return path_probs
def build_path(leaf):
    parent = [-1, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 15, 15, 16, 17]
    cur=leaf
    res=[]
    while cur!=-1:
        res.append(cur)
        cur=parent[cur]
    return res
    

class simpleCls(nn.Module):
    def __init__(self,hiddens):
        super().__init__()
        self.layers=[]
        self.layers.append(nn.Linear(6,hiddens[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hiddens)-1):
            self.layers.extend([nn.Linear(hiddens[i],hiddens[i+1]),nn.ReLU()])
        self.layers.extend([nn.Linear(hiddens[-1],28)])
        self.layers=nn.Sequential( *self.layers)

    def forward(self,x):
        return self.layers(x)

    def mytrain(self, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
        #model = model.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        cri=nn.CrossEntropyLoss()
        for epoch in range(1, epochs+1):
            self.train()
            total_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                path_probs= self(features)
                target_probs = torch.zeros_like(path_probs)#path_probs[torch.arange(len(labels)), labels] + 1e-10
                for i in range(labels.shape[0]):
                    target_probs[i, build_path(labels[i])]=1
                #print('labels',labels)
                #print('path_probs',path_probs.shape)
                #print('target_probs',target_probs)
                #return False
                
                #loss = torch.norm(path_probs-target_probs,dim=1).mean()#-torch.log(target_probs).mean()
                #loss=cri(path_probs,labels)
                loss=get_cls_loss(path_probs,labels)
                #print('loss',loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(features)

            avg_loss = total_loss / len(train_loader.dataset)
            #val_acc = evaluate(self, val_loader, device)
            print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f}")# | Val Acc: {val_acc:.2f}%")
            scheduler.step()
    
sim=simpleCls([128,128,64,32,16])

def get_leaf_probs(pred):
    '''pred: Tensor(N,28)-->predicted node logits
        return-->Tensor(N,28) returns the probs for all leafs, 0 for non-leaf
        '''
    parent = [-1, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 15, 15, 16, 17]
    lg_soft=torch.log_softmax(pred,1)
    res=torch.zeros_like(pred)
    res[:,0]=1.
    for i in range(1,28):
        res[:,i]=pred[:,i]+res[:,parent[i]]
    return res
def get_cls_loss(pred,labels):
    leaf_log_probs=get_leaf_probs(pred)
    res=torch.zeros(leaf_log_probs.shape[0])
    for i in range(labels.shape[0]):
        res[i]=leaf_log_probs[i][labels[i]]
    return -res.mean()
    

    

# -------------------- 训练函数 --------------------
def train(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    cri=nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            path_probs, _ = model(features)
            target_probs = torch.zeros_like(path_probs)#path_probs[torch.arange(len(labels)), labels] + 1e-10
            for i in range(labels.shape[0]):
                target_probs[i, build_path(labels[i])]=1
            #print('labels',labels)
            #print('path_probs',path_probs.shape)
            #print('target_probs',target_probs)
            loss=cri(path_probs,labels)
            #loss = cri(path_probs,target_probs)#torch.norm(path_probs-target_probs,dim=1).mean()#-torch.log(target_probs).mean()
            #print('loss',loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            path_probs, _ = model(features)
            # 叶子节点索引为18~27
            preds = torch.argmax(path_probs[:, 18:], dim=1) + 18
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# -------------------- 主程序 --------------------

if __name__ == "__main__":
    df, feature_cols = generate_biological_dataset(n_samples_per_species=20, noise_scale=0.1)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["种"])

    train_dataset = BioDataset(train_df, feature_cols, species_to_idx)
    val_dataset = BioDataset(val_df, feature_cols, species_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    input_dim = len(feature_cols)  # 6
    model = HierarchicalGCNPyG(input_dim, hidden_dims=[128, 64, 32, 64,32,8], num_layers=4, dropout=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    sim.mytrain( train_loader, val_loader, epochs=500, lr=0.001, device=device)

    
    #train(model, train_loader, val_loader, epochs=100, lr=0.001, device=device)

    #torch.save(model.state_dict(), "hierarchical_gcn.pth")
    
def test(x=[240, 210, 290, 18, 42, 1]):
    new_sample = x
    species, prob, all_probs = predict_species(model, new_sample, device)
    print(f"预测物种: {species}")

    print(f"置信度: {prob:.4f}")

    leaf_names = NODE_NAMES[18:]
    for name, p in zip(leaf_names, all_probs):
            print(f"  {name}: {p:.4f}")
def predict_species(model, features, device='cpu'):
    """
    对单个样本进行推理
    features: list or numpy array of 6 features in order: 体长,体重,脑容量,寿命,奔跑速度,食性
    returns: predicted species name and its probability
    """
    model.eval()
    # 转换为tensor并增加batch维度
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1,6)
    with torch.no_grad():
        path_probs, _ = model(x)  # (1, num_nodes)
    
    # 叶子节点索引范围（根据之前的定义，叶子节点是18-27）
    leaf_start = 18
    leaf_probs = path_probs[0, leaf_start:].cpu().numpy()
    leaf_indices = np.arange(leaf_start, leaf_start + len(leaf_probs))
    
    # 找到概率最大的叶子节点
    best_idx = leaf_indices[np.argmax(leaf_probs)]
    best_prob = np.max(leaf_probs)
    species_name = NODE_NAMES[best_idx]
    
    return species_name, best_prob, leaf_probs
def test_all_cat(features):
    dic={"人类":18,
    "黑猩猩":19,
    "大猩猩":20,
    "狮子":21,
    "老虎":22,
    "豹":23,
    "亚洲象":24,
    "非洲象":25,
    "家鼠":26,
    "松鼠":27}
    r=sim(torch.Tensor(features))
    for i,j in dic.items():
        print(i,r[build_path(j)])

