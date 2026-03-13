import json
import re 

class Node:
    """树节点类，存储分类单元信息"""
    def __init__(self, name, rank, info=None):
        self.name = name          # 分类单元名称
        self.rank = rank          # 分类阶元（如 'Kingdom', 'Phylum', 'Genus' 等）
        self.info = info if info is not None else {}   # 详细信息（字典）
        self.children = []        # 子节点列表
        self.parent=None
        self.images_index=None #only exist for leaf

        self.head_id=None   #classify head for this node

    def add_child(self, child):
        """添加子节点"""
        if child not in self.children:
            self.children.append(child)

    def is_leaf(self):
        return self.children==[]

    def __repr__(self):
        return f"Node({self.name}, {self.rank})"



class Tree:
    """分类树，管理多个根节点（界）"""
    def __init__(self):
        self.roots = []            # 根节点列表（通常为不同的界）

        self.class_name_map=None

        
    def add_root(self, node):
        """添加根节点"""
        if node not in self.roots:
            self.roots.append(node)

    def walk(self,node=None):
        child=[]
        if node==None:
            child=self.roots
        else:
            child=node.children

        for i in child:
            yield i
            for j in self.walk(i):
                yield j

    def get_leaf(self):
        res=[]
        for n in self.walk():
            if n.is_leaf():
                res.append(n)

        return res

    def lookup(self,names):
        '''look up the smallest nodes in the tree with names'''
        
        #l=self.get_leaf()
        #for i in l:
            #if i.name==name:
                #return i
        res=dict.fromkeys(names)
        for n in self.walk():
            #the smallest node will be visited last, so it ensures returning the smallest
            if n.name in names:
                res[n.name]=n
        return res

    def bind_dataset(self,mapjson):
        with open(mapjson,encoding='utf-8') as f:
            mp=json.load(f)
            mp2={}
            for index,longname in mp.items():
                engname=re.search(r'\((.*?)\)', longname).group(1)
                mp2[index]=engname
            nd=self.lookup(mp2.values())
            self.class_name_map=mp2
            nsuc=0
            for index,name in mp2.items():
                if nd[name] is None:
                    print('warnning!!! NOT FOUND: ',name,'index: ',index)
                    continue
                nd[name].images_index=index
                nsuc+=1
            print('successfully create index for ',nsuc,'nodes')

    def get_route(self,node):
        '''get a path from son to ancestors'''
        res=[node]
        while node.parent!=None:
            node=node.parent
            res.append(node)
        return res

    def get_cls_label(self,class_id):
        '''get label for a detection class
            return different heads' labels'''
        engname=self.class_name_map[class_id]
        leaf=self.lookup([engname])[engname]
        if engname is None:
            return None
        rt=self.get_route(leaf)
        res={}
        for i in rt:
            if not i.head_id in res:
                res[i.head_id]=i.name

        return res

    def display(self):
        """打印树结构（用于调试）"""
        def _print(node, level=0):
            indent = "  " * level
            info_summary = f" (info: {list(node.info.keys())})" if node.info else ""
            print(f"{indent}{node.rank}: {node.name}{info_summary}")
            for child in node.children:
                _print(child, level + 1)

        for root in self.roots:
            _print(root)

    @staticmethod
    def from_json(json_file):
        """
        从 JSON 文件构建分类树
        :param json_file: JSON 文件路径
        :return: Tree 对象
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tree = Tree()
        nodes = {}          # 缓存已创建的节点，键为 (名称, 阶元)

        nskip=0

        print('record items:',len(data))
        
        for name, item in data.items():
            # 跳过 info 为空的条目
            if not item.get('info'):
                nskip+=1
                continue

            record = item['info'][0]          # 取第一个记录
            rank = record.get('rank')
            if not rank:
                nskip+=1
                continue                      # 无阶元信息则跳过

            # 构建分类路径：从界到当前节点
            path = [
                (record.get('kingdom'), 'Kingdom'),
                (record.get('phylum'), 'Phylum'),
                (record.get('class'), 'Class'),
                (record.get('order'), 'Order'),
                (record.get('family'), 'Family'),
                (record.get('genus'), 'Genus'),
            ]
            # 如果是种，追加种节点（种名使用 JSON 的键，即全名）
            # 因为有些分类没到种
            if rank == 'Species':
                
                path.append((name, 'Species'))

            # 过滤掉缺失的层级
            path = [(n, r) for n, r in path if n is not None]

            parent = None
            for node_name, node_rank in path:
                key = (node_name, node_rank)

                # 判断当前节点是否为独立记录（即 JSON 中的名称且阶元匹配）
                is_own_record = (node_name == name and node_rank == rank)

                if key not in nodes:
                    # 创建新节点
                    if is_own_record:
                        # 独立记录：复制 record 并添加中文名
                        info = record.copy()
                        info['chn'] = item.get('chn', '')
                    else:
                        info = {}
                    node = Node(node_name, node_rank, info)
                    nodes[key] = node

                    # 如果是根节点（界），添加到树的根列表
                    if node_rank == 'Kingdom':
                        tree.add_root(node)
                else:
                    node = nodes[key]
                    # 如果当前是独立记录但节点已存在（如之前被中间节点创建），则更新 info
                    if is_own_record and not node.info:
                        node.info = record.copy()
                        node.info['chn'] = item.get('chn', '')

                # 将当前节点挂到父节点下
                if parent is not None:
                    parent.add_child(node)
                    node.parent=parent
                parent = node

        print('skip:',nskip)

        return tree

t=Tree.from_json(r'C:/Users/pytho/Desktop/mycode/proj/aicomp/plankton_det/classes.json')
#t.from_json(r'C:/Users/pytho/Desktop/mycode/proj/aicomp/plankton_det/classes.json')
t.bind_dataset(r'C:/Users/pytho/Desktop/mycode/proj/aicomp/Fuyo_YOLO_Dataset/Fuyo_YOLO_Dataset/浮游生物.json')

lt=t.get_route(t.lookup(['Leptodora'])['Leptodora'])

#define different classify head for different nodes
lt[0].head_id=1#属/种
lt[1].head_id=2#科
lt[2].head_id=2
lt[3].head_id=3#纲
lt[4].head_id=3
lt[5].head_id=3

print(t.get_cls_label('31'))
