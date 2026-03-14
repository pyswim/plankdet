import json
from ete3 import Tree, TreeStyle, NodeStyle, TextFace
import os

def draw_taxonomy_tree(json_path, output_img="浮游生物分类树.png"):
    # 1. 定义阶元映射和颜色
    rank_map = {
        'kingdom': '界', 'phylum': '门', 'class': '纲',
        'order': '目', 'family': '科', 'genus': '属', 'species': '种'
    }
    ranks_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    color_palette = {
        '界': '#1f77b4', '门': '#ff7f0e', '纲': '#2ca02c',
        '目': '#d62728', '科': '#9467bd', '属': '#8c564b', '种': '#e377c2'
    }

    # 2. 读取 JSON
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 构建 Newick 字符串 (最稳妥的字符串构建法)
    # 我们手动构建一个嵌套字典来表示树
    tree_dict = {}

    for key, item in data.items():
        if 'info' not in item or not item['info']:
            continue
        entry = item['info'][0]
        
        # 提取当前路径
        current_level = tree_dict
        for r in ranks_order:
            val = entry.get(r)
            if r == 'species' and not val:
                val = entry.get('scientificname')
            
            if not val or val == '':
                continue
            
            # 生成带阶元的名字
            node_name = f"{rank_map[r]}-{val}"
            
            # 嵌套字典赋值
            if node_name not in current_level:
                current_level[node_name] = {}
            current_level = current_level[node_name]

    # 递归将嵌套字典转为 Newick 格式
    def dict_to_newick(d):
        if not d:
            return ""
        parts = []
        for name, children in d.items():
            sub = dict_to_newick(children)
            if sub:
                parts.append(f"({sub}){name}")
            else:
                parts.append(name)
        return ",".join(parts)

    newick_raw = dict_to_newick(tree_dict)
    newick_str = f"({newick_raw})Root;"  # 加一个总根

    #print(newick_str[:1000])
    
    # 4. 解析树并设置样式
    try:
        t = Tree(newick_str, format=1)
    except Exception as e:
        print(f"树解析错误: {e}")
        print("生成的Newick字符串:", newick_str)
        return

    # 遍历节点设置样式
    for node in t.traverse():
        # 初始化样式
        style = NodeStyle()
        style["shape"] = "circle"
        style["size"] = 0  # 默认隐藏节点圆点，保持整洁
        style["vt_line_color"] = "#333333"
        style["hz_line_color"] = "#333333"
        
        # 根据节点名称判断阶元并设置颜色
        node.add_feature("rank_cn", "Root") # 默认特征
        for rank_cn in color_palette:
            if node.name.startswith(f"{rank_cn}-"):
                style["size"] = 5
                style["fgcolor"] = color_palette[rank_cn]
                node.add_feature("rank_cn", rank_cn)
                if rank_cn == '种':
                    style["size"] = 7 # 种级放大
                break
        
        node.set_style(style)

    # 5. 设置布局 (使用 ete3 原生支持最好的方式)
    ts = TreeStyle()
    ts.show_leaf_name = False  # 显示叶子名
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.mode = "r"
    ts.branch_vertical_margin = 12
    ts.scale = 120
    
    # 标题
    ts.title.add_face(TextFace("浮游生物分类树", fsize=16, bold=True), column=0)
    
    # 关键：强制显示内部节点名称的布局函数
    def layout(node):
        if node.name != "Root"  :#and not node.is_leaf() :
            # 对于内部节点（界门纲目科属种），在分支上方添加文本
            tf = TextFace(node.name, fsize=12, bold=True, fgcolor="#555555")
            tf.margin_left = 5
            tf.margin_right = 5
            tf.margin_top = 2
            tf.margin_bottom = 2
            tf.border.width = 1
            tf.border.color = "#cccccc"
            node.add_face(tf, 1, position="branch-top")

    ts.layout_fn = layout

    # 6. 保存图片 (跳过不稳定的 show()，直接渲染)
    print(f"正在生成图片: {output_img} ...")
    try:
        # 直接渲染到文件，这比弹出窗口稳定得多
        t.show()
        t.render(output_img, tree_style=ts)#, w=1400, h=1000, dpi=300, units="px")
        print(f"成功！图片已保存至: {os.path.abspath(output_img)}")
    except Exception as e:
        print(f"渲染图片时出错: {e}")
        # 如果带布局的渲染失败，尝试最简模式保存
        print("尝试使用最简模式保存...")
        ts_simple = TreeStyle()
        ts_simple.show_leaf_name = True
        t.render(output_img.replace('.png', '_simple.png'), tree_style=ts_simple, dpi=300)

# --- 运行 ---
if __name__ == "__main__":
    # 请确保 classes.json 在同一目录下，或者修改下面的路径
    json_file = "classes.json" 
    draw_taxonomy_tree(json_file)
