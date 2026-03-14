import io
import base64
from PIL import Image
from nicegui import ui, app
from ultralytics import YOLO

# ================== 配置 ==================
MODEL_PATH = "best.pt"
CLASS_NAMES_EN = [
    "Chydorus",
    "Ceriodaphnia",
    "Daphnia",
    "Scapholeberis",
    "Holopedium",
    "Leptodora",
    "Moina",
    "Diaphanosoma",
    "Sinocalanus",
    "Neodiaptomus",
    "Neutrodiaptomus",
    "Sinodiaptomus"
]
CLASS_NAMES_CN = [
    "盘肠溞属",
    "网纹溞属",
    "溞属",
    "船卵溞属",
    "圆囊溞属",
    "薄皮溞属",
    "裸腹溞属",
    "透明溞属",
    "华哲水蚤属",
    "新镖水蚤属",
    "中镖水蚤属",
    "华哲水蚤属"
]
INTROS = [
    "盘肠溞属（Chydorus）：小型底栖枝角类，壳瓣近圆形，常栖息于水生植物间或底泥表面，滤食有机碎屑。",
    "网纹溞属（Ceriodaphnia）：头部常有网状花纹，第二触角发达，广泛分布于富营养湖泊，作为鱼类饵料。",
    "溞属（Daphnia）：枝角类典型代表，体侧扁，具壳刺，滤食藻类和细菌，在水生态系统中起关键调控作用。",
    "船卵溞属（Scapholeberis）：身体呈船形，头部窄，眼点明显，喜生活于浅水区，游泳缓慢。",
    "圆囊溞属（Holopedium）：体表包被透明胶质囊，营浮游生活，常见于贫营养湖泊，可避免被小型鱼捕食。",
    "薄皮溞属（Leptodora）：大型透明枝角类，为凶猛捕食者，以其他浮游动物为食，在温带湖泊常见。",
    "裸腹溞属（Moina）：腹部裸露，无壳瓣覆盖，耐污染，常见于富营养池塘，可作为环境指示生物。",
    "透明溞属（Diaphanosoma）：身体透明，第二触角长，滤食性，分布于湖泊和水库的敞水区。",
    "华哲水蚤属（Sinocalanus）：桡足类，哲水蚤科，第五胸足雌性对称，多见于沿海河口或淡水。",
    "新镖水蚤属（Neodiaptomus）：桡足类，镖水蚤科，雄性执握肢不对称，广泛分布于亚洲淡水水体。",
    "中镖水蚤属（Neutrodiaptomus）：桡足类，体型中等，第五胸足结构特殊，常见于水库和湖泊。",
    "华哲水蚤属（Sinodiaptomus）：桡足类，哲水蚤科，雄性左执握肢，常在池塘和浅水湖泊中发现。"
]

# ================== 加载模型 ==================
model = None
try:
    model = YOLO(MODEL_PATH)
    app.on_startup(lambda: ui.notify("模型加载成功", type="positive"))
except Exception as e:
    ui.notify(f"模型加载失败: {e}", type="negative")

# ================== 全局UI元素引用 ==================
preview = None
classify_btn = None
result_card = None
cn_name = None
en_name = None
confidence = None
progress = None
intro_text = None
status = None

uploaded_image = None
uploaded_bytes = None

# ================== 业务逻辑函数 ==================
def handle_upload(event):
    global uploaded_image, uploaded_bytes
    uploaded_bytes = event.content.read()
    if not event.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        ui.notify("仅支持 JPG/PNG 图片", type="warning")
        return

    try:
        uploaded_image = Image.open(io.BytesIO(uploaded_bytes)).convert('RGB')
    except Exception:
        ui.notify("图片格式无效", type="negative")
        return

    b64 = base64.b64encode(uploaded_bytes).decode()
    preview.set_source(f'data:image/png;base64,{b64}')
    preview.classes(remove='hidden')

    classify_btn.classes(remove='hidden')
    status.set_text('图片已上传，点击“开始分类”')

def classify():
    if model is None:
        ui.notify("模型未加载，无法分类", type="negative")
        return
    if uploaded_image is None:
        ui.notify("请先上传图片", type="warning")
        return

    status.set_text("分类中...")
    classify_btn.disable()
    ui.notify("正在推理，请稍候", type="info", timeout=2000)

    try:
        results = model(uploaded_image)
        probs = results[0].probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)

        en = CLASS_NAMES_EN[top1_idx]
        cn = CLASS_NAMES_CN[top1_idx]
        intro = INTROS[top1_idx]

        cn_name.set_text(cn)
        en_name.set_text(en)
        confidence.set_text(f"{top1_conf*100:.2f}%")
        progress.set_value(top1_conf)
        intro_text.set_text(intro)

        result_card.set_visibility(True)
        status.set_text("")
    except Exception as e:
        ui.notify(f"分类出错: {e}", type="negative")
        status.set_text("分类失败，请重试")
    finally:
        classify_btn.enable()

# ================== UI 布局 ==================
ui.query('body').style('background-color: #f5f7fa; font-family: "Inter", sans-serif;')

with ui.header(elevated=True).classes('items-center justify-between px-4'):
    ui.label('🌊 浮游生物智能分类').classes('text-h5 font-weight-bold')
    #ui.space()
    ui.label('安腾杯 · 基于YOLO26').classes('text-subtitle2 text-grey-7')

with ui.column().classes('items-center w-full max-w-3xl mx-auto p-6'):
    ui.label('上传浮游生物图片').classes('text-h6 text-grey-8 q-mb-md')

    uploader = ui.upload(
        label='点击或拖拽图片',
        on_upload=handle_upload,
        max_file_size=10_000_000,
        max_files=1,
    ).classes('w-full q-mb-lg').props('dense')

    preview = ui.image().classes('max-h-64 max-w-full rounded-lg shadow-md hidden')
    classify_btn = ui.button('开始分类', icon='search', on_click=classify).props('unelevated color=primary').classes('q-mt-md hidden')

    with ui.card().classes('w-full h-half q-mt-xl shadow-2') as result_card:
        result_card.set_visibility(False)
        with ui.row().classes('items-center w-full'):
            ui.icon('help', size='md').classes('text-primary q-mr-sm')
            ui.label('分类结果').classes('text-h6')

        ui.separator().classes('q-my-md')

        with ui.column().classes('w-full'):
            with ui.row().classes('items-baseline'):
                ui.label('中文名：').classes('text-subtitle2 text-grey-7')
                cn_name = ui.label().classes('text-h6 text-primary q-ml-sm')
                ui.label('英文名：').classes('text-subtitle2 text-grey-7 q-ml-md')
                en_name = ui.label().classes('text-h6 text-primary q-ml-sm')

            with ui.row().classes('items-center q-mt-md'):
                ui.label('置信度：').classes('text-subtitle2 text-grey-7')
                confidence = ui.label().classes('text-h6 text-secondary')
                #ui.space()
                progress = ui.linear_progress(value=0, size='sm').classes('w-1/2')

            ui.label('简介').classes('text-subtitle2 text-grey-7 q-mt-md')
            intro_text = ui.label().classes('text-body2')

    status = ui.label().classes('text-grey-6 q-mt-sm')

# ================== 启动 ==================
ui.run(title='浮游生物分类器', favicon='🧪')
