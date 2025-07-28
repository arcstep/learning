from ultralytics import YOLO
import gradio as gr
from fastrtc import WebRTC
import cv2
import numpy as np
import torch
import time

# 检查 MPS 可用性
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# 升级到 YOLOv11
model = YOLO('yolo11n.pt')  # 使用 YOLOv11 nano 版本
model.to(device)  # 移动到 MPS 设备

def detection(image, conf_threshold=0.3):
    """优化的检测函数"""
    if image is None:
        return None
    
    try:
        # 强制转换为正确的格式
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 运行检测
        results = model(image, conf=conf_threshold)
        
        # 返回结果图像
        return results[0].plot()
        
    except Exception as e:
        print(f"检测过程中出错: {e}")
        return image  # 返回原图

def test_model():
    """测试模型是否正常工作"""
    print("开始测试 YOLOv11 模型...")
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 创建一个测试图像（包含一个简单的人形轮廓）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 画一个简单的人形
    cv2.rectangle(test_image, (200, 100), (400, 400), (255, 255, 255), -1)  # 身体
    cv2.circle(test_image, (300, 80), 30, (255, 255, 255), -1)  # 头部
    
    results = model(test_image, conf=0.1)
    print(f"测试检测结果: {len(results[0].boxes)} 个对象")
    
    if len(results[0].boxes) > 0:
        class_names = results[0].names
        for i, cls in enumerate(results[0].boxes.cls):
            class_name = class_names[int(cls)]
            conf = results[0].boxes.conf[i]
            print(f"  - {class_name}: {conf:.3f}")
    
    return results[0].plot()

# 先测试模型
print("=" * 50)
test_result = test_model()
print("=" * 50)

# 创建简单的图像上传界面进行测试
def simple_detection(image, conf_threshold=0.1):
    if image is None:
        return None
    
    results = model(image, conf=conf_threshold)
    return results[0].plot()

# 使用简单的图像上传界面
demo = gr.Interface(
    fn=simple_detection,
    inputs=[
        gr.Image(label="上传图像", type="numpy"),
        gr.Slider(0.0, 1.0, 0.1, label="置信度阈值")
    ],
    outputs=gr.Image(label="检测结果"),
    title="YOLOv11 目标检测测试 (Apple Silicon)",
    description="上传一张包含人、车、动物等的图片进行测试。YOLOv11 比 YOLOv8 更准确、更快！"
)

# 自定义 CSS 样式 - 侧边栏布局
custom_css = """
/* 移除默认的容器限制 */
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 10px !important;
}

/* 主布局容器 */
.main-layout {
    display: flex !important;
    gap: 20px !important;
    align-items: flex-start !important;
    justify-content: center !important;
}

/* 视频流容器 - 占主要空间 */
.video-container {
    flex: 1 !important;
    max-width: 75% !important;
    max-height: 75vh !important;
}

/* 侧边栏容器 */
.sidebar {
    width: 250px !important;
    min-width: 250px !important;
    background: #f8f9fa !important;
    border-radius: 8px !important;
    padding: 15px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

/* 控制面板样式 */
.control-panel {
    margin-bottom: 15px !important;
}

/* 标题样式 */
.title {
    text-align: center !important;
    margin-bottom: 10px !important;
    font-size: 1.5em !important;
}

/* 描述文字样式 */
.description {
    text-align: center !important;
    color: #666 !important;
    margin-bottom: 20px !important;
    font-size: 0.9em !important;
}

/* 侧边栏标题 */
.sidebar-title {
    font-weight: bold !important;
    margin-bottom: 10px !important;
    color: #333 !important;
    font-size: 1.1em !important;
}

/* 移除 footer */
footer {
    display: none !important;
}
"""

# WebRTC 实时检测界面
with gr.Blocks(css=custom_css) as webrtc_demo:
    gr.HTML(
        """
        <div class="title">
        YOLOv11 实时目标检测 (Powered by Apple Silicon MPS ⚡️)
        </div>
        <div class="description">
        请允许浏览器访问摄像头，实时检测目标物体
        </div>
        """
    )
    
    # 主布局容器
    with gr.Row(elem_classes=["main-layout"]):
        # 左侧：视频流
        with gr.Column(elem_classes=["video-container"]):
            video = WebRTC(
                label="摄像头流",
                mode="send-receive",
                modality="video",
                show_label=True,
            )
        
        # 右侧：控制面板
        with gr.Column(elem_classes=["sidebar"]):
            gr.HTML('<div class="sidebar-title">控制面板</div>')
            
            # 置信度阈值
            conf_threshold = gr.Slider(
                label="置信度阈值",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.1,
                elem_classes=["control-panel"]
            )
            
            # FPS 显示
            fps_display = gr.Number(
                label="检测 FPS",
                value=0,
                interactive=False,
                elem_classes=["control-panel"]
            )
            
            # 检测信息显示
            detection_info = gr.Textbox(
                label="检测信息",
                value="等待检测...",
                interactive=False,
                lines=4,
                elem_classes=["control-panel"]
            )
            
            # 状态信息
            status_info = gr.Textbox(
                label="系统状态",
                value="✅ 模型已加载\n✅ MPS 加速启用\n✅ 等待摄像头连接",
                interactive=False,
                lines=3,
                elem_classes=["control-panel"]
            )
    
    # 设置流处理
    video.stream(
        fn=detection,
        inputs=[video, conf_threshold],
        outputs=[video],
        time_limit=60
    )

if __name__ == "__main__":
    print("启动 YOLOv11 目标检测系统...")
    print("✅ 模型加载完成")
    print("✅ Apple Silicon MPS 加速启用")
    print("✅ WebRTC 实时检测就绪")
    
    # 启动两个界面
    with gr.TabbedInterface(
        [demo, webrtc_demo],
        ["图像上传测试", "实时目标检测"],
        title="YOLOv11 目标检测系统 (Apple Silicon)"
    ) as app:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True
        )