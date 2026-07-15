# YOLO-Q26 Auto Aiming

基于地平线 BPU（Brain Processing Unit）的实时 YOLO 目标检测与自动瞄准系统。使用海康工业相机采集图像，在地平线 RDK 系列开发板上运行量化模型推理，通过串口发送云台控制指令实现自动瞄准。

## 系统架构

```
┌─────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────┐
│  HIK Camera │ ──▶ │ Capture Thread │ ──▶ │ Infer Thread │ ──▶ │ 主线程   │
│  (USB3.0)   │     │ (HIK_TEF_Driver)│    │ (BPU DNN)    │     │          │
└─────────────┘     └────────────────┘     └──────────────┘     └────┬─────┘
                                                                     │
                                              ┌──────────────────────┼─────┐
                                              ▼                      ▼
                                         ┌──────────┐          ┌──────────┐
                                         │ 串口 (UART)│          │ UDP Debug │
                                         │ 云台控制   │          │ 数据回传   │
                                         └──────────┘          └──────────┘
```

多线程流水线架构：
1. **采集线程** — 海康相机取流（BGR 图像）
2. **推理线程** — BPU 上运行 YOLO 量化模型推理（NV12 输入）
3. **主线程** — 后处理 → Alpha-Beta 滤波器跟踪 → 运动预测 → 串口控制指令编码 → 发送

## 硬件平台

- **开发板**：地平线 RDK 系列（RDK X3 / X5 等），支持 `hobot_dnn` BPU 推理
- **相机**：海康工业相机（USB3.0，通过 HIK_TEF_Driver 驱动）
- **云台**：串口通信协议（UART），波特率 115200
- **目标**：RoboMaster 竞赛装甲板（4 个灯条角点，4 种颜色 × 2 种尺寸 × 8 种类别）

## 目录结构

```
yoloq26-aiming/
├── multithreading2.py        # ★ 主程序：多线程实时检测 + 跟踪 + 瞄准
├── infer_bin.py              # BPU .bin 模型推理（串行版，含串口控制）
├── infer_onnx.py             # ONNX 模型推理（PC 端验证，CPU/GPU）
├── accuracy_bin.py           # 批量图片推理精度测试
├── get_img.py                # 相机图像批量采集工具
├── sin_test.py               # 串口正弦波测试（云台调试用）
├── sort.py                   # Alpha-Beta 滤波器（目标跟踪）
├── bpu_yolov5_tools.py       # BPU 模型工具（验证/可视化/格式转换）
├── HIK_TEF_Driver_module/    # 海康相机 C++/Pybind11 驱动
│   ├── HIK_TEF_Driver.cpp    #   驱动源码
│   ├── HIK_TEF_Driver.hpp    #   驱动头文件
│   ├── setup.py              #   编译脚本
│   ├── include/              #   海康 SDK 头文件
│   │   ├── MvCameraControl.h
│   │   ├── CameraParams.h
│   │   └── ...
│   └── lib/                  #   海康 SDK 动态库 (.so)
├── models/                   # 模型文件
│   ├── auto_aim_modified.onnx # 原始 ONNX 模型
│   └── auto_aim_modified*.bin # 多版本 BPU 量化模型
├── resized_img/              # 预处理后图像
├── video_img/                # 原始采集图像
└── video_result/             # 推理结果可视化
```

## 环境要求

- **运行平台**：地平线 RDK 开发板（ARM aarch64 Linux）
- **Python**：3.10
- **BPU SDK**：`hobot_dnn`（地平线 BPU 推理库）
- **驱动依赖**：海康工业相机 SDK（MvCameraControl）
- **外设**：USB 摄像头 + USB 串口（CH340 等）

```bash
# 编译海康相机驱动
cd HIK_TEF_Driver_module
python setup.py build_ext --inplace

# Python 依赖
pip install opencv-python numpy pyserial
```

## 使用方法

### 1. 相机图像采集

```bash
python get_img.py --count 100 --out video_img --prefix frame
```

参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--count` | 采集帧数 | 100 |
| `--out` | 保存目录 | `video_img` |
| `--prefix` | 文件名前缀 | `frame` |
| `--ext` | 格式 (jpg/png) | `jpg` |
| `--quality` | JPEG 质量 | 95 |

### 2. 批量推理验证（精度测试）

```bash
python accuracy_bin.py
```

读取 `video_img/` 下图片，使用 `.bin` 模型推理，结果保存至 `video_result/`。

### 3. ONNX 模型推理（PC 端）

```bash
python infer_onnx.py
```

在 PC 端使用 ONNX Runtime 验证模型，不依赖地平线硬件。

### 4. 实时自动瞄准（主程序）

```bash
python multithreading2.py
```

启动多线程实时检测与自动瞄准。程序通过串口 `/dev/ttyUSB0` 发送云台控制指令，通过 UDP (`192.168.127.1:9870`) 回传调试数据。

### 5. 串口调试

```bash
python sin_test.py
```

通过串口发送正弦波控制指令，用于测试云台响应。

## 模型说明

模型基于 YOLO 架构改进，在 640×640 输入下输出三层特征图（stride 8/16/32），每层包含：

| 输出头 | 通道数 | 含义 |
|--------|--------|------|
| box | 4 | bbox 偏移量 (l, t, r, b) |
| cls | 64 | 分类得分 (4 色 × 2 尺寸 × 8 类别 = 64) |
| kpt | 8 | 4 个关键点坐标 (x, y) × 4 |

**8 个目标类别**：`G`（哨兵）、`1`–`5`（1–5 号）、`O`（前哨站）、`B`（基地）  
**4 种颜色**：`blue`、`red`、`none`、`purple`  
**2 种尺寸**：`s`、`b`

## 串口协议

控制指令帧格式（7 字节）：

| 字节 | 含义 |
|------|------|
| byte[0] | 帧头 `0x50` |
| byte[1] | X 轴位置高 8 位 |
| byte[2] | X 轴位置低 8 位 |
| byte[3] | Y 轴位置高 8 位 |
| byte[4] | Y 轴位置低 8 位 |
| byte[5] | 射击标志：`0x0F` = 开火，`0xF0` = 待命 |
| byte[6] | 校验和（前 6 字节求和，低 8 位） |

位置映射：`byte[1:2] = int(pos / 20.0 * 0xFFFF)`，范围 0–20 对应 0x0000–0xFFFF。

## 跟踪与瞄准算法

1. **目标选择**：综合考虑面积分、中心距离分、历史粘性分，选择最优检测目标
2. **Alpha-Beta 滤波**：对目标中心 (x, y) 进行位置和速度估计，平滑抖动
3. **丢帧预测**：丢失目标时（≤ 3 帧），用上一时刻速度外推位置，速度逐帧衰减
4. **前馈补偿**：根据端到端延迟（图像采集 → 推理 → 控制），预测目标未来位置
5. **死区控制**：目标距画面中心 < 5% 时触发开火

## 调试数据（UDP）

通过 UDP 发送 JSON 格式调试数据至 `192.168.127.1:9870`：

```json
{
  "raw_x1": 0.52,       // 滤波后 X 坐标（归一化 0–1）
  "raw_y1": 0.48,       // 滤波后 Y 坐标
  "raw_x_cmd": 0.53,    // 前馈补偿后 X 指令
  "raw_y_cmd": 0.50,    // 前馈补偿后 Y 指令
  "vx": 0.012,          // X 方向速度
  "vy": -0.008          // Y 方向速度
}
```

## 核心配置

在 `multithreading2.py` 中可调整：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `TRACKER_MAX_AGE` | 最大丢帧数 | 3 |
| `MAX_VX / MAX_VY` | 速度限幅 | 0.3 |
| `W_AREA` | 面积权重 | 2 |
| `W_CENTER` | 中心距离权重 | 0.5 |
| `W_HISTORY` | 历史粘性权重 | 0.3 |
| `conf_threshold` | 检测置信度阈值 | 0.45 |
| `nms_threshold` | NMS 阈值 | 0.45 |
