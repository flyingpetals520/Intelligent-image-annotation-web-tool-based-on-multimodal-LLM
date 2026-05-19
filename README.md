<div align="center">
<!-- 项目首图/Logo -->
<img src="sakuna.png" alt="Image Annotation Tool Banner" width="250">
</div>

# 智能多模态图像标注工具

<p>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/Flask-2.0%2B-000000?logo=flask&logoColor=white" alt="Flask">
  </a>
  <a href="https://huggingface.co/Qwen">
    <img src="https://img.shields.io/badge/Qwen3.5-4B%2F27B%2F35B--A3B-yellow?logo=huggingface&logoColor=white" alt="Qwen">
  </a>
  <br>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" alt="Status">
</p>

这是一个面向文生图模型训练合成数据、体育、人机交互等（最初是为了搓二次元动漫/插画数据集，老二刺猿了）的多标签智能图像分类标注系统。开发的动机是获得分层、超长的图像细粒度文本描述的人工成本较高，耗时较多，而利用成熟的多模态模型可以完成自动化高质量的图像描述来完成标注，进而为下游任务提供高质量的文本描述数据集，为姿态识别、文生图任务的模型训练等提供支持。支持 **本地 VLM 模型自动标注** + **远程 API 自动标注** + **人工校正** 三种模式。后续还会尝试努力打通和 ckn lab 的 ChenkinNoob-XL-V0.5 模型的生态，让数据集到生图全流程直接端到端贯通。（还要非常感谢智谱团队的 GLM 5.1 模型出色的 agent 能力，项目大部分编码由其完成，vibe coding 开源心目第一强）。此外还有自建的十万+张高质量二次元图片数据集，关注并联系我的 X 账号 https://x.com/flyingpetal472

## 功能特性

- 10 大分类、80+ 预设标签（性别、发色、发型、瞳色、角色特征、服装、姿势、场景、风格、人物数量等等）
- 支持本地部署多模态 VLM 模型自动标注（部署 Qwen3.5-4B 等，完全离线），预设调教 Prompt（也可以自己修改）
- 支持远程 API 自动标注（OpenAI / Anthropic）
- 支持部署 DWpose 模型进行图像人物姿态识别并保存骨骼元数据。
- Web 可视化标注界面，轻量易用，支持键盘快捷键
- 图片状态管理：未标注 -> 自动标注(黄) -> 已验证(绿)
- 导出 JSON / CSV 格式

## 文件说明

```
Project_path
├── annotate.py          # 主程序：Flask 标注服务器
├── local_vlm.py         # 本地 VLM 推理模块（Qwen3.5）
├── label_config.json    # 标签分类配置（可自定义）
├── pose_estimator.py    # 姿态估计模块（Dwpose）
├── annotations.json     # 标注数据（运行时生成）
├── templates/
│   └── index.html       # Web 标注界面
└── (你的图片文件...)
```
role_name.json 文件中已经整理了包含原神、崩坏星穹铁道、碧蓝档案等多个游戏的角色名称列表，如有需要可自行编辑和添加。
label_config.json 文件中整理了常见人物场景等的 10 大分类、80+ 预设标签，如有需要可自行编辑和添加。

## 安装依赖

### 1. 基础依赖（手动标注模式）

```bash
pip install flask pillow
```

### 2. 本地模型依赖（使用 Qwen3.5 自动标注）

```bash
# 安装 PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# transformers 5.x 已支持 Qwen3.5，确认版本
pip install transformers>=5.0.0
```

### 3. 远程 API 依赖（可选）

无需额外安装，使用 Python 内置 urllib。

## 使用方法

### 模式一：本地模型自动标注（推荐）

个人推荐使用 Qwen3.5-27B 和 Qwen3.5-35B-A3B。以我的 Qwen3.5-4B 为例，使用本地部署的 Qwen3.5-4B 多模态模型，完全离线运行：

```bash
python annotate.py --local-model Your_model_path
```

启动后：
1. 模型会自动加载到 GPU（约 8GB 显存，首次加载需 1-2 分钟）
2. 浏览器打开 `http://localhost:5000`
3. 点击 **「自动标注」** 标注当前图片，或 **「批量自动标注」** 批量处理
4. 自动标注的结果以黄色标签显示，点击可修改
5. 确认无误后点击 **「确认无误」**

可选参数：
```bash
# 指定精度（默认 bfloat16，可改 float16 节省显存）
python annotate.py --local-model Your_model_path --dtype float16

# 指定端口
python annotate.py --local-model Your_model_path --port 8080
```

### 模式二：使用命令行批量标注（无 Web 界面）

直接用本地模型批量标注所有图片：

```bash
python local_vlm.py --model Your_model_path --image-dir . --batch-size 50
```

参数说明：
- `--model`：模型路径（默认 `your_model_path`）
- `--image-dir`：图片目录（默认当前目录）
- `--output`：标注输出文件（默认 `annotations.json`）
- `--batch-size`：标注数量限制（不填则处理全部）
- `--dtype`：模型精度（默认 `bfloat16`）

### 模式三：远程 API 自动标注

#### OpenAI 兼容 API（GPT-4o、DeepSeek、通义千问等）

```bash
python annotate.py --api-key YOUR_KEY --api-type openai --base-url https://api.xxx.com
```

#### Anthropic Claude API

```bash
python annotate.py --api-key YOUR_KEY --api-type anthropic
```

### 模式四：仅手动标注

```bash
python annotate.py
```

## Web 界面操作

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| 左右箭头 | 切换上/下一张图片 |
| Ctrl+S | 保存当前标注 |
| Ctrl+Enter | 保存并跳转下一张 |
| Ctrl+Shift+Enter | 确认无误并下一张 |
| Ctrl+1 | 自动标注当前图片 |
| Q | 显示/隐藏快捷键 |

### 标注流程

```
未标注图片 -> [自动标注/手动选标签] -> 保存 -> 逐张确认 -> 导出
```

1. 左侧图片列表可按状态筛选（全部/未标/自动/已验）
2. 中间预览区查看大图
3. 右侧标签面板点选标签
4. 支持添加自定义标签

## 自定义标签体系

编辑 `label_config.json` 可自定义标签分类：

```json
{
  "性别": {
    "labels": ["女性", "男性"],
    "multi": false
  },
  "发色": {
    "labels": ["白色", "黑色", "金色", "橙色", "粉色", "蓝色"],
    "multi": true
  }
}
```

- `multi: false` 为单选分类
- `multi: true` 为多选分类
- 修改后刷新网页即可生效

## 导出标注数据

在 Web 界面点击 **「导出 JSON」** 或 **「导出 CSV」**，也可以直接访问：

```bash
# JSON 格式
curl http://localhost:5000/api/export?format=json -o annotations.json

# CSV 格式
curl http://localhost:5000/api/export?format=csv -o annotations.csv
```

### JSON 格式示例

```json
{
  "1743957338845.jpeg": {
    "labels": {
      "性别": "女性",
      "发色": ["橙色"],
      "发型": ["长发", "波浪"],
      "瞳色": ["紫色"],
      "角色特征": [],
      "服装": ["运动装"],
      "姿势": ["站立"],
      "背景": ["白色背景"],
      "画面风格": ["动漫"],
      "人物数量": "单人",
      ...
    },
    "custom_tags": [],
    "auto_labeled": true,
    "verified": false
    ...
  }
}
```

## 硬件要求

| 模式 | 最低要求 |
|------|----------|
| 仅手动标注 | 任意电脑，浏览器即可 |
| 本地模型 (推荐最低 Qwen3.5-4B，不然多模态标注能力太弱了) | NVIDIA GPU >= 12GB 显存（推荐 RTX 3070+） |
| 远程 API | 需要网络连接和 API Key |

## 故障排除

### 模型加载失败

```
[ERROR] 模型加载失败: No module named 'torch'
```

需安装 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### CUDA out of memory

```
[ERROR] CUDA out of memory
```

尝试降低精度：`--dtype float16`，或关闭其他占用显存的程序。

### 图片无法显示

确保图片文件名不含特殊字符，支持的格式：JPG、JPEG、PNG、BMP、GIF、WebP。
