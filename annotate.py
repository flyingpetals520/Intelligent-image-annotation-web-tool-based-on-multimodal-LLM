"""
图像多标签标注工具 - 自动标注 + 人工校正

支持三种自动标注模式:
  1. 本地模型: python annotate.py --local-model F:/qwen3_5
  2. OpenAI API: python annotate.py --api-key YOUR_KEY --api-type openai
  3. Claude API: python annotate.py --api-key YOUR_KEY --api-type anthropic
  4. 仅手动:    python annotate.py
"""

import os
import sys
import json
import base64
import argparse
import threading
import time
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_from_directory
from PIL import Image

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR
ANNOTATIONS_FILE = BASE_DIR / "annotations.json"
LABEL_CONFIG_FILE = BASE_DIR / "label_config.json"
LAST_POSITION_FILE = BASE_DIR / "last_position.json"
MODEL_CONFIG_FILE = BASE_DIR / "model_config.json"
BATCH_HISTORY_FILE = BASE_DIR / "batch_history.json"
API_PROFILES_FILE = BASE_DIR / "api_profiles.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
POSE_SKELETONS_DIR = BASE_DIR / "pose_skeletons"

# ============================================================
# 标签配置 (预设分类)
# ============================================================
DEFAULT_LABEL_CONFIG = {
    "性别": {
        "labels": ["女性", "男性"],
        "multi": False
    },
    "发色": {
        "labels": ["白色", "黑色", "金色", "橙色", "粉色", "蓝色", "灰色", "棕色", "红色", "绿色", "紫色", "银色"],
        "multi": True
    },
    "发型": {
        "labels": ["长发", "短发", "马尾", "双马尾", "波波头", "卷发", "直发", "波浪", "丸子头", "散发"],
        "multi": True
    },
    "瞳色": {
        "labels": ["蓝色", "绿色", "红色", "棕色", "紫色", "黄色", "灰色", "黑色", "异色瞳"],
        "multi": True
    },
    "角色特征": {
        "labels": ["猫耳", "兽耳", "角", "翅膀", "尾巴", "眼镜", "帽子", "蝴蝶结", "发带", "光环", "精灵耳", "呆毛"],
        "multi": True
    },
    "服装": {
        "labels": ["女仆装", "校服", "泳装", "比基尼", "哥特", "铠甲", "休闲", "运动装", "裙子", "和服", "制服", "连衣裙", "圣诞装", "旗袍"],
        "multi": True
    },
    "姿势": {
        "labels": ["站立", "坐姿", "动态", "躺姿", "行走", "跑步", "蹲姿"],
        "multi": True
    },
    "背景": {
        "labels": ["白色背景", "深色背景", "室外", "室内", "简单背景", "复杂背景", "透明背景"],
        "multi": True
    },
    "画面风格": {
        "labels": ["动漫", "Q版", "写实", "水彩", "素描"],
        "multi": True
    },
    "人物数量": {
        "labels": ["单人", "双人", "多人"],
        "multi": False
    }
}


def load_label_config():
    """加载标签配置，自动补全缺失的 enabled 字段"""
    if LABEL_CONFIG_FILE.exists():
        with open(LABEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = DEFAULT_LABEL_CONFIG
    # 补全 enabled 字段: 旧配置没有此字段，默认全部启用
    for cat, info in config.items():
        if "enabled" not in info:
            info["enabled"] = True
    save_label_config(config)
    return config


def save_label_config(config):
    with open(LABEL_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_annotations():
    """加载标注数据"""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(data):
    """保存标注数据"""
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 标注文件读写锁: 防止并发请求的 load-modify-write 竞态导致字段丢失
annotation_lock = threading.Lock()


def atomic_update_annotation(filename, updater):
    """原子更新标注: 加锁 -> 重新加载 -> 复制 existing -> 应用 updater 返回的更新 -> 保存。
    updater(existing_copy) -> dict of updates to apply。返回合并后的完整标注。"""
    with annotation_lock:
        annotations = load_annotations()
        existing = dict(annotations.get(filename, {}))
        updates = updater(existing)
        merged = dict(existing)
        merged.update(updates)
        annotations[filename] = merged
        save_annotations(annotations)
        return merged


def load_last_position():
    """加载上次查看位置"""
    if LAST_POSITION_FILE.exists():
        with open(LAST_POSITION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_last_position(filename):
    """保存当前查看位置"""
    data = {
        "filename": filename,
        "timestamp": datetime.now().isoformat()
    }
    with open(LAST_POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_model_config():
    """加载已保存的模型配置 (api_type/api_key/base_url/model/model_path/model_dtype)"""
    if not MODEL_CONFIG_FILE.exists():
        return None
    try:
        with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_api_profiles():
    """加载 API 配置 profiles"""
    if API_PROFILES_FILE.exists():
        try:
            with open(API_PROFILES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "profiles" not in data:
                    data["profiles"] = []
                if "active_profile_id" not in data:
                    data["active_profile_id"] = None
                return data
        except Exception:
            pass
    return {"profiles": [], "active_profile_id": None}


def save_api_profiles(data):
    """保存 API 配置 profiles"""
    try:
        with open(API_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存 API profiles 失败: {e}")


def mask_api_key(key):
    """脱敏 API Key: 前4后4，中间用 *** 替代"""
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "***" + key[-2:] if len(key) >= 4 else "***"
    return key[:4] + "***" + key[-4:]


def sync_active_profile_to_config():
    """将 active profile 的配置同步到 app_config"""
    data = load_api_profiles()
    active_id = data.get("active_profile_id")
    if not active_id:
        return
    profiles = data.get("profiles", [])
    for p in profiles:
        if p.get("id") == active_id:
            app_config["api_type"] = p.get("api_type", "openai")
            app_config["api_key"] = p.get("api_key", "")
            app_config["base_url"] = p.get("base_url", "") or None
            app_config["model"] = p.get("model", "") or None
            print(f"[INFO] 已激活 API Profile: {p.get('name')} ({p.get('api_type')})")
            return


def migrate_old_to_profiles():
    """首次迁移: 将 model_config.json 中的 API 配置导入为默认 profile"""
    if API_PROFILES_FILE.exists():
        return False
    saved = load_model_config()
    if not saved:
        return False
    api_type = saved.get("api_type", "openai")
    api_key = saved.get("api_key", "")
    if not api_key:
        return False
    import uuid
    profile = {
        "id": uuid.uuid4().hex[:12],
        "name": "默认配置",
        "api_type": api_type,
        "api_key": api_key,
        "base_url": saved.get("base_url", "") or "",
        "model": saved.get("model", "") or "",
    }
    data = {"profiles": [profile], "active_profile_id": profile["id"]}
    save_api_profiles(data)
    print(f"[INFO] 已迁移旧 API 配置到 profile: {profile['name']}")
    return True


def save_model_config():
    """保存当前模型配置到文件（仅本地模型部分），下次启动自动恢复"""
    data = {
        "model_path": app_config.get("model_path"),
        "model_dtype": app_config.get("model_dtype"),
    }
    try:
        with open(MODEL_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存模型配置失败: {e}")


def get_image_list():
    """获取所有图片文件名"""
    files = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(f)
    return files


def _build_openai_url(base_url):
    """构建 OpenAI chat completions URL，兼容各种 base_url 输入格式。
    处理: 尾部斜杠、已含 /v1、已含 /chat/completions 等情况。"""
    base = (base_url or "https://api.openai.com").rstrip('/')
    if base.endswith('/chat/completions'):
        return base
    if base.endswith('/v1'):
        base = base[:-3]
    return base + "/v1/chat/completions"


def image_to_base64(filepath, max_size=512):
    """将图片转为 base64 (用于 API 调用)"""
    img = Image.open(filepath)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _crop_image(filepath, crop):
    """按 crop={x,y,w,h} (natural 坐标) 裁剪图片，返回 PIL Image (RGB)"""
    img = Image.open(filepath)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    x = max(0, int(crop.get("x", 0)))
    y = max(0, int(crop.get("y", 0)))
    w = max(1, int(crop.get("w", 0)))
    h = max(1, int(crop.get("h", 0)))
    x2 = min(img.width, x + w)
    y2 = min(img.height, y + h)
    if x2 <= x or y2 <= y:
        return img  # 无效裁剪，返回整图
    return img.crop((x, y, x2, y2))


def _crop_to_base64(filepath, crop, max_size=1024):
    """裁剪图片并转 base64 JPEG (用于远程 API)"""
    import io
    pil_img = _crop_image(filepath, crop)
    if max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# 自动标注 - 支持多种后端
# ============================================================

def auto_label_with_openai(filepath, api_key, model="gpt-4o", base_url=None):
    """使用 OpenAI 兼容 API 自动标注"""
    import urllib.request
    import urllib.error

    filename = os.path.basename(filepath)
    print(f"[OpenAI] 开始标注: {filename}")
    b64 = image_to_base64(filepath)
    config = load_label_config()

    # 只包含启用的分类
    enabled_config = {cat: info for cat, info in config.items() if info.get("enabled", True)}
    categories_desc = []
    for cat, info in enabled_config.items():
        mode = "单选" if not info["multi"] else "多选"
        labels = ", ".join(info["labels"])
        categories_desc.append(f"- {cat}({mode}): [{labels}]")

    example = {}
    for cat, info in enabled_config.items():
        if info.get("multi", False):
            example[cat] = ["标签1"]
        else:
            example[cat] = "标签"
    example_str = json.dumps(example, ensure_ascii=False, indent=2)

    prompt = f"""请分析这张图片，为以下每个分类选择最合适的标签。

分类列表:
{chr(10).join(categories_desc)}

请严格按照以下 JSON 格式输出，不要包含其他内容:
{example_str}

注意:
- 单选分类用字符串, 多选分类用数组
- 只能从上面列出的标签中选择
- 每个标签必须选择一个类别符合图中情况的，不许跳过返回空
- 必须是纯JSON, 不要markdown代码块"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }

    url = _build_openai_url(base_url)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(f"[OpenAI] 正在请求 API: {model}, URL: {url}")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        _batch_http_resp[0] = resp
        result = json.loads(resp.read().decode())
        resp.close()
        _batch_http_resp[0] = None
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500] if hasattr(e, 'read') else ''
        raise RuntimeError(f"OpenAI API HTTP {e.code} {e.reason} | URL: {url} | 响应: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI API 连接失败: {e.reason} | URL: {url}")

    text = result["choices"][0]["message"]["content"].strip()
    print(f"[OpenAI] 收到响应，长度: {len(text)} 字符")
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    labels = json.loads(text)
    print(f"[OpenAI] 标注完成: {filename}")#-> {list(labels.keys())}
    return labels


def auto_label_with_anthropic(filepath, api_key, model="claude-sonnet-4-20250514"):
    """使用 Anthropic Claude API 自动标注"""
    import urllib.request

    filename = os.path.basename(filepath)
    print(f"[Anthropic] 开始标注: {filename}")
    b64 = image_to_base64(filepath)
    config = load_label_config()

    # 只包含启用的分类
    enabled_config = {cat: info for cat, info in config.items() if info.get("enabled", True)}
    categories_desc = []
    for cat, info in enabled_config.items():
        mode = "单选" if not info["multi"] else "多选"
        labels = ", ".join(info["labels"])
        categories_desc.append(f"- {cat}({mode}): [{labels}]")

    example = {}
    for cat, info in enabled_config.items():
        if info.get("multi", False):
            example[cat] = ["标签1"]
        else:
            example[cat] = "标签"
    example_str = json.dumps(example, ensure_ascii=False, indent=2)

    prompt = f"""请分析这张图片，为以下每个分类选择最合适的标签。

分类列表:
{chr(10).join(categories_desc)}

请严格按照以下 JSON 格式输出，不要包含其他内容:
{example_str}

注意:
- 单选分类用字符串, 多选分类用数组
- 只能从上面列出的标签中选择
- 每个标签必须选择一个类别符合图中情况的，不许跳过返回空
- 必须是纯JSON, 不要markdown代码块"""

    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    print(f"[Anthropic] 正在请求 API: {model}")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers=headers
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        _batch_http_resp[0] = resp
        result = json.loads(resp.read().decode())
        resp.close()
        _batch_http_resp[0] = None
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500] if hasattr(e, 'read') else ''
        raise RuntimeError(f"Anthropic API HTTP {e.code} {e.reason} | 响应: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Anthropic API 连接失败: {e.reason}")

    text = result["content"][0]["text"].strip()
    print(f"[Anthropic] 收到响应，长度: {len(text)} 字符")
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    labels = json.loads(text)
    print(f"[Anthropic] 标注完成: {filename}") #-> {list(labels.keys())}
    return labels


def auto_label_with_local(filepath):
    """使用本地 VLM 模型自动标注"""
    filename = os.path.basename(filepath)
    print(f"[Local] 开始标注: {filename}")
    vlm = app_config.get("local_vlm")
    if vlm is None:
        raise RuntimeError("本地模型未加载")
    labels = vlm.label_image(str(filepath))
    print(f"[Local] 标注完成: {filename}")#-> {list(labels.keys())} 
    return labels


def auto_label_image(filepath, api_key=None, api_type="openai", model=None, base_url=None):
    """自动标注单张图片 (统一入口)"""
    if api_type == "local":
        return auto_label_with_local(filepath)
    elif api_type == "anthropic":
        return auto_label_with_anthropic(
            filepath, api_key, model or "claude-sonnet-4-20250514"
        )
    else:
        return auto_label_with_openai(
            filepath, api_key, model or "gpt-4o", base_url
        )


# ============================================================
# Flask 应用
# ============================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# 全局配置 (运行时设置)
app_config = {
    "api_key": None,
    "api_type": "openai",  # openai / anthropic / local
    "model": None,
    "base_url": None,
    "local_vlm": None,     # 本地 VLM 实例
    "pose_estimator": None, # 姿态估计模型实例
    "auto_labeling_progress": {"total": 0, "done": 0, "running": False},
    "batch_progress": {
        "total": 0, "done": 0, "running": False, "cancel": False,
        "current_file": "", "current_task": "",
        "success": 0, "failed": 0, "errors": [],
        "start_time": 0, "elapsed": 0
    },
    "model_path": None,           # 本地模型路径 (即使延迟加载也保存)
    "model_dtype": "bfloat16",    # 本地模型精度
    "model_state": {              # 模型加载状态
        "loaded": False,
        "loading": False,
        "progress": 0,
        "status": "idle",         # idle / loading / loaded / unloaded / error
        "error": None,
    },
}


@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route("/api/images")
def list_images():
    """获取图片列表及标注状态"""
    files = get_image_list()
    annotations = load_annotations()
    result = []
    for f in files:
        ann = annotations.get(f, None)
        result.append({
            "filename": f,
            "annotated": ann is not None,
            "auto_labeled": ann.get("auto_labeled", False) if ann else False,
            "verified": ann.get("verified", False) if ann else False,
            "labels": ann.get("labels", {}) if ann else {},
            "custom_tags": ann.get("custom_tags", []) if ann else []
        })
    return jsonify(result)


@app.route("/api/image/<path:filename>")
def serve_image(filename):
    """提供图片文件"""
    return send_from_directory(str(IMAGE_DIR), filename)


@app.route("/api/image-info/<path:filename>")
def image_info(filename):
    """获取图片文件元信息（大小、修改时间、尺寸等）"""
    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404
    stat = filepath.stat()
    info = {
        "filename": filename,
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }
    try:
        with Image.open(filepath) as img:
            info["dimensions"] = list(img.size)
            info["format"] = img.format or ""
    except Exception:
        pass
    return jsonify(info)


@app.route("/api/annotation/<path:filename>", methods=["GET"])
def get_annotation(filename):
    """获取单个标注"""
    annotations = load_annotations()
    return jsonify(annotations.get(filename, None))


@app.route("/api/annotation/<path:filename>", methods=["POST"])
def save_annotation(filename):
    """保存标注 (原子更新: 未提供的字段保留已有值, 加锁防竞态)"""
    data = request.json or {}
    def updater(existing):
        updates = {}
        for field in ["labels", "custom_tags", "description", "description_history",
                       "review", "review_history", "danbooru_tags", "auto_labeled", "verified"]:
            if field in data:
                updates[field] = data[field]
        updates["updated_at"] = datetime.now().isoformat()
        return updates
    atomic_update_annotation(filename, updater)
    return jsonify({"status": "ok"})


@app.route("/api/annotation/<path:filename>", methods=["DELETE"])
def delete_annotation(filename):
    """删除标注"""
    with annotation_lock:
        annotations = load_annotations()
        if filename in annotations:
            del annotations[filename]
            save_annotations(annotations)
    return jsonify({"status": "ok"})


# ============================================================
# 姿态估计
# ============================================================

@app.route("/api/pose-estimate/<path:filename>", methods=["POST"])
def pose_estimate(filename):
    """对单张图片进行姿态估计"""
    pose_est = app_config.get("pose_estimator")
    if pose_est is None:
        return jsonify({"error": "姿态模型未加载，请使用 --pose-model 参数启动"}), 400

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    try:
        result = pose_est.estimate(str(filepath))
        pose_image = result["pose_image"]

        # 保存骨骼图到 pose_skeletons 目录
        POSE_SKELETONS_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem
        pose_filename = f"{stem}_pose.png"
        pose_path = POSE_SKELETONS_DIR / pose_filename
        pose_image.save(str(pose_path))

        # base64 编码用于前端即时预览
        import io
        buf = io.BytesIO()
        pose_image.save(buf, format="PNG")
        pose_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 原子保存: 加锁防竞态，字段赋值保留其他字段
        def updater_pose(existing):
            return {
                "pose": {
                    "keypoints": result.get("keypoints", []),
                    "scores": result.get("scores", []),
                    "pose_image_path": pose_filename,
                    "updated_at": datetime.now().isoformat()
                },
                "updated_at": datetime.now().isoformat()
            }
        atomic_update_annotation(filename, updater_pose)

        return jsonify({
            "status": "ok",
            "pose_image_b64": pose_b64,
            "keypoints": result.get("keypoints", []),
            "scores": result.get("scores", []),
            "pose_image_path": pose_filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pose-image/<path:filename>")
def serve_pose_image(filename):
    """提供姿态骨骼图"""
    return send_from_directory(str(POSE_SKELETONS_DIR), filename)


@app.route("/api/pose-usability/<path:filename>", methods=["POST"])
def set_pose_usability(filename):
    """设置姿态可用性评级 (00=不可用, 01=尚可, 11=可用)"""
    data = request.json or {}
    usability = data.get("usability")
    if usability not in ("00", "01", "11"):
        return jsonify({"error": "无效的可用性值，需为 00/01/11"}), 400
    def updater(existing):
        pose = dict(existing.get("pose") or {})
        pose["usability"] = usability
        pose["updated_at"] = datetime.now().isoformat()
        return {"pose": pose}
    atomic_update_annotation(filename, updater)
    return jsonify({"status": "ok", "usability": usability})


@app.route("/api/pose/<path:filename>", methods=["DELETE"])
def delete_pose(filename):
    """删除姿态数据"""
    with annotation_lock:
        annotations = load_annotations()
        if filename in annotations and "pose" in annotations[filename]:
            pose_info = annotations[filename]["pose"]
            # 删除骨骼图文件
            if pose_info and pose_info.get("pose_image_path"):
                pose_file = POSE_SKELETONS_DIR / pose_info["pose_image_path"]
                if pose_file.exists():
                    pose_file.unlink()
            del annotations[filename]["pose"]
            save_annotations(annotations)
    return jsonify({"status": "ok"})


def _has_auto_label():
    """检查是否配置了自动标注能力"""
    return app_config.get("api_key") or app_config.get("api_type") == "local"


@app.route("/api/auto-label/<path:filename>", methods=["POST"])
def auto_label_single(filename):
    """自动标注单张图片"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    try:
        labels = auto_label_image(
            str(filepath),
            api_key=app_config.get("api_key"),
            api_type=app_config["api_type"],
            model=app_config.get("model"),
            base_url=app_config.get("base_url"),
        )
        # 原子保存: 加锁防竞态
        def updater_auto(existing):
            return {
                "labels": labels,
                "auto_labeled": True,
                "verified": False,
                "updated_at": datetime.now().isoformat()
            }
        atomic_update_annotation(filename, updater_auto)
        return jsonify({"status": "ok", "labels": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auto-label-batch", methods=["POST"])
def auto_label_batch():
    """批量自动标注"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    data = request.json or {}
    batch_size = data.get("batch_size", 20)
    overwrite = data.get("overwrite", False)

    if app_config["auto_labeling_progress"]["running"]:
        return jsonify({"error": "正在标注中，请等待完成"}), 409

    files = get_image_list()
    annotations = load_annotations()

    if not overwrite:
        files = [f for f in files if f not in annotations]

    files = files[:batch_size]
    total = len(files)

    if total == 0:
        return jsonify({"status": "ok", "message": "没有需要标注的图片"})

    def run_batch():
        app_config["auto_labeling_progress"] = {"total": total, "done": 0, "running": True}
        print(f"[Batch] 批量标注开始，共 {total} 张图片")
        for f in files:
            try:
                filepath = IMAGE_DIR / f
                labels = auto_label_image(
                    str(filepath),
                    api_key=app_config.get("api_key"),
                    api_type=app_config["api_type"],
                    model=app_config.get("model"),
                    base_url=app_config.get("base_url"),
                )
                # 每张图原子保存: 加锁防竞态，保留已有字段
                def updater_ok(existing, _labels=labels):
                    return {
                        "labels": _labels,
                        "auto_labeled": True,
                        "verified": False,
                        "updated_at": datetime.now().isoformat()
                    }
                atomic_update_annotation(f, updater_ok)
            except Exception as e:
                print(f"[Batch] 标注失败 {f}: {e}")
                err_msg = str(e)
                def updater_err(existing, _err=err_msg):
                    return {
                        "labels": {},
                        "auto_labeled": False,
                        "error": _err,
                        "updated_at": datetime.now().isoformat()
                    }
                atomic_update_annotation(f, updater_err)
            app_config["auto_labeling_progress"]["done"] += 1
            print(f"[Batch] 进度: {app_config['auto_labeling_progress']['done']}/{total}")

        app_config["auto_labeling_progress"]["running"] = False
        print(f"[Batch] 批量标注完成")

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()

    return jsonify({"status": "started", "total": total})


@app.route("/api/auto-label-progress")
def auto_label_progress():
    """获取批量标注进度"""
    return jsonify(app_config["auto_labeling_progress"])



# ============================================================
# 统一批量处理 (标签 + 描述 + 审核)
# ============================================================


def save_batch_history(batch_config, result):
    """保存批量处理历史到 batch_history.json"""
    history = []
    if BATCH_HISTORY_FILE.exists():
        try:
            with open(BATCH_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": batch_config.get("mode"),
        "model": batch_config.get("model"),
        "base_url": batch_config.get("base_url"),
        "tasks": batch_config.get("tasks"),
        "overwrite": batch_config.get("overwrite"),
        "enable_thinking": batch_config.get("enable_thinking"),
        "total": result.get("total", 0),
        "success": result.get("success", 0),
        "failed": result.get("failed", 0),
        "cancelled": result.get("cancelled", False),
        "duration_seconds": result.get("duration_seconds", 0),
        "duration_str": result.get("duration_str", "00:00:00"),
        "errors_count": len(result.get("errors", [])),
        "errors": result.get("errors", [])[:30],  # 存前 30 条错误详情 (含 file/task/error)
    }
    history.append(entry)
    history = history[-100:]
    try:
        with open(BATCH_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[Batch] 历史记录已保存到 {BATCH_HISTORY_FILE}")
    except Exception as e:
        print(f"[Batch] 保存历史失败: {e}")



_batch_http_resp = [None]  # 批量时存储当前 HTTP 响应，取消时主动关闭避免连接泄漏


def _run_with_cancel(func, cancel_fn, poll_interval=1):
    """daemon 子线程调 func，主线程每秒检查 cancel。
    cancel 时主动关闭 _batch_http_resp 中的 HTTP 连接，
    让 urlopen 立即报错退出，连接正确释放。"""
    result = [None]
    error = [None]
    done = [False]
    def worker():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e
        finally:
            _batch_http_resp[0] = None
            done[0] = True
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while not done[0]:
        if cancel_fn():
            # 主动关闭 HTTP 连接，让子线程 urlopen 立即报错
            if _batch_http_resp[0] is not None:
                try:
                    _batch_http_resp[0].close()
                except Exception:
                    pass
                _batch_http_resp[0] = None
            # 等子线程退出 (连接关闭后 urlopen 会立即报错)
            for _ in range(10):
                if done[0]:
                    break
                time.sleep(0.3)
            raise RuntimeError("用户已取消")
        time.sleep(poll_interval)
    if error[0]:
        raise error[0]
    return result[0]


def run_batch_process(files, tasks, overwrite, enable_thinking):
    """批量处理线程: 逐张图执行选中的任务 (标注/描述/审核)"""
    progress = app_config["batch_progress"]
    start_time = time.time()
    progress.update({
        "total": len(files), "done": 0, "running": True, "cancel": False,
        "current_file": "", "current_task": "",
        "success": 0, "failed": 0, "errors": [],
        "start_time": start_time, "elapsed": 0
    })
    # capture model info for history
    batch_config = {
        "mode": app_config.get("api_type"),
        "model": (os.path.basename(app_config.get("model_path", "")) if app_config.get("api_type") == "local" and app_config.get("model_path") else app_config.get("model")),
        "base_url": app_config.get("base_url"),
        "tasks": tasks,
        "overwrite": overwrite,
        "enable_thinking": enable_thinking,
    }
    print(f"[Batch] 统一批量处理开始，共 {len(files)} 张图片，任务: {tasks}")

    for f in files:
        if progress["cancel"]:
            break

        progress["current_file"] = f
        filepath = IMAGE_DIR / f
        if not filepath.exists():
            progress["failed"] += 1
            progress["errors"].append({"file": f, "task": "-", "error": "文件不存在"})
            progress["done"] += 1
            continue

        annotations = load_annotations()
        existing = annotations.get(f, {})

        # 任务 1: 标签标注
        if tasks.get("label") and not progress["cancel"]:
            progress["current_task"] = "标注"
            if overwrite or not existing.get("labels"):
                try:
                    labels = _run_with_cancel(
                        lambda: auto_label_image(
                            str(filepath),
                        api_key=app_config.get("api_key"),
                        api_type=app_config["api_type"],
                        model=app_config.get("model"),
                        base_url=app_config.get("base_url"),
                        ),
                        lambda: progress["cancel"]
                    )
                    def ul(ex, _l=labels):
                        return {"labels": _l, "auto_labeled": True, "verified": False,
                                "updated_at": datetime.now().isoformat()}
                    atomic_update_annotation(f, ul)
                    progress["success"] += 1
                except Exception as e:
                    if "取消" in str(e):
                        progress["cancel"] = True
                        break
                    progress["failed"] += 1
                    progress["errors"].append({"file": f, "task": "标注", "error": str(e)[:200]})

        # 任务 2: 半自由描述
        if tasks.get("desc") and not progress["cancel"]:
            progress["current_task"] = "描述"
            if overwrite or not existing.get("description"):
                try:
                    desc = _run_with_cancel(
                        lambda: _generate_desc_core(str(filepath), enable_thinking=enable_thinking),
                        lambda: progress["cancel"]
                    )
                    def ud(ex, _d=desc):
                        return {"description": _d,
                                "description_history": [{"role": "assistant", "content": _d}],
                                "updated_at": datetime.now().isoformat()}
                    atomic_update_annotation(f, ud)
                    progress["success"] += 1
                except Exception as e:
                    if "取消" in str(e):
                        progress["cancel"] = True
                        break
                    progress["failed"] += 1
                    progress["errors"].append({"file": f, "task": "描述", "error": str(e)[:200]})

        # 任务 3: R18+审核
        if tasks.get("review") and not progress["cancel"]:
            progress["current_task"] = "审核"
            if overwrite or not existing.get("review"):
                try:
                    fresh_labels = load_annotations().get(f, {}).get("labels", {})
                    review = _run_with_cancel(
                        lambda: _generate_review_core(str(filepath), fresh_labels, enable_thinking=enable_thinking),
                        lambda: progress["cancel"]
                    )
                    def ur(ex, _r=review):
                        return {"review": _r,
                                "review_history": [{"role": "assistant", "content": _r}],
                                "updated_at": datetime.now().isoformat()}
                    atomic_update_annotation(f, ur)
                    progress["success"] += 1
                except Exception as e:
                    if "取消" in str(e):
                        progress["cancel"] = True
                        break
                    progress["failed"] += 1
                    progress["errors"].append({"file": f, "task": "审核", "error": str(e)[:200]})

        progress["done"] += 1
        print(f"[Batch] {progress['done']}/{progress['total']} (成功 {progress['success']}, 失败 {progress['failed']})")

    elapsed = int(time.time() - start_time)
    progress["running"] = False
    progress["current_file"] = ""
    progress["current_task"] = ""
    progress["elapsed"] = elapsed
    cancelled = progress["cancel"]
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60
    duration_str = f"{h:02d}:{m:02d}:{s:02d}"
    print(f"[Batch] {'已取消' if cancelled else '完成'}: 成功 {progress['success']}, 失败 {progress['failed']}, 耗时 {duration_str}")
    # 保存历史
    save_batch_history(batch_config, {
        "total": progress["total"],
        "success": progress["success"],
        "failed": progress["failed"],
        "cancelled": cancelled,
        "duration_seconds": elapsed,
        "duration_str": duration_str,
        "errors": progress["errors"],
    })


@app.route("/api/batch-process", methods=["POST"])
def batch_process():
    """统一批量处理: 标签 + 描述 + 审核"""
    if not _has_auto_label():
        return jsonify({"error": "未配置模型，请先配置本地模型或 API"}), 400
    data = request.json or {}
    files = data.get("files", [])
    tasks = data.get("tasks", {})
    overwrite = data.get("overwrite", False)
    enable_thinking = data.get("enable_thinking", False)
    if not files:
        return jsonify({"error": "未选择图片"}), 400
    if not any(tasks.get(k) for k in ("label", "desc", "review")):
        return jsonify({"error": "未选择任务"}), 400
    if app_config["batch_progress"]["running"]:
        return jsonify({"error": "正在处理中，请等待完成或取消"}), 409
    thread = threading.Thread(target=run_batch_process, args=(files, tasks, overwrite, enable_thinking), daemon=True)
    thread.start()
    return jsonify({"status": "started", "total": len(files)})


@app.route("/api/batch-progress")
def batch_progress_api():
    progress = dict(app_config["batch_progress"])
    if progress["running"] and progress.get("start_time"):
        progress["elapsed"] = int(time.time() - progress["start_time"])
    return jsonify(progress)


@app.route("/api/batch-cancel", methods=["POST"])
def batch_cancel():
    if app_config["batch_progress"]["running"]:
        app_config["batch_progress"]["cancel"] = True
    return jsonify({"status": "ok"})


@app.route("/api/batch-history")
def batch_history_api():
    """获取批量处理历史"""
    history = []
    if BATCH_HISTORY_FILE.exists():
        try:
            with open(BATCH_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    # 按时间逆序 (最新在前)
    history.reverse()
    return jsonify(history)


# ============================================================
# 模型加载/卸载管理
# ============================================================

def _load_model_thread(model_path, dtype):
    """后台加载模型线程 (带进度模拟)"""
    state = app_config["model_state"]
    state["loading"] = True
    state["loaded"] = False
    state["error"] = None
    state["progress"] = 5
    state["status"] = "loading"

    # 进度模拟: from_pretrained 是阻塞调用且无原生进度回调，
    # 用一个定时器线程在加载期间缓慢推进到 90%，加载完成后跳到 100%
    stop_flag = [False]
    def progress_timer():
        p = 5
        while not stop_flag[0] and p < 90:
            time.sleep(1.0)
            if stop_flag[0]:
                break
            p = min(90, p + 3)
            state["progress"] = p

    timer_thread = threading.Thread(target=progress_timer, daemon=True)
    timer_thread.start()

    try:
        from local_vlm import LocalVLM
        vlm = LocalVLM(model_path, dtype=dtype)
        vlm.load()
        app_config["local_vlm"] = vlm
        stop_flag[0] = True
        timer_thread.join(timeout=2)
        state["progress"] = 100
        state["loaded"] = True
        state["status"] = "loaded"
        print(f"[OK] 模型加载完成: {model_path}")
    except Exception as e:
        stop_flag[0] = True
        timer_thread.join(timeout=2)
        state["error"] = str(e)
        state["status"] = "error"
        print(f"[ERROR] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        state["loading"] = False


def _unload_model():
    """卸载模型，释放显存"""
    vlm = app_config.get("local_vlm")
    if vlm is not None:
        try:
            vlm.unload()
        except Exception:
            pass
        app_config["local_vlm"] = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    state = app_config["model_state"]
    state["loaded"] = False
    state["loading"] = False
    state["progress"] = 0
    state["status"] = "unloaded"
    state["error"] = None
    print("[OK] 模型已卸载，显存已释放")


@app.route("/api/model/load", methods=["POST"])
def load_model_api():
    """触发模型加载 (异步)"""
    model_path = app_config.get("model_path")
    if not model_path:
        return jsonify({"error": "未配置模型路径，请使用 --local-model 参数启动"}), 400
    state = app_config["model_state"]
    if state["loading"]:
        return jsonify({"error": "模型正在加载中，请等待"}), 409
    if state["loaded"]:
        return jsonify({"status": "ok", "message": "模型已加载"})
    dtype = app_config.get("model_dtype", "bfloat16")
    threading.Thread(target=_load_model_thread, args=(model_path, dtype), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/model/unload", methods=["POST"])
def unload_model_api():
    """卸载模型"""
    state = app_config["model_state"]
    if state["loading"]:
        return jsonify({"error": "模型正在加载中，无法卸载"}), 409
    _unload_model()
    return jsonify({"status": "ok"})


@app.route("/api/model/status")
def model_status_api():
    """获取模型状态 (不泄露 api_key 明文)"""
    state = dict(app_config["model_state"])
    api_type = app_config.get("api_type", "openai")
    if api_type == "local":
        state["mode"] = "local"
    elif app_config.get("api_key"):
        state["mode"] = api_type  # "openai" / "anthropic"
    else:
        state["mode"] = "none"
    mp = app_config.get("model_path")
    state["model_path"] = mp
    state["model_name"] = os.path.basename(mp) if mp else None
    state["model_dtype"] = app_config.get("model_dtype", "bfloat16")
    state["api_key_set"] = bool(app_config.get("api_key"))
    state["api_key"] = app_config.get("api_key")  # 本地工具，回填到 password 输入框
    state["base_url"] = app_config.get("base_url")
    state["api_model"] = app_config.get("model")
    # 新增: 返回 profile 信息
    profiles_data = load_api_profiles()
    state["active_profile_id"] = profiles_data.get("active_profile_id")
    active_profile = None
    for p in profiles_data.get("profiles", []):
        if p.get("id") == state["active_profile_id"]:
            active_profile = dict(p)
            active_profile["api_key"] = mask_api_key(p.get("api_key", ""))
            break
    state["active_profile"] = active_profile
    state["profiles"] = [
        {**dict(p), "api_key": mask_api_key(p.get("api_key", ""))}
        for p in profiles_data.get("profiles", [])
    ]
    return jsonify(state)


@app.route("/api/model/config", methods=["POST"])
def set_model_config():
    """运行时切换模型模式 (local / openai / anthropic) 并配置参数"""
    data = request.json or {}
    mode = data.get("mode")

    if mode == "local":
        model_path = data.get("model_path")
        if model_path:
            app_config["model_path"] = model_path
        dtype = data.get("dtype")
        if dtype in ("bfloat16", "float16", "float32"):
            app_config["model_dtype"] = dtype
        # 只切换 api_type，保留 api_key/base_url/model 不清空，
        # 以便用户随时切回 OpenAI/Anthropic 模式时无需重新填写
        app_config["api_type"] = "local"
    elif mode in ("openai", "anthropic"):
        api_key = data.get("api_key")
        # 允许切换模式时不重新输入 key（已存在时保留）
        if not api_key and not app_config.get("api_key"):
            return jsonify({"error": "API Key 不能为空"}), 400
        if api_key:
            app_config["api_key"] = api_key
        if data.get("base_url") is not None:
            app_config["base_url"] = data.get("base_url") or None
        if data.get("model"):
            app_config["model"] = data.get("model")
        app_config["api_type"] = mode
    else:
        return jsonify({"error": "无效模式，需为 local / openai / anthropic"}), 400

    print(f"[OK] 模型配置已切换: mode={app_config.get('api_type')}, "
          f"key={'set' if app_config.get('api_key') else 'none'}, "
          f"base_url={app_config.get('base_url')}, model={app_config.get('model')}")
    save_model_config()  # 持久化到 model_config.json
    return jsonify({"status": "ok", "mode": app_config.get("api_type")})


# ============================================================
# API Profile 管理 (多 API 配置)
# ============================================================

@app.route("/api/profiles", methods=["GET"])
def list_api_profiles():
    """列出所有 API profiles，key 脱敏"""
    data = load_api_profiles()
    profiles = data.get("profiles", [])
    result = []
    for p in profiles:
        masked = dict(p)
        masked["api_key"] = mask_api_key(p.get("api_key", ""))
        result.append(masked)
    return jsonify({
        "profiles": result,
        "active_profile_id": data.get("active_profile_id")
    })


@app.route("/api/profiles", methods=["POST"])
def create_api_profile():
    """新建 API profile"""
    body = request.json or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "配置名称不能为空"}), 400
    api_type = body.get("api_type", "openai")
    if api_type not in ("openai", "anthropic"):
        return jsonify({"error": "api_type 无效"}), 400
    import uuid
    profile = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "api_type": api_type,
        "api_key": body.get("api_key", ""),
        "base_url": (body.get("base_url") or "").strip(),
        "model": (body.get("model") or "").strip(),
    }
    data = load_api_profiles()
    data.setdefault("profiles", []).append(profile)
    if len(data["profiles"]) == 1:
        data["active_profile_id"] = profile["id"]
    save_api_profiles(data)
    if data["active_profile_id"] == profile["id"]:
        sync_active_profile_to_config()
    masked = dict(profile)
    masked["api_key"] = mask_api_key(profile["api_key"])
    return jsonify({"status": "ok", "profile": masked})


@app.route("/api/profiles/<profile_id>", methods=["GET"])
def get_api_profile(profile_id):
    """获取单个 API profile（含完整 key，用于编辑）"""
    data = load_api_profiles()
    for p in data.get("profiles", []):
        if p.get("id") == profile_id:
            return jsonify({"status": "ok", "profile": dict(p)})
    return jsonify({"error": "配置不存在"}), 404


@app.route("/api/profiles/<profile_id>", methods=["PUT"])
def update_api_profile(profile_id):
    """更新 API profile"""
    data = load_api_profiles()
    profiles = data.get("profiles", [])
    target = None
    for p in profiles:
        if p.get("id") == profile_id:
            target = p
            break
    if target is None:
        return jsonify({"error": "配置不存在"}), 404
    body = request.json or {}
    if "name" in body:
        name = (body["name"] or "").strip()
        if not name:
            return jsonify({"error": "配置名称不能为空"}), 400
        target["name"] = name
    if "api_type" in body:
        if body["api_type"] not in ("openai", "anthropic"):
            return jsonify({"error": "api_type 无效"}), 400
        target["api_type"] = body["api_type"]
    if "api_key" in body:
        target["api_key"] = body["api_key"]
    if "base_url" in body:
        target["base_url"] = (body["base_url"] or "").strip()
    if "model" in body:
        target["model"] = (body["model"] or "").strip()
    save_api_profiles(data)
    if data.get("active_profile_id") == profile_id:
        sync_active_profile_to_config()
    masked = dict(target)
    masked["api_key"] = mask_api_key(target.get("api_key", ""))
    return jsonify({"status": "ok", "profile": masked})


@app.route("/api/profiles/<profile_id>", methods=["DELETE"])
def delete_api_profile(profile_id):
    """删除 API profile"""
    data = load_api_profiles()
    profiles = data.get("profiles", [])
    before = len(profiles)
    data["profiles"] = [p for p in profiles if p.get("id") != profile_id]
    if len(data["profiles"]) == before:
        return jsonify({"error": "配置不存在"}), 404
    if data.get("active_profile_id") == profile_id:
        data["active_profile_id"] = None
        app_config["api_type"] = "openai"
        app_config["api_key"] = None
        app_config["base_url"] = None
        app_config["model"] = None
    save_api_profiles(data)
    return jsonify({"status": "ok"})


@app.route("/api/profiles/<profile_id>/activate", methods=["POST"])
def activate_api_profile(profile_id):
    """激活指定 API profile"""
    data = load_api_profiles()
    found = any(p.get("id") == profile_id for p in data.get("profiles", []))
    if not found:
        return jsonify({"error": "配置不存在"}), 404
    data["active_profile_id"] = profile_id
    save_api_profiles(data)
    sync_active_profile_to_config()
    return jsonify({"status": "ok", "active_profile_id": profile_id})


@app.route("/api/label-config", methods=["GET"])
def get_label_config():
    """获取标签配置"""
    return jsonify(load_label_config())


@app.route("/api/last-position", methods=["GET"])
def get_last_position():
    """获取上次查看位置"""
    position = load_last_position()
    if position:
        return jsonify(position)
    return jsonify({"filename": None, "timestamp": None})


@app.route("/api/last-position", methods=["POST"])
def save_last_position_api():
    """保存当前查看位置"""
    data = request.json
    filename = data.get("filename") if data else None
    if filename:
        save_last_position(filename)
        return jsonify({"status": "ok"})
    return jsonify({"error": "filename is required"}), 400


@app.route("/api/role-names", methods=["GET"])
def get_role_names():
    """获取角色名称数据"""
    role_file = BASE_DIR / "role_name.json"
    if role_file.exists():
        with open(role_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/label-config", methods=["POST"])
def update_label_config():
    """更新标签配置"""
    config = request.json
    save_label_config(config)
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def stats():
    """获取统计信息"""
    files = get_image_list()
    annotations = load_annotations()
    total = len(files)
    annotated = sum(1 for f in files if f in annotations)
    auto_labeled = sum(
        1 for f in files
        if f in annotations and annotations[f].get("auto_labeled")
    )
    verified = sum(
        1 for f in files
        if f in annotations and annotations[f].get("verified")
    )
    return jsonify({
        "total": total,
        "annotated": annotated,
        "auto_labeled": auto_labeled,
        "verified": verified,
        "remaining": total - annotated,
        "progress": round(annotated / total * 100, 1) if total > 0 else 0
    })


@app.route("/api/storage-stats")
def storage_stats():
    """获取图片存储分布统计"""
    files = get_image_list()
    buckets = {"gt10": 0, "7to10": 0, "4to7": 0, "1to4": 0, "lt1": 0}
    total_bytes = 0
    for f in files:
        filepath = IMAGE_DIR / f
        try:
            size_mb = filepath.stat().st_size / (1024 * 1024)
        except OSError:
            continue
        total_bytes += filepath.stat().st_size
        if size_mb > 10:
            buckets["gt10"] += 1
        elif size_mb >= 7:
            buckets["7to10"] += 1
        elif size_mb >= 4:
            buckets["4to7"] += 1
        elif size_mb >= 1:
            buckets["1to4"] += 1
        else:
            buckets["lt1"] += 1
    return jsonify({
        "total": len(files),
        "total_gb": round(total_bytes / (1024 ** 3), 2),
        "buckets": buckets
    })


COMPACT_DIR = BASE_DIR / "compact"


# 压缩强度预设: (target_mb, max_dim, quality_floor)
COMPACT_PRESETS = {
    1: (3.0, 8192, 75),   # 轻度 — 几乎不缩放，高质量
    2: (2.0, 4096, 65),   # 较轻
    3: (1.5, 3072, 55),   # 标准
    4: (1.0, 2048, 45),   # 较强
    5: (0.6, 1280, 35),   # 强力
}


@app.route("/api/compact-images", methods=["POST"])
def compact_images():
    """压缩所有 >7MB 的图片到 compact/ 目录，返回压缩报告"""
    data = request.json or {}
    level = int(data.get("level", 3))
    if level not in COMPACT_PRESETS:
        level = 3
    target_mb, max_dim, quality_floor = COMPACT_PRESETS[level]

    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    files = get_image_list()
    large = []
    for f in files:
        fp = IMAGE_DIR / f
        size_mb = fp.stat().st_size / (1024 * 1024)
        if size_mb >= 7:
            large.append((f, fp, size_mb))

    if not large:
        return jsonify({"status": "ok", "message": "没有需要压缩的大图片", "count": 0})

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_before = 0
    results = []
    workers = min(8, len(large))

    def _compact_one(fname, fp, size_mb):
        """单张图片压缩（线程安全）"""
        try:
            img = Image.open(fp)
            original_size = fp.stat().st_size
            out_path = COMPACT_DIR / fname
            ext = out_path.suffix.lower()

            if ext in (".jpg", ".jpeg"):
                img = _limit_dimensions(img, max_dim)
                _save_jpeg_compact(img, out_path, target_mb=target_mb, quality_floor=quality_floor)
            elif ext == ".png":
                if img.mode == "RGBA" and _has_transparency(img):
                    img = _limit_dimensions(img, max_dim)
                    # 颜色量化: 等级1→256色, 等级5→64色
                    colors = max(64, 320 - level * 64)
                    try:
                        img = img.quantize(colors=colors, method=Image.Quantize.LIBIMAGEQUANT, dither=Image.Dither.FLOYDSTEINBERG)
                    except (ValueError, OSError):
                        img = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
                    img.save(out_path, format="PNG", optimize=True)
                else:
                    img = img.convert("RGB")
                    img = _limit_dimensions(img, max_dim)
                    rgb_path = out_path.with_suffix(".jpg")
                    _save_jpeg_compact(img, rgb_path, target_mb=target_mb, quality_floor=quality_floor)
                    out_path = rgb_path
            elif ext in (".bmp", ".gif", ".webp"):
                img = img.convert("RGB")
                img = _limit_dimensions(img, max_dim)
                jpg_path = out_path.with_suffix(".jpg")
                _save_jpeg_compact(img, jpg_path, target_mb=target_mb, quality_floor=quality_floor)
                out_path = jpg_path
            else:
                img = _limit_dimensions(img, max_dim)
                img.save(out_path, optimize=True)

            compressed_size = out_path.stat().st_size
            ratio = round(compressed_size / original_size * 100, 1)
            msg = f"[Compact] {fname}: {original_size/(1024*1024):.1f}MB → {compressed_size/(1024*1024):.1f}MB ({ratio}%)"
            return {
                "file": fname,
                "original_mb": round(original_size / (1024 * 1024), 2),
                "compressed_mb": round(compressed_size / (1024 * 1024), 2),
                "ratio_pct": ratio,
                "_orig_bytes": original_size,
                "_comp_bytes": compressed_size,
                "log": msg,
            }
        except Exception as e:
            print(f"[Compact] 失败 {fname}: {e}")
            return {"file": fname, "error": str(e), "_orig_bytes": fp.stat().st_size, "_comp_bytes": 0}

    print(f"[Compact] 并行压缩 {len(large)} 张图片，{workers} 线程 ...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_compact_one, f, fp, sz): f for f, fp, sz in large}
        for future in as_completed(futures):
            r = future.result()
            if "log" in r:
                print(r.pop("log"))
            results.append(r)

    total_before = sum(r.get("_orig_bytes", 0) for r in results)
    total_after = sum(r.get("_comp_bytes", 0) for r in results)
    # 清理内部字段
    for r in results:
        r.pop("_orig_bytes", None)
        r.pop("_comp_bytes", None)
    saved_mb = round((total_before - total_after) / (1024 * 1024), 2)
    return jsonify({
        "status": "ok",
        "count": len(large),
        "total_before_mb": round(total_before / (1024 * 1024), 2),
        "total_after_mb": round(total_after / (1024 * 1024), 2),
        "saved_mb": saved_mb,
        "saved_pct": round((total_before - total_after) / total_before * 100, 1) if total_before > 0 else 0,
        "results": results,
    })


@app.route("/api/delete-large-images", methods=["POST"])
def delete_large_images():
    """删除 IMAGE_DIR 下 >7MB 的图片及其标注（需先在 compact/ 有备份）"""
    if not COMPACT_DIR.exists() or not any(COMPACT_DIR.iterdir()):
        return jsonify({"error": "请先执行压缩再删除"}), 400
    files = get_image_list()
    deleted = []
    with annotation_lock:
        annotations = load_annotations()
        for f in files:
            fp = IMAGE_DIR / f
            if fp.stat().st_size / (1024 * 1024) >= 7:
                try:
                    fp.unlink()
                except OSError as e:
                    print(f"[Delete] 无法删除 {f}: {e}")
                    continue
                if f in annotations:
                    del annotations[f]
                deleted.append(f)
        save_annotations(annotations)
    print(f"[Delete] 已删除 {len(deleted)} 张大图片")
    return jsonify({"status": "ok", "deleted": len(deleted), "files": deleted})


def _limit_dimensions(img, max_dim):
    """限制图片最大边长"""
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _has_transparency(img):
    """检查 RGBA 图片是否有实际透明像素"""
    alpha = img.split()[3]
    return alpha.getextrema()[0] < 255


def _save_jpeg_compact(img, path, target_mb=3.0, quality_floor=50):
    """保存 JPEG 逐步降质量到目标大小以下"""
    import io as _io
    quality = 85
    while quality >= quality_floor:
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        size_mb = buf.tell() / (1024 * 1024)
        if size_mb <= target_mb or quality == quality_floor:
            with open(path, "wb") as f:
                f.write(buf.getvalue())
            return
        quality -= 10


@app.route("/api/export")
def export_annotations():
    """导出标注数据"""
    fmt = request.args.get("format", "json")
    annotations = load_annotations()

    if fmt == "csv":
        import csv
        import io as _io
        output = _io.StringIO()
        config = load_label_config()

        all_categories = list(config.keys())
        fieldnames = ["filename"] + all_categories + ["custom_tags", "description", "review", "auto_labeled", "verified"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for filename, ann in annotations.items():
            row = {"filename": filename}
            labels = ann.get("labels", {})
            for cat in all_categories:
                val = labels.get(cat, "")
                if isinstance(val, list):
                    val = ";".join(val)
                row[cat] = val
            row["custom_tags"] = ";".join(ann.get("custom_tags", []))
            row["description"] = ann.get("description", "")
            row["review"] = ann.get("review", "")
            row["auto_labeled"] = ann.get("auto_labeled", False)
            row["verified"] = ann.get("verified", False)
            writer.writerow(row)

        from flask import Response
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=annotations.csv"}
        )
    else:
        from flask import Response
        return Response(
            json.dumps(annotations, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=annotations.json"}
        )


@app.route("/api/verify/<path:filename>", methods=["POST"])
def verify_annotation(filename):
    """标记标注已验证"""
    with annotation_lock:
        annotations = load_annotations()
        if filename in annotations:
            annotations[filename]["verified"] = True
            save_annotations(annotations)
        return jsonify({"status": "ok"})
    return jsonify({"error": "标注不存在"}), 404


@app.route("/api/generate-description/<path:filename>", methods=["POST"])
def generate_description(filename):
    """根据标签生成图片描述"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    annotations = load_annotations()
    ann = annotations.get(filename)

    if not ann or not ann.get("labels"):
        return jsonify({"error": "该图片还没有标签，无法生成描述"}), 400

    labels = ann.get("labels", {})
    description = _generate_description_from_labels(labels)

    def updater(existing):
        return {
            "description": description,
            "description_history": [{"role": "assistant", "content": description}],
            "updated_at": datetime.now().isoformat()
        }
    merged = atomic_update_annotation(filename, updater)

    return jsonify({"status": "ok", "description": description, "history": merged.get("description_history", [])})



def _generate_desc_core(filepath, enable_thinking=False, crop=None):
    """半自由描述核心逻辑 (首次生成): 构建 prompt -> 调模型 -> 返回 description 字符串"""
    config = load_label_config()
    categories = list(config.keys())
    categories_desc = "、".join(categories)

    prompt = f"""你是一位专业的图像描述师。请用**一段连贯的自然语言**描述这张图片，要求：

**内容维度（需自然融入，不要分点）：**
- 人物特征：发型、表情、姿势、服装等
- 环境氛围：场景、光线、色调、空间感（已有的标签分类参考：{categories_desc}）
- 细节捕捉：任何你认为重要的视觉元素、情绪氛围或者故事情节

**写作要求：**
- 用正常自然语言，像文学描写一样流畅，同时不要斛揉造作、太多的不必要比喻，要简洁扼要
- 不要出现“标签是...”、“特征包括...”这类结构化表达
- 不要列举，要描绘；不要说明，要呈现
- 要包含传递画面的整体氛围和视觉感受

请直接输出描述段落，不要加标题或分点。"""

    if crop:
        prompt += "\n\n" + f"【注意】用户截取了图片的一个区域（选区尺寸 {crop.get('w')}×{crop.get('h')}px），本次输入的图像即为该选区内容。"

    pil_crop = _crop_image(filepath, crop) if crop else None
    if app_config["api_type"] == "local":
        vlm = app_config.get("local_vlm")
        if vlm is None:
            raise RuntimeError("本地模型未加载")
        if pil_crop is not None:
            desc = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking, pil_image=pil_crop)
        else:
            desc = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
        return _extract_description_from_response(desc)
    elif app_config["api_type"] == "anthropic":
        return _call_anthropic_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)
    else:
        return _call_openai_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)


def _generate_review_core(filepath, labels, enable_thinking=False, crop=None):
    """内容审核核心逻辑 (首次生成): 构建 prompt -> 调模型 -> 返回 review 字符串"""
    review_labels = labels or {}
    if isinstance(review_labels, str):
        review_labels = [review_labels]
    labels_info = json.dumps(review_labels, ensure_ascii=False, indent=2) if review_labels else "无"

    prompt = f"""你是一个平台内容审核员。请根据已有的图片标签信息，客观分析这张图片的内容，用一段话输出审核结果，不分点。

已有标签信息：
{labels_info}

**输出要求：**
- 用一段连贯的话输出你的描述
- 分析画面的主体内容、氛围和视觉元素
- 评估内容的适宜程度
- 客观描述，不要添加额外的内容或解释该怎么做，专注于画面。

请直接输出审核结果段落，不要加标题或分点。"""

    if crop:
        prompt += "\n\n" + f"【注意】用户截取了图片的一个区域（选区尺寸 {crop.get('w')}×{crop.get('h')}px）。"

    pil_crop = _crop_image(filepath, crop) if crop else None
    if app_config["api_type"] == "local":
        vlm = app_config.get("local_vlm")
        if vlm is None:
            raise RuntimeError("本地模型未加载")
        if pil_crop is not None:
            review = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking, pil_image=pil_crop)
        else:
            review = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
        return _extract_description_from_response(review)
    elif app_config["api_type"] == "anthropic":
        return _call_anthropic_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)
    else:
        return _call_openai_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)



@app.route("/api/generate-semi-free-description/<path:filename>", methods=["POST"])
def generate_semi_free_description(filename):
    """调用模型生成半自由描述 (支持多轮对话)"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    data = request.json or {}
    enable_thinking = data.get("enable_thinking", False)
    user_input = data.get("user_input", "")
    crop = data.get("crop")  # {x, y, w, h} 选区坐标 (natural)，None 表示全图

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    annotations = load_annotations()
    ann = annotations.get(filename)
    labels = ann.get("labels", {}) if ann else {}

    if filename not in annotations:
        annotations[filename] = {
            "labels": {},
            "custom_tags": [],
            "auto_labeled": False,
            "verified": False,
            "updated_at": datetime.now().isoformat()
        }

    config = load_label_config()
    categories = list(config.keys())
    categories_desc = "、".join(categories)

    # 获取已有描述对话历史
    existing_history = list(annotations.get(filename, {}).get("description_history", []))

    if user_input:
        # 追问模式: 基于历史对话继续
        existing_history.append({"role": "user", "content": user_input})
        history_str = "\n\n".join(
            f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in existing_history[:-1]
        )
        prompt = f"""你是一位专业的图像描述师。用户对你之前的描述提出了追问，请基于图片和之前的对话继续回答。

之前的对话历史：
{history_str}

用户追问：{user_input}

请回答，不要加标题或分点。"""
    else:
        # 首次生成描述
        prompt = f"""你是一位专业的图像描述师。请用**一段连贯的自然语言**描述这张图片，要求：

**内容维度（需自然融入，不要分点）：**
- 人物特征：发型、表情、姿势、服装等
- 环境氛围：场景、光线、色调、空间感（已有的标签分类参考：{categories_desc}）
- 细节捕捉：任何你认为重要的视觉元素、情绪氛围或者故事情节

**写作要求：**
- 用正常自然语言，像文学描写一样流畅，同时不要矫揉造作、太多的不必要比喻（如"像"、""仿佛""），要简洁扼要
- 不要出现"标签是..."、"特征包括..."这类结构化表达
- 不要列举，要描绘；不要说明，要呈现
- 要包含传递画面的整体氛围和视觉感受

请直接输出描述段落，不要加标题或分点。"""

    # 选区提示: 让模型明确知道收到的是裁剪区域
    if crop:
        prompt += f"\n\n【注意】用户截取了图片的一个区域（选区尺寸 {crop.get('w')}×{crop.get('h')}px），本次输入的图像即为该选区内容，请针对选区可见内容重点分析。"

    try:
        pil_crop = _crop_image(filepath, crop) if crop else None
        if app_config["api_type"] == "local":
            vlm = app_config.get("local_vlm")
            if vlm is None:
                return jsonify({"error": "本地模型未加载"}), 400
            if pil_crop is not None:
                description = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking, pil_image=pil_crop)
            else:
                description = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
            description = _extract_description_from_response(description)
        elif app_config["api_type"] == "anthropic":
            description = _call_anthropic_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)
        else:
            description = _call_openai_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)

        # 原子保存: 锁内重新加载最新数据，避免与并发 save_annotation 请求互相覆盖
        def updater(existing):
            fresh_history = list(existing.get("description_history", []))
            if user_input:
                user_msg = {"role": "user", "content": user_input}
                if crop:
                    user_msg["crop"] = crop
                fresh_history.append(user_msg)
            asst_msg = {"role": "assistant", "content": description}
            if crop and not user_input:
                asst_msg["crop"] = crop  # 首次生成: crop 关联到 assistant 消息
            fresh_history.append(asst_msg)
            return {
                "description": description,
                "description_history": fresh_history,
                "updated_at": datetime.now().isoformat()
            }
        merged = atomic_update_annotation(filename, updater)

        return jsonify({"status": "ok", "description": description, "history": merged.get("description_history", [])})
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return jsonify({"error": error_msg}), 500


@app.route("/api/generate-review/<path:filename>", methods=["POST"])
def generate_review(filename):
    """调用模型生成审核结果"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    data = request.json or {}
    enable_thinking = data.get("enable_thinking", False)
    crop = data.get("crop")  # {x, y, w, h} 选区坐标，None 表示全图

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    annotations = load_annotations()
    ann = annotations.get(filename)
    labels = ann.get("labels", {}) if ann else {}

    if filename not in annotations:
        annotations[filename] = {
            "labels": {},
            "custom_tags": [],
            "auto_labeled": False,
            "verified": False,
            "updated_at": datetime.now().isoformat()
        }

    config = load_label_config()
    data = request.json or {}
    user_input = data.get("user_input", "")

    # 获取已有的审核历史
    existing_history = list(annotations.get(filename, {}).get("review_history", []))

    if user_input:
        # 用户追问模式
        existing_history.append({"role": "user", "content": user_input})

        prompt = f"""你是一个平台内容审核员。用户对你之前的审核结果提出了追问，请基于图片和之前的对话继续回答。

之前的对话历史：
{chr(10).join(f'{"用户" if h["role"]=="user" else "助手"}: {h["content"]}' for h in existing_history[:-1])}

用户追问：{user_input}

请回答，不要加标题或分点。"""

        if crop:
            prompt += f"\n\n【注意】用户截取了图片的一个区域（选区尺寸 {crop.get('w')}×{crop.get('h')}px），本次输入的图像即为该选区内容，请针对选区可见内容重点分析。"

        try:
            pil_crop = _crop_image(filepath, crop) if crop else None
            if app_config["api_type"] == "local":
                vlm = app_config.get("local_vlm")
                if vlm is None:
                    return jsonify({"error": "本地模型未加载"}), 400
                if pil_crop is not None:
                    reply = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking, pil_image=pil_crop)
                else:
                    reply = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
                reply = _extract_description_from_response(reply)
            elif app_config["api_type"] == "anthropic":
                reply = _call_anthropic_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)
            else:
                reply = _call_openai_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)

            existing_history.append({"role": "assistant", "content": reply})
            # 原子保存: 锁内重新加载，避免并发覆盖 description_history 等
            def updater_followup(existing):
                fresh = list(existing.get("review_history", []))
                user_msg = {"role": "user", "content": user_input}
                if crop:
                    user_msg["crop"] = crop
                fresh.append(user_msg)
                fresh.append({"role": "assistant", "content": reply})
                return {
                    "review": reply,
                    "review_history": fresh,
                    "updated_at": datetime.now().isoformat()
                }
            merged = atomic_update_annotation(filename, updater_followup)

            return jsonify({"status": "ok", "history": merged.get("review_history", [])})
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return jsonify({"error": error_msg}), 500
    else:
        # 首次生成审核
        labels_info = json.dumps(labels, ensure_ascii=False, indent=2) if labels else "无"

        prompt = f"""你是一个平台内容审核员。请根据已有的图片标签信息，客观分析这张图片的内容，用一段话输出审核结果，不分点。

已有标签信息：
{labels_info}

**输出要求：**
- 用一段连贯的话输出你的描述
- 分析画面的主体内容、氛围和视觉元素
- 评估内容的适宜程度
- 客观描述，不要添加额外的内容或解释该怎么做，专注于画面。

请直接输出审核结果段落，不要加标题或分点。"""

        if crop:
            prompt += f"\n\n【注意】用户截取了图片的一个区域（选区尺寸 {crop.get('w')}×{crop.get('h')}px），本次输入的图像即为该选区内容，请针对选区可见内容重点分析。"

        try:
            pil_crop = _crop_image(filepath, crop) if crop else None
            if app_config["api_type"] == "local":
                vlm = app_config.get("local_vlm")
                if vlm is None:
                    return jsonify({"error": "本地模型未加载"}), 400
                if pil_crop is not None:
                    review = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking, pil_image=pil_crop)
                else:
                    review = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
                review = _extract_description_from_response(review)
            elif app_config["api_type"] == "anthropic":
                review = _call_anthropic_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)
            else:
                review = _call_openai_for_description(str(filepath), prompt, crop=crop, enable_thinking=enable_thinking)

            # 原子保存
            def updater_first(existing):
                asst_msg = {"role": "assistant", "content": review}
                if crop:
                    asst_msg["crop"] = crop  # 首次审核: crop 关联到 assistant 消息
                return {
                    "review": review,
                    "review_history": [asst_msg],
                    "updated_at": datetime.now().isoformat()
                }
            merged = atomic_update_annotation(filename, updater_first)

            return jsonify({"status": "ok", "review": review, "history": merged.get("review_history", [])})
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return jsonify({"error": error_msg}), 500


def _call_openai_for_description(filepath, prompt, crop=None, enable_thinking=False):
    """调用 OpenAI API 生成描述 (crop 非空时发送裁剪后的图片)"""
    import urllib.request
    import urllib.error

    b64 = _crop_to_base64(filepath, crop) if crop else image_to_base64(filepath)

    payload = {
        "model": app_config.get("model") or "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.7
    }
    if enable_thinking:
        payload["enable_thinking"] = True
        payload["extra_body"] = {"enable_thinking": True}
        payload["stream"] = True

    url = _build_openai_url(app_config.get("base_url"))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {app_config.get('api_key')}"
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        _batch_http_resp[0] = resp
        try:
            if enable_thinking:
                text = _parse_sse_stream(resp)
            else:
                raw = resp.read().decode()
                result = json.loads(raw)
                text = result["choices"][0]["message"]["content"].strip()
        finally:
            resp.close()
            _batch_http_resp[0] = None
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500] if hasattr(e, 'read') else ''
        raise RuntimeError(f"OpenAI API HTTP {e.code} {e.reason} | URL: {url} | 响应: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI API 连接失败: {e.reason} | URL: {url}")

    return _extract_description_from_response(text)


def _parse_sse_stream(resp):
    """解析 SSE 流式响应，拼接 delta.content 返回完整文本"""
    content_parts = []
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if not line_str or not line_str.startswith("data:"):
            continue
        data_str = line_str[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                c = delta.get("content", "")
                if c:
                    content_parts.append(c)
        except json.JSONDecodeError:
            continue
    return "".join(content_parts).strip()


def _call_anthropic_for_description(filepath, prompt, crop=None, enable_thinking=False):
    """调用 Anthropic API 生成描述 (crop 非空时发送裁剪后的图片)"""
    import urllib.request

    b64 = _crop_to_base64(filepath, crop) if crop else image_to_base64(filepath)

    payload = {
        "model": app_config.get("model") or "claude-sonnet-4-20250514",
        "max_tokens": 800,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled", "budget_tokens": 16000}
        payload["max_tokens"] = max(payload["max_tokens"], 16000 + 800)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": app_config.get("api_key"),
        "anthropic-version": "2023-06-01"
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers=headers
    )
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        _batch_http_resp[0] = resp
        result = json.loads(resp.read().decode())
        resp.close()
        _batch_http_resp[0] = None
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500] if hasattr(e, 'read') else ''
        raise RuntimeError(f"Anthropic API HTTP {e.code} {e.reason} | 响应: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Anthropic API 连接失败: {e.reason}")

    text = result["content"][0]["text"].strip()
    return _extract_description_from_response(text)


def _extract_description_from_response(text):
    """从模型响应中提取描述文本"""
    if not isinstance(text, str):
        return str(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


def _generate_description_from_labels(labels):
    """根据标签内容生成描述文本"""
    parts = []

    gender = labels.get("性别", "")
    if gender:
        parts.append(f"一个{gender}角色")

    hair_color = labels.get("发色", [])
    if hair_color:
        colors = "、".join(hair_color) if isinstance(hair_color, list) else hair_color
        parts.append(f"头发颜色为{colors}")

    hair_style = labels.get("发型", [])
    if hair_style:
        styles = "、".join(hair_style) if isinstance(hair_style, list) else hair_style
        parts.append(f"发型为{styles}")

    eye_color = labels.get("瞳色", [])
    if eye_color:
        colors = "、".join(eye_color) if isinstance(eye_color, list) else eye_color
        parts.append(f"眼睛颜色为{colors}")

    features = labels.get("角色特征", [])
    if features:
        feat_str = "、".join(features) if isinstance(features, list) else features
        parts.append(f"具有{feat_str}")

    clothing = labels.get("服装", [])
    if clothing:
        cloth_str = "、".join(clothing) if isinstance(clothing, list) else clothing
        parts.append(f"穿着{cloth_str}")

    pose = labels.get("姿势", [])
    if pose:
        pose_str = "、".join(pose) if isinstance(pose, list) else pose
        parts.append(f"姿势为{pose_str}")

    background = labels.get("背景", [])
    if background:
        bg_str = "、".join(background) if isinstance(background, list) else background
        parts.append(f"背景为{bg_str}")

    style = labels.get("画面风格", [])
    if style:
        style_str = "、".join(style) if isinstance(style, list) else style
        parts.append(f"画面风格为{style_str}")

    count = labels.get("人物数量", "")
    if count:
        parts.append(f"共{count}")

    if not parts:
        return "一张图片"

    description = "，".join(parts)
    if not description.endswith("。"):
        description += "。"
    return description


@app.route("/api/convert-to-danbooru/<path:filename>", methods=["POST"])
def convert_to_danbooru(filename):
    """将图片描述转换为 Danbooru 标签格式（直接复用已加载的 Qwen3.5 模型）"""
    if not _has_auto_label():
        return jsonify({"error": "本地模型未加载。请使用 --local-model 参数启动。"}), 400

    # 读取当前图片的描述和已有标签
    annotations = load_annotations()
    ann = annotations.get(filename, {})
    description = ann.get("description", "").strip()
    labels = ann.get("labels", {})

    data = request.json or {}
    description = data.get("text", description).strip()
    if not description:
        return jsonify({"error": "没有可转换的描述文本。请先生成半自由描述。"}), 400

    # 从已有标签中提取关键信息作为上下文
    labels_context = ""
    if labels:
        parts = []
        for cat, vals in labels.items():
            if vals:
                v = ", ".join(vals) if isinstance(vals, list) else vals
                if v:
                    parts.append(f"{cat}: {v}")
        if parts:
            labels_context = "\\n已知图片标签: " + "; ".join(parts)

    prompt = f"""You are a Danbooru tag converter. Convert the following image description into Danbooru tags.

RULES:
1. Output ONLY comma-separated Danbooru tags, one line, no markdown, no explanation
2. Tag order: [count] -> [character] -> [composition] -> [appearance: hair/eyes] -> [clothing] -> [pose] -> [expression] -> [accessories] -> [environment] -> [lighting] -> [style]
3. Always start with 1girl/1boy + solo if single character
4. Use precise Danbooru clothing, accessory, environment tags
5. End with atmosphere/style tags
{labels_context}

Image description:
{description}

Danbooru tags:"""

    vlm = app_config.get("local_vlm")
    if vlm is None:
        return jsonify({"error": "本地模型未加载，请先加载模型"}), 400
    try:
        raw_tags = vlm.generate_text_only(prompt, max_new_tokens=256, temperature=0.5)
    except Exception as e:
        return jsonify({"error": f"模型推理失败: {str(e)}"}), 500

    # 清理
    raw_tags = raw_tags.strip()
    # 去除可能的 markdown
    import re
    raw_tags = re.sub(r'^```\w*\n?', '', raw_tags)
    raw_tags = re.sub(r'\n?```$', '', raw_tags)
    raw_tags = raw_tags.strip()

    return jsonify({
        "status": "ok",
        "danbooru_tags": raw_tags,
        "source_description": description[:200] + ("..." if len(description) > 200 else ""),
    })


@app.route("/api/open-folder/image/<path:filename>", methods=["POST"])
def open_image_folder(filename):
    """打开图片所在文件夹"""
    try:
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # 获取文件夹路径
        folder_path = filepath.parent
        
        # 使用 explorer 打开文件夹并选中文件
        import subprocess
        subprocess.Popen(f'explorer /select,"{filepath}"', shell=True)
        
        return jsonify({"status": "ok", "path": str(folder_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/open-folder/annotations", methods=["POST"])
def open_annotations_folder():
    """打开标注文件所在文件夹"""
    try:
        if not ANNOTATIONS_FILE.exists():
            return jsonify({"error": "标注文件不存在"}), 404
        
        # 获取文件夹路径
        folder_path = ANNOTATIONS_FILE.parent
        
        # 使用 explorer 打开文件夹并选中文件
        import subprocess
        subprocess.Popen(f'explorer /select,"{ANNOTATIONS_FILE}"', shell=True)
        
        return jsonify({"status": "ok", "path": str(folder_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image/<path:filename>", methods=["DELETE"])
def delete_image(filename):
    """删除图片文件及其标注"""
    try:
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # 删除图片文件
        filepath.unlink()
        
        # 删除标注数据（如果存在）
        with annotation_lock:
            annotations = load_annotations()
            if filename in annotations:
                del annotations[filename]
                save_annotations(annotations)
        
        return jsonify({"status": "ok", "message": f"已删除 {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="图像多标签标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python annotate.py                                    # 仅手动标注
  python annotate.py --local-model F:/qwen3_5           # 本地模型自动标注
  python annotate.py --api-key YOUR_KEY --api-type openai   # OpenAI API
  python annotate.py --api-key YOUR_KEY --api-type anthropic  # Claude API
        """
    )
    parser.add_argument("--port", type=int, default=5000, help="端口号 (默认 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--api-key", help="远程 API Key (用于在线 API 自动标注)")
    parser.add_argument("--api-type", choices=["openai", "anthropic"], default="openai",
                        help="远程 API 类型 (默认 openai)")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--base-url", help="OpenAI 兼容 API 的 base URL")
    parser.add_argument("--local-model", metavar="PATH",
                        help="本地 VLM 模型路径 (如 F:/qwen3_5)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="本地模型精度 (默认 bfloat16)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--defer-load", action="store_true",
                        help="延迟加载本地模型 (启动时不加载，通过界面按需加载)")
    parser.add_argument("--pose-model", action="store_true",
                        help="启用 DWPose 姿态估计")
    parser.add_argument("--pose-device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="姿态模型设备 (默认 cuda)")
    parser.add_argument("--pose-input-size", type=int, default=2048,
                        help="姿态估计输入图像最大尺寸 (默认 2048)")
    parser.add_argument("--pose-bbox-scale", type=float, default=2.0,
                        help="边界框扩展系数 (默认 2.0)")
    parser.add_argument("--pose-conf-thr", type=float, default=0.45,
                        help="人物检测置信度阈值 (默认 0.45)")
    args = parser.parse_args()

    # 加载已保存的模型配置 — 仅本地模型部分 (API 部分由 profiles 管理)
    saved = load_model_config()
    if saved:
        for k in ("model_path", "model_dtype"):
            if saved.get(k) is not None:
                app_config[k] = saved[k]
        if saved.get("model_path"):
            print(f"[INFO] 已恢复模型路径: {saved['model_path']}")

    # 迁移旧 API 配置到 profiles（首次）
    migrated = migrate_old_to_profiles()

    # 恢复上次激活的 API profile
    profiles_data = load_api_profiles()
    active_id = profiles_data.get("active_profile_id")
    if active_id:
        sync_active_profile_to_config()
        active_profile = None
        for p in profiles_data.get("profiles", []):
            if p.get("id") == active_id:
                active_profile = p
                break
        if active_profile:
            print(f"[INFO] 已恢复 API Profile: {active_profile.get('name')} ({active_profile.get('api_type')})")
            if active_profile.get("base_url"):
                print(f"       Base URL: {active_profile['base_url']}")
            if active_profile.get("model"):
                print(f"       模型: {active_profile['model']}")

    # CLI 参数覆盖保存的配置
    if args.local_model:
        app_config["model_path"] = args.local_model
        app_config["model_dtype"] = args.dtype
        app_config["api_type"] = "local"
    if args.api_key:
        app_config["api_key"] = args.api_key
        app_config["api_type"] = args.api_type
        if args.model:
            app_config["model"] = args.model
        if args.base_url:
            app_config["base_url"] = args.base_url

    # 根据最终配置决定加载行为
    if app_config.get("api_type") == "local" and app_config.get("model_path"):
        if args.defer_load:
            print(f"[INFO] 模型路径已配置: {app_config['model_path']}")
            print(f"       --defer-load 已启用，启动时不加载模型到显存")
            print(f"       可在界面点击「加载模型」按钮，或使用自动标注等功能时按需加载")
            app_config["model_state"]["status"] = "unloaded"
        else:
            print(f"[INFO] 正在加载本地模型: {app_config['model_path']}")
            print(f"       精度: {app_config.get('model_dtype', 'bfloat16')}")
            print(f"       首次加载可能需要 1-2 分钟...")
            try:
                from local_vlm import LocalVLM
                vlm = LocalVLM(app_config["model_path"], dtype=app_config.get("model_dtype", "bfloat16"))
                vlm.load()
                app_config["local_vlm"] = vlm
                app_config["model_state"]["loaded"] = True
                app_config["model_state"]["status"] = "loaded"
                app_config["model_state"]["progress"] = 100
                print(f"[OK] 本地模型加载成功, 自动标注已启用")
            except ImportError as e:
                print(f"[ERROR] 缺少依赖: {e}")
                print(f"        请安装: pip install torch")
                sys.exit(1)
            except Exception as e:
                print(f"[ERROR] 模型加载失败: {e}")
                sys.exit(1)
    elif app_config.get("api_type") in ("openai", "anthropic"):
        if app_config.get("api_key"):
            print(f"[OK] 已配置 {app_config['api_type']} API, 自动标注已启用")
            if app_config.get("base_url"):
                print(f"     Base URL: {app_config['base_url']}")
            if app_config.get("model"):
                print(f"     模型: {app_config['model']}")
        else:
            print(f"[INFO] {app_config['api_type']} API 模式但未配置 Key")
            print(f"       请在界面中点击 VLM 标识 -> 模式设置 配置 API Key")
    else:
        print("[INFO] 未配置自动标注, 仅手动标注模式")
        print("       启用自动标注:")
        print("         本地模型: python annotate.py --local-model F:/qwen3_5 --defer-load")
        print("         远程 API: python annotate.py --api-key YOUR_KEY --api-type openai")
        print("         或启动后在界面中点击 VLM 标识 -> 模式设置")

    # 配置姿态估计
    if args.pose_model:
        print(f"[INFO] 正在加载姿态估计模型 (DWPose)...")
        try:
            from pose_estimator import PoseEstimator
            pose_est = PoseEstimator(
                device=args.pose_device,
                input_size=args.pose_input_size,
                bbox_scale=args.pose_bbox_scale,
                conf_thr=args.pose_conf_thr
            )
            pose_est.load()
            app_config["pose_estimator"] = pose_est
            print(f"[OK] 姿态估计模型加载成功")
        except ImportError as e:
            print(f"[ERROR] 缺少依赖: {e}")
            print(f"        请安装: pip install controlnet-aux")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 姿态模型加载失败: {e}")
            sys.exit(1)
    else:
        print("[INFO] 姿态估计未启用，使用 --pose-model 启用")

    print(f"\n  标注工具已启动: http://localhost:{args.port}")
    print(f"  图片目录: {IMAGE_DIR}")
    print(f"  标注文件: {ANNOTATIONS_FILE}")
    print(f"  图片总数: {len(get_image_list())}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
