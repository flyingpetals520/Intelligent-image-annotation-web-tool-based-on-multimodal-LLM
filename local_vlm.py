"""
本地多模态 VLM 自动标注模块
支持 Qwen3.5-4B 等基于 HuggingFace Transformers 的视觉语言模型
"""

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_CONFIG_FILE = Path(__file__).parent / "label_config.json"


def load_label_config():
    if LABEL_CONFIG_FILE.exists():
        with open(LABEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_prompt(config):
    """根据标签配置构建提示词 (只包含启用的分类)"""
    enabled_config = {cat: info for cat, info in config.items() if info.get("enabled", True)}
    categories_desc = []
    for cat, info in enabled_config.items():
        mode = "单选" if not info.get("multi", False) else "多选"
        labels = ", ".join(info["labels"])
        categories_desc.append(f"  \"{cat}\"({mode}): [{labels}]")

    example = {}
    for cat, info in enabled_config.items():
        if info.get("multi", False):
            example[cat] = ["标签1"]
        else:
            example[cat] = "标签"
    example_str = json.dumps(example, ensure_ascii=False, indent=2)

    prompt = f"""你是一个专业的动漫/插画图像标注助手。请仔细观察这张图片，为以下每个分类选择最合适的标签。

分类列表:
{chr(10).join(categories_desc)}

要求:
1. 只能从上面列出的标签中选择，不能自创标签
2. 单选分类输出字符串，多选分类输出字符串数组
3. 尽量每个分类都输出标签，只有如果某个分类在图片中不明显，输出空字符串或空数组
4. 严格按照 JSON 格式输出，不要输出其他任何内容
5. 不要使用 markdown 代码块

输出格式示例:
{example_str}

现在请分析这张图片并输出标签 JSON:"""

    return prompt


def parse_model_output(text):
    """解析模型输出的 JSON，兼容 Qwen3.5 thinking 模式"""
    if not text:
        return None

    # Qwen3.5 thinking 模式: 去掉 <thinkvi>...</thinkvi> 思考块
    # 模型输出格式: <thinkvi>思考内容</thinkvi>\n\n实际JSON
    think_end = text.find("</thinkvi>")
    if think_end != -1:
        text = text[think_end + len("</thinkvi>"):]

    # 也处理纯文本 thinking 标记 (有些版本用不同格式)
    # 查找最后一个 JSON 对象
    text = text.strip()

    # 清理 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # 方法1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 方法2: 查找最外层 { } JSON 对象 (正确处理嵌套)
    brace_count = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0 and start >= 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1

    # 方法3: 正则匹配 (非贪婪，取最后一个)
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    logger.warning(f"Failed to parse model output: {text[:300]}")
    return None


def normalize_labels(raw_labels, config):
    """将模型输出标准化为 label_config 格式"""
    if raw_labels is None:
        return {}
    result = {}
    for cat, info in config.items():
        val = raw_labels.get(cat, None)
        if val is None:
            val = [] if info.get("multi", False) else ""

        if info.get("multi", False):
            if isinstance(val, str):
                val = [val] if val else []
            elif not isinstance(val, list):
                val = []
            valid_labels = info["labels"]
            val = [v for v in val if v in valid_labels]
        else:
            if isinstance(val, list):
                val = val[0] if val else ""
            if val and val not in info["labels"]:
                val = ""

        result[cat] = val
    return result


# ============================================================
# 本地模型管理器
# ============================================================

class LocalVLM:
    """本地多模态视觉语言模型管理器"""

    def __init__(self, model_path, device="cuda", dtype="bfloat16",
                 temperature=0.9, label_temperature=0.5, top_p=0.9, top_k=50, do_sample=True):
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.label_temperature = label_temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.model = None
        self.processor = None
        self.config = load_label_config()
        self.prompt = build_prompt(self.config)

    def load(self):
        """加载模型和处理器"""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info(f"Loading model from {self.model_path} ...")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        logger.info(f"  dtype={self.dtype}, device={self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # 关键: 必须用 AutoModelForImageTextToText 而非 AutoModelForCausalLM
        # AutoModelForCausalLM 会加载纯文本模型，不接受 pixel_values 等图像输入
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else "cpu",
            trust_remote_code=True,
        )

        self.model.eval()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            logger.info(f"Model loaded. GPU memory: {allocated:.1f} GB")

        logger.info("Model ready for inference.")

    def is_loaded(self):
        return self.model is not None

    def unload(self):
        """释放模型和处理器，清理显存"""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory cleared.")

    def _prepare_image(self, image_path):
        """加载并预处理图片 (从文件路径)"""
        from PIL import Image
        img = Image.open(image_path)
        return self._normalize_image(img)

    def _normalize_image(self, img):
        """归一化 PIL Image: RGBA->RGB、其他模式->RGB、限制最大边 1024px"""
        from PIL import Image
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        return img

    def _build_inputs(self, img):
        """构建模型输入 tensor"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]

        # 应用聊天模板 (禁用思考模式以加速推理)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 提取图片列表
        image_inputs = [img]
        video_inputs = []

        # 构建 processor 输入
        # 关键: 只有 videos 非空时才传 videos 参数，否则会触发 IndexError
        processor_kwargs = dict(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        if video_inputs:
            processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs).to(self.model.device)
        return inputs

    def label_image(self, image_path):
        """
        对单张图片进行自动标注

        Args:
            image_path: 图片文件路径

        Returns:
            dict: 标注结果 (符合 label_config.json 格式)
        """
        import torch

        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load()")

        # 每次标注时重新加载配置，使 enabled 开关立即生效
        self.config = load_label_config()
        self.prompt = build_prompt(self.config)

        img = self._prepare_image(image_path)
        inputs = self._build_inputs(img)

        # 推理
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=self.do_sample,
                temperature=self.label_temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        # 解码输出 (只取新生成的 token)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        decoded_list = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # 显式释放 GPU 内存
        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

        if not decoded_list:
            logger.warning(f"Model returned empty output for {image_path}")
            return {}

        output_text = decoded_list[0].strip()
        logger.info(f"Model output: {output_text[:200]}")

        # 解析标签
        raw_labels = parse_model_output(output_text)
        if raw_labels is None:
            logger.warning(f"Failed to parse output for {image_path}")
            return {}

        labels = normalize_labels(raw_labels, self.config)
        return labels

    def generate_text(self, image_path, custom_prompt, max_new_tokens=4096, enable_thinking=False, pil_image=None):
        """
        使用自定义 prompt 生成文本描述

        Args:
            image_path: 图片文件路径 (pil_image 为 None 时使用)
            custom_prompt: 自定义提示词
            max_new_tokens: 最大生成 token 数
            enable_thinking: 是否启用思考模式
            pil_image: 可选 PIL Image，提供则直接使用而非从 image_path 加载 (用于选区裁剪)

        Returns:
            str: 模型生成的文本
        """
        import torch

        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load()")

        img = self._normalize_image(pil_image) if pil_image is not None else self._prepare_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": custom_prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        image_inputs = [img]
        processor_kwargs = dict(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = self.processor(**processor_kwargs).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        decoded_list = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

        if not decoded_list:
            logger.warning(f"Model returned empty output for {image_path}")
            return ""

        return decoded_list[0].strip()

    def generate_text_only(self, prompt, max_new_tokens=512, temperature=None):
        """纯文本生成（不需要图片），用于标签转换等文本任务"""
        import torch

        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load()")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        processor_kwargs = dict(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = self.processor(**processor_kwargs).to(self.model.device)

        gen_temp = temperature if temperature is not None else self.label_temperature

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=gen_temp,
                top_p=0.9,
                top_k=50,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        decoded_list = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

        if not decoded_list:
            return ""
        return decoded_list[0].strip()


# ============================================================
# 命令行批量标注
# ============================================================

def batch_label(model_path, image_dir, output_file=None, batch_size=None,
                temperature=0.7, top_p=0.9, top_k=50, do_sample=True):
    """
    命令行批量标注入口

    Args:
        model_path: 模型路径
        image_dir: 图片目录
        output_file: 标注输出文件路径
        batch_size: 处理数量限制
    """
    import torch

    image_dir = Path(image_dir)
    output_file = Path(output_file) if output_file else image_dir / "annotations.json"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将使用 CPU 推理 (速度较慢)")

    vlm = LocalVLM(model_path, temperature=temperature, top_p=top_p,
                   top_k=top_k, do_sample=do_sample)
    vlm.load()

    files = sorted([
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    annotations = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        logger.info(f"Loaded {len(annotations)} existing annotations")

    todo = [f for f in files if f.name not in annotations]
    if batch_size:
        todo = todo[:batch_size]

    logger.info(f"Images to label: {len(todo)} / {len(files)} total")

    for i, filepath in enumerate(todo):
        logger.info(f"[{i + 1}/{len(todo)}] Labeling: {filepath.name}")
        try:
            labels = vlm.label_image(str(filepath))
        except Exception as e:
            logger.error(f"Error labeling {filepath.name}: {e}")
            labels = {}

        from datetime import datetime
        annotations[filepath.name] = {
            "labels": labels,
            "custom_tags": [],
            "auto_labeled": True,
            "verified": False,
            "updated_at": datetime.now().isoformat(),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Done! Annotations saved to {output_file}")
    return annotations


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="本地 VLM 批量标注")
    parser.add_argument("--model", default="F:/qwen3_5", help="模型路径")
    parser.add_argument("--image-dir", default=".", help="图片目录")
    parser.add_argument("--output", default=None, help="标注输出文件")
    parser.add_argument("--batch-size", type=int, default=None, help="标注数量限制")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度 (0.0-1.0, 越高越有创造性)")
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样概率阈值")
    parser.add_argument("--top-k", type=int, default=20, help="top-k 采样参数")
    parser.add_argument("--no-sample", action="store_true", help="禁用采样 (使用贪婪解码)")
    args = parser.parse_args()

    do_sample = not args.no_sample
    batch_label(args.model, args.image_dir, args.output, args.batch_size,
                temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, do_sample=do_sample)
