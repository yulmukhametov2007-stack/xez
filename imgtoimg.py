import spaces
import os
import gc
import gradio as gr
import torch
import json
import utils
import logging
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
# --- ДОБАВЛЕН ИМПОРТ COMPEL ---
from compel import Compel, ReturnedEmbeddingsType
# ------------------------------

# --- ДОБАВЛЕНЫ ИМПОРТЫ ДЛЯ GOOGLE DRIVE (WEB APP) ---
import requests
import base64
import re
import random
from PIL import Image as PILImage
# ----------------------------------------------------

from config import (
    MIN_IMAGE_SIZE,
    MAX_IMAGE_SIZE,
    OUTPUT_DIR,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_ASPECT_RATIO,
    QUALITY_TAGS,
    sampler_list,
    aspect_ratios,
    css as config_css 
)
import time
from typing import List, Dict, Tuple

# Улучшенное логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Константы
IS_COLAB = utils.is_google_colab() or os.getenv("IS_COLAB") == "1"
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1" 

# Настройки PyTorch для производительности
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- НАСТРОЙКИ GOOGLE DRIVE ---
WEBAPP_URL = "https://script.google.com/macros/s/AKfycbzX3vYzBBR8bvJ8csbqhPg2ykV-e3hbe-13Nvw9e0SJal-Zg0YLowYP6vRck9rMMtca/exec"

def upload_to_gdrive(file_path: str, metadata: dict) -> str:
    """
    Отправляет файл в Google Drive через Web App.
    """
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        description_text = (
            f"Prompt: {metadata.get('prompt')}\n"
            f"Negative: {metadata.get('negative_prompt')}\n"
            f"Seed: {metadata.get('seed')}\n"
            f"Strength: {metadata.get('strength')}"
        )

        payload = {
            "fileName": os.path.basename(file_path),
            "fileData": encoded_string,
            "description": description_text
        }

        response = requests.post(WEBAPP_URL, json=payload)
        
        if response.status_code == 200:
            resp_data = response.json()
            if resp_data.get("status") == "success":
                logger.info(f"Изображение успешно загружено в Google Drive! ID: {resp_data.get('fileId')}")
                return resp_data.get("fileId")
            else:
                logger.error(f"Ошибка скрипта Google: {resp_data.get('message')}")
        else:
            logger.error(f"Ошибка сети при загрузке: HTTP {response.status_code}")
            
        return None
        
    except Exception as e:
        logger.error(f"Локальная ошибка при отправке в Drive: {e}")
        return None
# ------------------------------

def save_result_to_gdrive(image_path, metadata):
    """Сохраняет текущую картинку в Google Drive по кнопке."""
    if not image_path:
        raise gr.Error("Сначала сгенерируйте изображение")
    file_id = upload_to_gdrive(image_path, metadata or {})
    if file_id:
        return "✅ Сохранено в Google Drive"
    return "❌ Не удалось сохранить — проверьте соединение и попробуйте ещё раз"

# --- ПЕРЕВОД РУССКОГО ОПИСАНИЯ В ПРОМПТ ---
def translate_ru_to_en(text: str) -> str:
    """Переводит текст с русского на английский через бесплатный эндпоинт Google Translate."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "ru", "tl": "en", "dt": "t", "q": text},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
        return translated.strip()
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
        raise gr.Error("Не удалось перевести текст. Проверьте интернет-соединение и попробуйте ещё раз.")

def ru_prompt_to_en(ru_prompt: str):
    """Переводит описание с русского в английский промпт."""
    if not ru_prompt or ru_prompt.isspace():
        raise gr.Error("Введите описание на русском")
    return translate_ru_to_en(ru_prompt).rstrip(" ,.")

def send_prompt_to_generation(en_prompt: str):
    if not en_prompt or en_prompt.isspace():
        raise gr.Error("Сначала переведите описание")
    return gr.update(value=en_prompt)
# ------------------------------------------

class GenerationError(Exception):
    pass

def validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise GenerationError("Промпт должен быть строкой")
    try:
        prompt = prompt.encode('utf-8').decode('utf-8')
        prompt = prompt.replace("!,", "! ,")
    except UnicodeError:
        raise GenerationError("Недопустимые символы в промпте")
    
    if not prompt or prompt.isspace():
        raise GenerationError("Промпт не может быть пустым")
    return prompt.strip()

def validate_dimensions(width: int, height: int) -> None:
    if not MIN_IMAGE_SIZE <= width <= MAX_IMAGE_SIZE:
        raise GenerationError(f"Ширина должна быть между {MIN_IMAGE_SIZE} и {MAX_IMAGE_SIZE}")
        
    if not MIN_IMAGE_SIZE <= height <= MAX_IMAGE_SIZE:
        raise GenerationError(f"Высота должна быть между {MIN_IMAGE_SIZE} и {MAX_IMAGE_SIZE}")

def fit_image_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Масштабирует размеры исходной картинки в пределы [MIN_IMAGE_SIZE, MAX_IMAGE_SIZE] с сохранением пропорций."""
    scale = 1.0
    if max(width, height) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / float(max(width, height))
    elif min(width, height) < MIN_IMAGE_SIZE:
        scale = MIN_IMAGE_SIZE / float(min(width, height))
        
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    new_w = max(MIN_IMAGE_SIZE, min(MAX_IMAGE_SIZE, new_w))
    new_h = max(MIN_IMAGE_SIZE, min(MAX_IMAGE_SIZE, new_h))
    return utils.preprocess_image_dimensions(new_w, new_h)

@spaces.GPU
def generate(
    init_image_path: str,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    strength: float = 0.7,
    use_init_size: bool = True,
    seed: int = 0,
    custom_width: int = 1024,
    custom_height: int = 1024,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 25,
    sampler: str = "Euler a",
    model_name: str = "Heartsync",
    aspect_ratio_selector: str = DEFAULT_ASPECT_RATIO,
    add_quality_tags: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, Dict]:
    start_time = time.time()
    backup_scheduler = None
    pipe = None
    
    try:
        torch.cuda.empty_cache()
        gc.collect()

        if not init_image_path:
            raise GenerationError("Загрузите исходное изображение для Img2Img")

        prompt = validate_prompt(prompt)
        if negative_prompt:
            negative_prompt = negative_prompt.encode('utf-8').decode('utf-8')
        
        init_img = PILImage.open(init_image_path).convert("RGB")
        
        if use_init_size:
            width, height = fit_image_dimensions(*init_img.size)
        else:
            validate_dimensions(custom_width, custom_height)
            width, height = utils.aspect_ratio_handler(
                aspect_ratio_selector,
                custom_width,
                custom_height,
            )
            width, height = utils.preprocess_image_dimensions(width, height)

        # Изменяем размер входного изображения под целевое разрешение
        resample_filter = getattr(PILImage, "Resampling", PILImage).LANCZOS
        init_img = init_img.resize((width, height), resample_filter)
        
        generator = utils.seed_everything(seed)

        if add_quality_tags:
            prompt = QUALITY_TAGS.format(prompt=prompt)

        prompt, negative_prompt = utils.preprocess_prompt(
            prompt, negative_prompt
        )    

        pipe = pipes.get(model_name)
        compel_proc = pipes.get("compel")
        
        if pipe is None or compel_proc is None:
            raise GenerationError(f"Модель {model_name} или модуль Compel не загружены")
            
        backup_scheduler = pipe.scheduler
        pipe.scheduler = utils.get_scheduler(pipe.scheduler.config, sampler)

        # Гарантируем, что num_inference_steps * strength >= 1
        effective_steps = max(num_inference_steps, int(1.0 / max(strength, 0.01)) + 1)
            
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": f"{width} x {height}",
            "guidance_scale": guidance_scale,
            "num_inference_steps": effective_steps,
            "strength": strength,
            "seed": seed,
            "sampler": sampler,
            "Model": "Heartsync/NSFW-Uncensored (Img2Img)",
        }
        
        # --- ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ЧЕРЕЗ COMPEL С ПОДДЕРЖКОЙ ВЕСОВ ---
        prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel_proc(negative_prompt if negative_prompt else "")
        
        [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
        # -------------------------------------------------------------
        
        images = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=init_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=effective_steps,
            generator=generator,
            output_type="pil",
        ).images

        image_paths = []
        if images:
            total = len(images)
            for idx, image in enumerate(images, 1):
                progress(idx/total, desc="Сохранение локально...")
                path = utils.save_image(image, metadata, OUTPUT_DIR, IS_COLAB)
                image_paths.append(path)
                
        metadata["generation_time"] = f"{time.time() - start_time:.2f}s"
        return image_paths[0] if image_paths else None, metadata

    except Exception as e:
        raise gr.Error(f"Ошибка генерации: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        if backup_scheduler is not None and pipe is not None:
            pipe.scheduler = backup_scheduler
        utils.free_memory()

# ------------------------------------------------------------
# ЗАГРУЗКА МОДЕЛИ HEARTSYNC (IMG2IMG) И COMPEL
# ------------------------------------------------------------
pipes = {}
logger.info("Запуск загрузки модели Heartsync/NSFW-Uncensored (Img2Img)...")
try:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "Heartsync/NSFW-Uncensored",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    if torch.cuda.is_available():
        for sub in (pipe.text_encoder, pipe.text_encoder_2, pipe.vae, pipe.unet):
            sub.to(torch.float16)
            
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
            
    pipes["Heartsync"] = pipe
    
    # --- ИНИЦИАЛИЗАЦИЯ COMPEL ДЛЯ SDXL ---
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )
    pipes["compel"] = compel
    # -------------------------------------
    
    logger.info("Модель Img2Img и парсер весов Compel успешно загружены!")
except Exception as e:
    logger.error(f"Критическая ошибка загрузки модели: {e}")

def update_history(new_images, metadata, current_history_data):
    if current_history_data is None:
        current_history_data = []

    if new_images:
        if isinstance(new_images, str):
            new_images = [new_images]

        entry_meta = {
            "prompt": str(metadata.get("prompt", "")),
            "negative_prompt": str(metadata.get("negative_prompt", "")),
            "seed": str(metadata.get("seed", "")),
            "steps": str(metadata.get("num_inference_steps", "")),
            "cfg": str(metadata.get("guidance_scale", "")),
            "strength": str(metadata.get("strength", "")),
        }

        new_entries = []
        for img_item in new_images:
            img_path = img_item[0] if isinstance(img_item, tuple) else img_item.get("image", img_item) if isinstance(img_item, dict) else img_item.path if hasattr(img_item, "path") else img_item
            new_entries.append({"image": img_path, "meta": entry_meta})

        current_history_data = new_entries + current_history_data

    gallery_images = [item["image"] for item in current_history_data]
    return current_history_data, gallery_images


# ------------------------------------------------------------
# ЧТЕНИЕ МЕТАДАННЫХ ИЗ PNG (вкладка "Метаданные")
# ------------------------------------------------------------
def _extract_from_json(data: dict) -> Tuple[str, str, str, str, str]:
    """Достаёт нужные поля из словаря метаданных."""
    def _get(*keys):
        for key in keys:
            if key in data and data[key] is not None:
                return str(data[key])
        return ""

    return (
        _get("prompt", "Prompt"),
        _get("negative_prompt", "Negative prompt", "negativePrompt"),
        _get("seed", "Seed"),
        _get("num_inference_steps", "steps", "Steps"),
        _get("guidance_scale", "cfg_scale", "CFG scale"),
    )


def _parse_a1111_parameters(raw: str) -> Tuple[str, str, str, str, str]:
    """Парсит строку параметров в формате Automatic1111."""
    prompt_text = raw
    negative_text = ""
    params_block = ""

    match = re.search(r"(?:^|\n)Steps:", raw)
    if match:
        params_block = raw[match.start():].strip()
        prompt_text = raw[:match.start()]

    if "Negative prompt:" in prompt_text:
        prompt_text, negative_text = prompt_text.split("Negative prompt:", 1)

    def _find(key: str) -> str:
        m = re.search(key + r":\s*([^,\n]+)", params_block)
        return m.group(1).strip() if m else ""

    return (
        prompt_text.strip(),
        negative_text.strip(),
        _find("Seed"),
        _find("Steps"),
        _find("CFG scale"),
    )


def read_image_metadata(image_path):
    """Извлекает промпт, негативный промпт, сид, шаги и CFG из PNG."""
    empty = ("", "", "", "", "")
    if not image_path:
        return empty

    try:
        with PILImage.open(image_path) as img:
            merged = {}
            merged.update(getattr(img, "text", {}) or {})
            merged.update(img.info or {})
            info = {k: v for k, v in merged.items() if isinstance(v, str)}
    except Exception as e:
        raise gr.Error(f"Не удалось открыть изображение: {e}")

    ordered_keys = ["metadata", "parameters"] + [
        k for k in info if k not in ("metadata", "parameters")
    ]
    for key in ordered_keys:
        raw = (info.get(key) or "").strip()
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return _extract_from_json(data)

    raw_params = info.get("parameters")
    if raw_params:
        return _parse_a1111_parameters(raw_params)

    raise gr.Error("Метаданные не найдены. Убедитесь, что это оригинальный PNG-файл генерации (без пересохранения или сжатия мессенджерами).")


# Интеграция CSS
fixed_css = config_css + """

/* ============================================================
   HEARTSYNC UI THEME v2 — «Aurora Glass»
   Палитра: фиолетово-синий градиент (#c136eb → #4EACEF)
   ============================================================ */

:root {
  --accent-1: #c136eb;
  --accent-2: #4EACEF;
  --gradient-primary: linear-gradient(120deg, #c136eb 0%, #7f6cf0 50%, #4EACEF 100%);
  --glass-bg: rgba(19, 21, 31, 0.55);
  --glass-border: rgba(255, 255, 255, 0.09);
  --box-shadow-custom: 0 8px 32px rgba(0, 0, 0, 0.45);
  --text-dim: rgba(255, 255, 255, 0.55);
}

body, .gradio-container, .wrap {
    background:
      radial-gradient(ellipse 55% 40% at 12% -5%, rgba(193, 54, 235, 0.18) 0%, transparent 60%),
      radial-gradient(ellipse 50% 35% at 88% 5%, rgba(78, 172, 239, 0.15) 0%, transparent 60%),
      radial-gradient(circle at 50% 0%, #241645 0%, #0d0f18 55%, #050508 100%) !important;
    background-attachment: fixed !important;
}

::selection { background: rgba(193, 54, 235, 0.4); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.14); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }

.main-card {
    width: 100% !important;
    max-width: 1000px !important;
    margin: 2.5rem auto !important;
    padding: 0 1rem !important;
    display: flex !important;
    flex-direction: column !important;
    align-self: center !important;
}

.custom-title {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0 0 0.35rem 0;
    background-image: var(--gradient-primary);
    -webkit-text-fill-color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    display: inline-block;
    filter: drop-shadow(0 0 18px rgba(150, 90, 240, 0.35));
}

.custom-subtitle {
    font-size: 0.95em;
    color: var(--text-dim);
    letter-spacing: 0.02em;
}
.custom-subtitle a { transition: color 0.2s ease; }
.custom-subtitle a:hover { color: var(--accent-2) !important; }

.custom-header-divider {
    width: 120px;
    height: 3px;
    margin: 0.9rem auto 0.2rem auto;
    border-radius: 3px;
    background: var(--gradient-primary);
    opacity: 0.85;
}

.panel, div[class*="panel"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) saturate(1.15) !important;
    -webkit-backdrop-filter: blur(14px) saturate(1.15) !important;
    border: 1px solid var(--glass-border) !important;
    border-top-color: rgba(255, 255, 255, 0.16) !important;
    border-radius: 16px !important;
    box-shadow: var(--box-shadow-custom) !important;
    padding: 1.5rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}

.panel:hover, div[class*="panel"]:hover {
    border-color: rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55) !important;
}

.main-card h3 {
    display: flex !important;
    align-items: center !important;
    gap: 0.6rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: rgba(255, 255, 255, 0.85) !important;
    margin: 1.4rem 0 0.6rem 0.2rem !important;
}
.main-card h3::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 1.1em;
    border-radius: 4px;
    background: var(--gradient-primary);
    flex: none;
}

button.primary {
    background-image: var(--gradient-primary) !important;
    background-size: 160% 160% !important;
    background-position: 0% 50% !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(150, 90, 240, 0.35) !important;
    transition: background-position 0.3s ease, box-shadow 0.25s ease, transform 0.2s ease !important;
}

button.primary:hover {
    background-position: 100% 50% !important;
    box-shadow: 0 6px 26px rgba(150, 90, 240, 0.55) !important;
    transform: translateY(-2px) !important;
}

button.primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 3px 12px rgba(150, 90, 240, 0.4) !important;
}

button.primary:disabled {
    opacity: 0.6 !important;
    transform: none !important;
    box-shadow: none !important;
}

button.secondary {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
button.secondary:hover {
    border-color: rgba(193, 54, 235, 0.5) !important;
    background: rgba(193, 54, 235, 0.08) !important;
}

textarea, input[type="text"], input[type="number"], .gr-box {
    background: rgba(8, 10, 16, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    resize: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

textarea::placeholder, input::placeholder {
    color: rgba(255, 255, 255, 0.35) !important;
}

textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: rgba(193, 54, 235, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(193, 54, 235, 0.18) !important;
    outline: none !important;
}

input[type="range"] { accent-color: #a55cf0 !important; }
input[type="checkbox"], input[type="radio"] { accent-color: #a55cf0 !important; }

#result-image, #init-image {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}
#result-image > div, #init-image > div {
    border-radius: 16px !important;
    overflow: hidden !important;
}
#result-image img, #init-image img {
    width: 100% !important;
    height: auto !important;
    max-height: 85vh !important;
    object-fit: cover !important;
}

#history-gallery .thumbnail-item {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}
#history-gallery .thumbnail-item:hover {
    transform: translateY(-3px) !important;
    border-color: rgba(193, 54, 235, 0.55) !important;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.5) !important;
}

#history-gallery button .caption-label,
#history-gallery button figcaption,
#history-gallery button .caption {
    display: none !important;
}

dialog .thumbnails, dialog button[aria-label^="Thumbnail"], dialog [data-testid="thumbnail-container"] {
    display: none !important;
}

dialog .image-container, dialog .wrapper {
    max-height: 75vh !important;
}

dialog .caption-label,
dialog figcaption,
dialog .caption {
    display: none !important;
}

dialog .thumbnail-item,
dialog .thumbnail-small,
dialog .svelte-7anmrz {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
}

.caption, figcaption, div[class*="caption"], span[class*="caption"] {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-y: auto !important;
    max-height: 150px !important;
    line-height: 1.35 !important;
    font-size: 0.9em !important;
    padding-bottom: 10px !important;
    padding-top: 10px !important;
}

.accordion {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) saturate(1.15) !important;
    -webkit-backdrop-filter: blur(14px) saturate(1.15) !important;
    border: 1px solid var(--glass-border) !important;
    border-top-color: rgba(255, 255, 255, 0.16) !important;
    border-radius: 16px !important;
    box-shadow: var(--box-shadow-custom) !important;
    padding: 1rem 1.5rem !important;
    margin-top: 1rem !important;
}

.accordion > button, .accordion .label-wrap {
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: rgba(255, 255, 255, 0.85) !important;
    font-size: 0.95rem !important;
}

.accordion .label-wrap:hover {
    color: #ffffff !important;
}

.tabs, .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.tab-nav, .tab-container, div[role="tablist"] {
    display: flex !important;
    justify-content: center !important;
    gap: 10px !important;
    border-bottom: none !important;
    background: transparent !important;
    margin: 1.5rem auto 0 auto !important;
}

button[role="tab"], .tab-nav button {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.04) !important;
    color: rgba(255, 255, 255, 0.65) !important;
    padding: 9px 20px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease !important;
}

button[role="tab"]:hover, .tab-nav button:hover {
    border-color: rgba(193, 54, 235, 0.5) !important;
    color: #ffffff !important;
}

button[role="tab"][aria-selected="true"],
button[role="tab"].selected,
.tab-nav button.selected {
    background: linear-gradient(120deg, rgba(193, 54, 235, 0.28), rgba(78, 172, 239, 0.28)) !important;
    border-color: rgba(193, 54, 235, 0.6) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 14px rgba(150, 90, 240, 0.25) !important;
}

#theme-btn { display: none !important; }
"""

with gr.Blocks() as demo:
    history_data_state = gr.State([])
    hidden_model_name = gr.State("Heartsync") 
    hidden_metadata = gr.State({}) 

    with gr.Tabs(elem_classes="main-tabs"):
        with gr.Tab("🎨 Генерация (Img2Img)"):
            with gr.Column(elem_classes="main-card"):
        
                with gr.Row(variant="panel"):
                    gr.HTML(
                        """
                        <div style="text-align: center; width: 100%;">
                            <h1 class="custom-title">Heartsync NSFW-Uncensored SDXL (Img2Img)</h1>
                            <div class="custom-subtitle">Интерфейс Img2Img-генерации на базе <a href="https://huggingface.com/Heartsync/NSFW-Uncensored" target="_blank" style="color: inherit; text-decoration: underline;">Heartsync SDXL</a></div>
                            <div class="custom-header-divider"></div>
                        </div>
                        """
                    )

                with gr.Row(variant="panel", equal_height=True):
                    with gr.Column(scale=4):
                        prompt = gr.Textbox(
                            label="Промпт",
                            lines=4,
                            placeholder="Опишите, во что превратить изображение... (Используйте скобки для весов, например: (слово:1.5))",
                            show_label=True,
                            container=True,
                        )
                    with gr.Column(scale=1, min_width=150):
                        run_button = gr.Button("Generate", variant="primary", size="lg")
                        strength = gr.Slider(
                            label="Сила изменений (Strength)",
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.70,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        init_image = gr.Image(
                            label="Исходное изображение",
                            type="filepath",
                            sources=["upload", "clipboard"],
                            show_label=True,
                            elem_classes="panel",
                            elem_id="init-image"
                        )
                    with gr.Column(scale=1):
                        result = gr.Image(
                            label="Сгенерированное изображение",
                            type="filepath",
                            show_label=True,
                            elem_classes="panel",
                            elem_id="result-image"
                        )

                save_gdrive_button = gr.Button("💾 Сохранить в Google Drive", size="lg")
                gdrive_status = gr.Markdown("")

                with gr.Accordion("⚙️ Расширенные настройки", open=True):
                    negative_prompt = gr.Textbox(
                        label="Негативный промпт",
                        lines=2,
                        placeholder="Опишите, чего не должно быть на изображении... (Можно использовать (слово:1.5))",
                        value=DEFAULT_NEGATIVE_PROMPT,
                    )

                    use_init_size = gr.Checkbox(
                        label="Сохранять пропорции и размер исходного изображения",
                        value=True,
                    )

                    with gr.Group(visible=False) as aspect_ratio_group:
                        aspect_ratio_selector = gr.Radio(
                            label="Соотношение сторон",
                            choices=aspect_ratios,
                            value=DEFAULT_ASPECT_RATIO,
                            container=True,
                        )
                
                        with gr.Group(visible=False) as custom_resolution:
                            with gr.Row():
                                custom_width = gr.Slider(label="Ширина", minimum=MIN_IMAGE_SIZE, maximum=MAX_IMAGE_SIZE, step=8, value=1024)
                                custom_height = gr.Slider(label="Высота", minimum=MIN_IMAGE_SIZE, maximum=MAX_IMAGE_SIZE, step=8, value=1024)
            
                    with gr.Row():
                        with gr.Column(scale=1):
                            sampler = gr.Dropdown(label="Сэмплер (Sampler)", choices=sampler_list, value="Euler a", interactive=True)
                            seed = gr.Slider(label="Сид (Seed)", minimum=0, maximum=utils.MAX_SEED, step=1, value=0)
                            randomize_seed = gr.Checkbox(label="Случайный сид", value=True)
                
                        with gr.Column(scale=1):
                            with gr.Group():
                                guidance_scale = gr.Slider(label="Шкала соответствия (CFG Scale)", minimum=1, maximum=12, step=0.1, value=5.0)
                                num_inference_steps = gr.Slider(label="Количество шагов (Steps)", minimum=1, maximum=50, step=1, value=25)
                            add_quality_tags = gr.Checkbox(label="Авто-теги качества", value=True)

                gr.Markdown("### 🕰️ История")
                with gr.Column(variant="panel"):
                    history_gallery = gr.Gallery(
                        label="История генераций",
                        columns=4,
                        height='auto',
                        show_label=False,
                        object_fit="contain",
                        elem_id="history-gallery"
                    )

                    hist_prompt = gr.Textbox(
                        label="Промпт",
                        lines=4,
                        interactive=False,
                    )
                    hist_negative_prompt = gr.Textbox(
                        label="Негативный промпт",
                        lines=3,
                        interactive=False,
                    )
                    with gr.Row():
                        hist_seed = gr.Textbox(label="Сид (Seed)", interactive=False)
                        hist_steps = gr.Textbox(label="Шаги (Steps)", interactive=False)
                        hist_cfg = gr.Textbox(label="CFG Scale", interactive=False)
                        hist_strength = gr.Textbox(label="Strength", interactive=False)

        with gr.Tab("🔍 Метаданные"):
            with gr.Column(elem_classes="main-card"):

                with gr.Row(variant="panel"):
                    gr.HTML(
                        """
                        <div style="text-align: center; width: 100%;">
                            <h2 class="custom-title" style="font-size: clamp(1.2rem, 3vw, 1.7rem);">Просмотр метаданных</h2>
                            <div class="custom-subtitle">Загрузите PNG генерации, чтобы увидеть промпт, сид, шаги и CFG</div>
                            <div class="custom-header-divider"></div>
                        </div>
                        """
                    )

                with gr.Column(variant="panel"):
                    meta_image_input = gr.Image(
                        label="Изображение (PNG)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=350,
                    )
                    meta_prompt = gr.Textbox(
                        label="Промпт",
                        lines=4,
                        interactive=False,
                    )
                    meta_negative_prompt = gr.Textbox(
                        label="Негативный промпт",
                        lines=3,
                        interactive=False,
                    )
                    with gr.Row():
                        meta_seed = gr.Textbox(label="Сид (Seed)", interactive=False)
                        meta_steps = gr.Textbox(label="Шаги (Steps)", interactive=False)
                        meta_cfg = gr.Textbox(label="CFG Scale", interactive=False)

        with gr.Tab("🇷🇺 Промпт на русском"):
            with gr.Column(elem_classes="main-card"):

                with gr.Row(variant="panel"):
                    gr.HTML(
                        """
                        <div style="text-align: center; width: 100%;">
                            <h2 class="custom-title" style="font-size: clamp(1.2rem, 3vw, 1.7rem);">Промпт на русском</h2>
                            <div class="custom-subtitle">Опишите картинку по-русски — получите готовый английский промпт</div>
                            <div class="custom-header-divider"></div>
                        </div>
                        """
                    )

                with gr.Column(variant="panel"):
                    ru_prompt_input = gr.Textbox(
                        label="Описание на русском",
                        lines=4,
                        placeholder="Например: девушка с длинными волосами в белой рубашке, закат, у озера, мягкий свет...",
                    )
                    translate_button = gr.Button("Перевести в промпт", variant="primary", size="lg")
                    en_prompt_output = gr.Textbox(
                        label="Промпт (английский) — можно отредактировать",
                        lines=4,
                        interactive=True,
                    )
                    send_to_gen_button = gr.Button("Использовать в генерации →", size="lg")
                    gr.Markdown("*После нажатия перейдите во вкладку «🎨 Генерация (Img2Img)» — промпт уже будет подставлен.*")

    translate_button.click(
        fn=ru_prompt_to_en,
        inputs=[ru_prompt_input],
        outputs=[en_prompt_output],
    )
    ru_prompt_input.submit(
        fn=ru_prompt_to_en,
        inputs=[ru_prompt_input],
        outputs=[en_prompt_output],
    )
    send_to_gen_button.click(
        fn=send_prompt_to_generation,
        inputs=[en_prompt_output],
        outputs=[prompt],
    )
    save_gdrive_button.click(
        fn=save_result_to_gdrive,
        inputs=[result, hidden_metadata],
        outputs=[gdrive_status],
    )

    use_init_size.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=use_init_size,
        outputs=aspect_ratio_group,
        queue=False,
        api_name=False,
    )

    aspect_ratio_selector.change(
        fn=lambda x: gr.update(visible=x == "Custom"),
        inputs=aspect_ratio_selector,
        outputs=custom_resolution,
        queue=False,
        api_name=False,
    )

    def precheck_inputs(p, img):
        if not img:
            raise gr.Error("Загрузите исходное изображение для Img2Img!")
        if not p or p.isspace():
            raise gr.Error("Промпт не может быть пустым! Введите текст для генерации.")

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=utils.randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=precheck_inputs, 
        inputs=[prompt, init_image],
        outputs=[],
        queue=False,
    ).then(
        fn=lambda: gr.update(interactive=False, value="Загрузка..."), 
        outputs=run_button,
    ).then(
        fn=generate,
        inputs=[
            init_image,
            prompt,
            negative_prompt,
            strength,
            use_init_size,
            seed,
            custom_width,
            custom_height,
            guidance_scale,
            num_inference_steps,
            sampler,
            hidden_model_name,
            aspect_ratio_selector,
            add_quality_tags,
        ],
        outputs=[result, hidden_metadata], 
    ).then(
        fn=update_history,
        inputs=[result, hidden_metadata, history_data_state],
        outputs=[history_data_state, history_gallery],
    ).then(
        fn=lambda: gr.update(interactive=True, value="Generate"),
        outputs=run_button,
    ).then(
        fn=lambda: "",
        outputs=gdrive_status,
        queue=False,
    )

    def on_history_select(evt: gr.SelectData, history_list):
        """При клике на картинку в истории заполняет поля."""
        if history_list and evt.index < len(history_list):
            m = history_list[evt.index].get("meta", {})
            return (
                m.get("prompt", ""),
                m.get("negative_prompt", ""),
                m.get("seed", ""),
                m.get("steps", ""),
                m.get("cfg", ""),
                m.get("strength", ""),
            )
        return "", "", "", "", "", ""

    history_gallery.select(
        fn=on_history_select,
        inputs=[history_data_state],
        outputs=[hist_prompt, hist_negative_prompt, hist_seed, hist_steps, hist_cfg, hist_strength],
        queue=False,
        api_name=False,
    )

    meta_image_input.change(
        fn=read_image_metadata,
        inputs=[meta_image_input],
        outputs=[meta_prompt, meta_negative_prompt, meta_seed, meta_steps, meta_cfg],
        queue=False,
        api_name=False,
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        debug=IS_COLAB, 
        share=IS_COLAB,
        css=fixed_css,
        theme="Nymbo/Nymbo_Theme_5"
    )