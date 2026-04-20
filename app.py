import spaces
import os
import gc
import gradio as gr
import torch
import json
import utils
import logging
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
# --- ДОБАВЛЕН ИМПОРТ COMPEL ---
from compel import Compel, ReturnedEmbeddingsType
# ------------------------------
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

@spaces.GPU
def generate(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 0,
    custom_width: int = 1024,
    custom_height: int = 1024,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 20,
    sampler: str = "Euler a",
    model_name: str = "Heartsync",
    aspect_ratio_selector: str = DEFAULT_ASPECT_RATIO,
    add_quality_tags: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[List[str], Dict]:
    start_time = time.time()
    backup_scheduler = None
    pipe = None
    
    try:
        torch.cuda.empty_cache()
        gc.collect()

        prompt = validate_prompt(prompt)
        if negative_prompt:
            negative_prompt = negative_prompt.encode('utf-8').decode('utf-8')
        
        validate_dimensions(custom_width, custom_height)
        
        generator = utils.seed_everything(seed)
        width, height = utils.aspect_ratio_handler(
            aspect_ratio_selector,
            custom_width,
            custom_height,
        )

        if add_quality_tags:
            prompt = QUALITY_TAGS.format(prompt=prompt)

        prompt, negative_prompt = utils.preprocess_prompt(
            prompt, negative_prompt
        )    

        width, height = utils.preprocess_image_dimensions(width, height)

        pipe = pipes.get(model_name)
        compel_proc = pipes.get("compel")
        
        if pipe is None or compel_proc is None:
            raise GenerationError(f"Модель {model_name} или модуль Compel не загружены")
            
        backup_scheduler = pipe.scheduler
        pipe.scheduler = utils.get_scheduler(pipe.scheduler.config, sampler)
            
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": f"{width} x {height}",
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "sampler": sampler,
            "Model": "Heartsync/NSFW-Uncensored",
        }
        
        # --- ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ЧЕРЕЗ COMPEL С ПОДДЕРЖКОЙ ВЕСОВ ---
        prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel_proc(negative_prompt if negative_prompt else "")
        
        # Выравнивание длины тензоров (обязательный шаг для Compel)
        [prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
        # -------------------------------------------------------------
        
        images = pipe(
            prompt_embeds=prompt_embeds,                                 # Используем эмбеддинги вместо текста
            pooled_prompt_embeds=pooled_prompt_embeds,                   #
            negative_prompt_embeds=negative_prompt_embeds,               #
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, #
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
        ).images

        if images:
            total = len(images)
            image_paths = []
            for idx, image in enumerate(images, 1):
                progress(idx/total, desc="Сохранение изображений...")
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
# ЗАГРУЗКА МОДЕЛИ HEARTSYNC И COMPEL
# ------------------------------------------------------------
pipes = {}
logger.info("Запуск загрузки модели Heartsync/NSFW-Uncensored...")
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "Heartsync/NSFW-Uncensored",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    if torch.cuda.is_available():
        # Оптимизация VRAM
        for sub in (pipe.text_encoder, pipe.text_encoder_2, pipe.vae, pipe.unet):
            sub.to(torch.float16)
            
        # Защита от OOM
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
            
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
    
    logger.info("Модель и парсер весов Compel успешно загружены!")
except Exception as e:
    logger.error(f"Критическая ошибка загрузки модели: {e}")

def update_history(new_images, metadata, current_history_data):
    if current_history_data is None:
        current_history_data = []
    
    if new_images:
        if isinstance(new_images, str):
            new_images = [new_images]
            prompt_text = metadata.get("prompt", "")
            seed = metadata.get("seed", "")
            res = metadata.get("resolution", "")
            steps = metadata.get("num_inference_steps", "")
            cfg = metadata.get("guidance_scale", "")
        
        caption = f"{prompt_text}\nSeed: {seed}, Res: {res}, Steps: {steps}, CFG: {cfg}"
        
        new_entries = []
        for img_item in new_images:
            img_path = img_item[0] if isinstance(img_item, tuple) else img_item.get("image", img_item) if isinstance(img_item, dict) else img_item.path if hasattr(img_item, "path") else img_item
            new_entries.append({"image": img_path, "caption": caption})
            
        current_history_data = new_entries + current_history_data
    
    gallery_images = [(item["image"], item["caption"]) for item in current_history_data]
    return current_history_data, gallery_images


# Интеграция CSS
fixed_css = config_css + """
:root {
  --gradient-primary: linear-gradient(45deg, #c136eb, #4EACEF);
  --box-shadow-custom: 0 4px 12px rgba(0, 0, 0, 0.08);
}

/* ЦЕНТРИРОВАНИЕ ГЛАВНОГО БЛОКА */
.main-card {
    width: 100% !important;
    max-width: 1000px !important;
    margin: 2rem auto !important;
    padding: 0 1rem !important;
    display: flex !important;
    flex-direction: column !important;
    align-self: center !important;
}

.custom-title {
    font-size: clamp(1.5rem, 4vw, 2.2rem);
    font-weight: 700;
    text-transform: uppercase;
    margin: 0 0 0.25rem 0;
    background-image: var(--gradient-primary);
    -webkit-text-fill-color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    display: inline-block;
}

.custom-subtitle {
    font-size: 0.9em;
    opacity: 0.8;
    color: #888;
}

.panel, div[class*="panel"] {
    border-radius: 12px !important;
    box-shadow: var(--box-shadow-custom) !important;
    border: 1px solid var(--border-color-primary) !important;
    transition: box-shadow 0.2s ease !important;
    padding: 1.5rem !important;
}

.panel:hover {
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12) !important;
}

button.primary {
    background: #1565c0 !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(21, 101, 192, 0.25) !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-weight: 600 !important;
}

button.primary:hover {
    background: #1976d2 !important;
    box-shadow: 0 4px 12px rgba(21, 101, 192, 0.35) !important;
    transform: translateY(-2px) !important;
}

textarea, input[type="text"] {
    border-radius: 8px !important;
    resize: none !important;
}

dialog .thumbnails, dialog button[aria-label^="Thumbnail"], dialog [data-testid="thumbnail-container"] {
    display: none !important;
}

.caption, figcaption, div[class*="caption"], span[class*="caption"] {
    white-space: pre-wrap !important; 
    word-break: break-word !important;
    overflow-y: auto !important; 
    max-height: 150px !important; 
    line-height: 1.3 !important;
    font-size: 0.9em !important; 
    padding-bottom: 10px !important;
    padding-top: 10px !important;
}

dialog .image-container, dialog .wrapper {
    max-height: 75vh !important;
}

#result-image {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}
#result-image > div {
    border-radius: 12px !important;
    overflow: hidden !important;
}
#result-image img {
    width: 100% !important;
    height: auto !important;
    max-height: 85vh !important;
    object-fit: cover !important;
}

body, .gradio-container, .wrap {
    background: radial-gradient(circle at 50% 0%, #2a164d 0%, #0d0f18 50%, #050508 100%) !important;
    background-attachment: fixed !important;
}

.panel, div[class*="panel"] {
    background: rgba(20, 22, 30, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
}

textarea, input[type="text"], input[type="number"], .gr-box {
    background: rgba(10, 12, 18, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #c136eb !important;
    box-shadow: 0 0 10px rgba(193, 54, 235, 0.3) !important;
}

#theme-btn { display: none !important; }

#history-gallery button .caption-label,
#history-gallery button figcaption,
#history-gallery button .caption {
    display: none !important;
}

dialog .caption-label,
dialog figcaption,
dialog .caption {
    display: block !important;
    opacity: 1 !important;
    visibility: visible !important;
    background: rgba(10, 12, 18, 0.85) !important;
    padding: 15px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    margin-top: 10px !important;
    color: #ffffff !important;
    white-space: pre-wrap !important;
    position: relative !important;
    z-index: 1000 !important;
}

dialog .thumbnail-item,
dialog .thumbnail-small,
dialog .svelte-7anmrz {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
}
"""

with gr.Blocks() as demo:
    history_data_state = gr.State([])
    hidden_model_name = gr.State("Heartsync") 
    hidden_metadata = gr.State({}) 

    with gr.Column(elem_classes="main-card"):
        
        with gr.Row(variant="panel"):
            gr.HTML(
                """
                <div style="text-align: center; width: 100%;">
                    <h1 class="custom-title">Heartsync NSFW-Uncensored SDXL</h1>
                    <div class="custom-subtitle">Интерфейс для генерации на базе <a href="https://huggingface.com/Heartsync/NSFW-Uncensored" target="_blank" style="color: inherit; text-decoration: underline;">Heartsync SDXL</a></div>
                </div>
                """
            )

        with gr.Row(variant="panel", equal_height=True):
            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    label="Промпт",
                    lines=4,
                    placeholder="Опишите, что вы хотите сгенерировать... (Используйте скобки для весов, например: (слово:1.5))",
                    show_label=True,
                    container=True,
                    # show_copy_button=True
                )
            with gr.Column(scale=1, min_width=150):
                run_button = gr.Button("Generate", variant="primary", size="lg")

        result = gr.Image(
            label="Сгенерированное изображение",
            type="filepath",
            show_label=True,
            elem_classes="panel",
            elem_id="result-image"
        )
        
        gr.Markdown("### ⚙️ Расширенные настройки")
        with gr.Column(variant="panel"):
            negative_prompt = gr.Textbox(
                label="Негативный промпт",
                lines=2,
                placeholder="Опишите, чего не должно быть на изображении... (Можно использовать (слово:1.5))",
                value=DEFAULT_NEGATIVE_PROMPT,
            )
            
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
                        num_inference_steps = gr.Slider(label="Количество шагов (Steps)", minimum=1, maximum=50, step=1, value=20)
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
            selected_history_info = gr.Textbox(
                label="Промпт и параметры выбранной генерации (нажмите на картинку в истории)",
                # show_copy_button=True,
                interactive=False,
                lines=3
            )

    aspect_ratio_selector.change(
        fn=lambda x: gr.update(visible=x == "Custom"),
        inputs=aspect_ratio_selector,
        outputs=custom_resolution,
        queue=False,
        api_name=False,
    )

    def precheck_prompt(p):
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
        fn=precheck_prompt, 
        inputs=[prompt],
        outputs=[],
        queue=False,
    ).then(
        fn=lambda: gr.update(interactive=False, value="Загрузка..."), 
        outputs=run_button,
    ).then(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
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
    )

    def on_history_select(evt: gr.SelectData, history_list):
        if history_list and evt.index < len(history_list):
            return history_list[evt.index]["caption"]
        return ""

    history_gallery.select(
        fn=on_history_select,
        inputs=[history_data_state],
        outputs=[selected_history_info]
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        debug=IS_COLAB, 
        share=IS_COLAB,
        css=fixed_css,
        theme="Nymbo/Nymbo_Theme_5"
    )