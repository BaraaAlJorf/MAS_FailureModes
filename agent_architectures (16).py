import os
import gc
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers import PaliGemmaForConditionalGeneration as AutoModelForMultimodalLM
from transformers import AutoModelForImageTextToText
from transformers.generation import GenerationMixin
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import re
import json

# --------------------------------------------------------------------
# 1. InternVL-specific helpers
# --------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _is_internvl_model(model_or_id):
    model_id = model_or_id if isinstance(model_or_id, str) else getattr(model_or_id, "model_id", "")
    return "internvl" in (model_id or "").lower()

def _get_module_device(module):
    try:
        return next(module.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_module_dtype(module):
    try:
        return next(module.parameters()).dtype
    except Exception:
        return torch.float32

def _build_internvl_device_map(model_id):
    short_name = model_id.split("/")[-1]
    if torch.cuda.device_count() <= 1:
        return None

    num_layers_lookup = {
        "InternVL2-1B": 24, "InternVL2-2B": 24, "InternVL2-4B": 32, "InternVL2-8B": 32,
        "InternVL2-26B": 48, "InternVL2-40B": 60, "InternVL2-Llama3-76B": 80,
    }
    num_layers = num_layers_lookup.get(short_name)
    if num_layers is None:
        return None

    world_size = torch.cuda.device_count()
    device_map = {}
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for gpu_idx, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers: break
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map

def _internvl_build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.BICUBIC)
    processed_images = []
    tiles_per_row = target_width // image_size

    for i in range(blocks):
        box = ((i % tiles_per_row) * image_size, (i // tiles_per_row) * image_size, ((i % tiles_per_row) + 1) * image_size, ((i // tiles_per_row) + 1) * image_size)
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.BICUBIC))

    return processed_images

def _internvl_load_image_tensor(image_obj, input_size=448, max_num=12):
    if isinstance(image_obj, str): image = Image.open(image_obj).convert("RGB")
    elif isinstance(image_obj, Image.Image): image = image_obj.convert("RGB")
    else: raise TypeError(f"Unsupported InternVL image type: {type(image_obj)}")

    transform = _internvl_build_transform(input_size=input_size)
    image_tiles = _internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in image_tiles], dim=0)

def _internvl_flatten_content(content):
    if isinstance(content, str): return content, []
    text_parts = []
    images = []
    for item in content:
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if text: text_parts.append(text)
        elif item_type == "image":
            images.append(item.get("image"))
            text_parts.append("<image>")
    rendered = ""
    for part in text_parts:
        if part == "<image>":
            if rendered and not rendered.endswith("\n"): rendered += "\n"
            rendered += "<image>\n"
        else:
            rendered += part
            if part and not part.endswith("\n"): rendered += "\n"
    return rendered.rstrip(), images

def _internvl_build_query_and_pixels(conversation, model, tokenizer, max_num=12):
    template = model.conv_template.copy()
    template.messages = []
    template.system_message = getattr(model, "system_message", template.system_message)
    rendered_turns = []
    collected_images = []

    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        rendered_text, msg_images = _internvl_flatten_content(content)
        collected_images.extend(msg_images)
        if role == "system":
            if rendered_text: template.system_message = template.system_message.rstrip() + "\n" + rendered_text if template.system_message else rendered_text
            continue
        if role == "user": rendered_turns.append((template.roles[0], rendered_text))
        elif role == "assistant": rendered_turns.append((template.roles[1], rendered_text))
        else: rendered_turns.append((template.roles[0], rendered_text))

    assistant_prefix = None
    if rendered_turns and rendered_turns[-1][0] == template.roles[1]:
        assistant_prefix = rendered_turns[-1][1] or ""
        rendered_turns = rendered_turns[:-1]

    for role, message in rendered_turns: template.append_message(role, message)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    if assistant_prefix: query += assistant_prefix

    if collected_images:
        pixel_values_list = [_internvl_load_image_tensor(img, input_size=448, max_num=max_num) for img in collected_images]
        num_patches_list = [pv.shape[0] for pv in pixel_values_list]
        pixel_values = torch.cat(pixel_values_list, dim=0)
    else:
        pixel_values = None
        num_patches_list = []

    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    for num_patches in num_patches_list:
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * model.num_image_token * num_patches) + "</img>"
        query = query.replace("<image>", image_tokens, 1)

    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    return query, pixel_values, num_patches_list, eos_token_id, template

def _internvl_generate_one(conversation, model, tokenizer, max_tokens, generation_kwargs):
    query, pixel_values, _, eos_token_id, template = _internvl_build_query_and_pixels(conversation, model, tokenizer)
    text_device = _get_module_device(model.language_model.get_input_embeddings())
    vision_device = _get_module_device(model.vision_model) if hasattr(model, "vision_model") else text_device
    vision_dtype = _get_module_dtype(model.vision_model) if hasattr(model, "vision_model") else _get_module_dtype(model)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(text_device)
    attention_mask = model_inputs["attention_mask"].to(text_device)
    if pixel_values is not None: pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)

    gen_kwargs = {"max_new_tokens": max_tokens, **generation_kwargs}
    gen_kwargs.setdefault("eos_token_id", eos_token_id)

    with torch.inference_mode():
        outputs = model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response.split(template.sep.strip())[0].strip()

def _internvl_yes_no_one(conversation, model, tokenizer):
    query, pixel_values, _, _, _ = _internvl_build_query_and_pixels(conversation, model, tokenizer)
    text_device = _get_module_device(model.language_model.get_input_embeddings())
    vision_device = _get_module_device(model.vision_model) if hasattr(model, "vision_model") else text_device
    vision_dtype = _get_module_dtype(model.vision_model) if hasattr(model, "vision_model") else _get_module_dtype(model)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(text_device)
    attention_mask = model_inputs["attention_mask"].to(text_device)

    with torch.inference_mode():
        input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)
            vit_embeds = model.extract_feature(pixel_values)
            bsz, seq_len, hidden = input_embeds.shape
            flat_embeds = input_embeds.reshape(bsz * seq_len, hidden)
            flat_ids = input_ids.reshape(bsz * seq_len)
            selected = flat_ids == model.img_context_token_id
            flat_embeds[selected] = vit_embeds.reshape(-1, hidden).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(bsz, seq_len, hidden)

        outputs = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

    if getattr(tokenizer, "yes_token_id", None) is not None and getattr(tokenizer, "no_token_id", None) is not None:
        probs = F.softmax(torch.stack([next_token_logits[:, tokenizer.no_token_id], next_token_logits[:, tokenizer.yes_token_id]], dim=-1), dim=-1)
        yes_prob = float(probs[:, 1].cpu().item())
    else: yes_prob = 0.0

    top_text = tokenizer.decode([int(torch.argmax(next_token_logits, dim=-1).item())], skip_special_tokens=True).strip()
    return top_text, yes_prob

# --------------------------------------------------------------------
# 2. Model Loading & Processor Setup
# --------------------------------------------------------------------

MODEL_CACHE = {}

def get_model_and_processor(args):
    """Loads and caches the model/processor."""
    model_id = args.model_id
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]

    print(f"Initializing Model and Processor for '{model_id}'...")
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {
        "token": hf_token, "device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    }

    try:
        # Load Processor
        if "InternVL" in model_id:
            processor = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True, use_fast=False)
            processor.padding_side = "left"
            if processor.pad_token is None: processor.pad_token = processor.eos_token
        elif "Qwen" in model_id:
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            if hasattr(processor, 'tokenizer'):
                processor.tokenizer.padding_side = "left"
                processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        else:
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True)

        # Load Model
        if "medgemma" in model_id.lower():
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        elif "Qwen" in model_id:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            except ImportError:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        elif "InternVL" in model_id:
            internvl_kwargs = load_kwargs.copy()
            internvl_kwargs.pop("device_map", None)
            internvl_kwargs["use_flash_attn"] = True
            device_map = _build_internvl_device_map(model_id)
            if device_map: internvl_kwargs["device_map"] = device_map
            model = AutoModel.from_pretrained(model_id, **internvl_kwargs)
            if not device_map and torch.cuda.is_available(): model = model.cuda()
            
            try:
                if hasattr(model, "language_model"):
                    lm_model = model.language_model
                    lm_class = lm_model.__class__
                    if not issubclass(lm_class, GenerationMixin):
                        lm_class.__bases__ = (GenerationMixin,) + tuple(b for b in lm_class.__bases__ if b is not GenerationMixin)
                    if not hasattr(lm_model, "generation_config") or lm_model.generation_config is None:
                        lm_model.generation_config = GenerationConfig.from_model_config(lm_model.config)
            except Exception as patch_e: print(f"Warning: InternVL compat patch: {patch_e}")
            model.eval()
            model.img_context_token_id = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        elif "Phi-4" in model_id:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
            config.attn_implementation = "eager"
            model = AutoModelForCausalLM.from_pretrained(model_id, config=config, **load_kwargs)
        elif "Ovis" in model_id:
            ovis_kwargs = load_kwargs.copy()
            ovis_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **ovis_kwargs).to("cuda")
        elif "llava" in model_id.lower():
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        else:
            try: model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            except Exception: model = AutoModel.from_pretrained(model_id, **load_kwargs)

        model.eval()
        model.model_id = model_id

        # Tokenizer Yes/No Handling
        try:
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            candidates_yes = ["Yes", " Yes", "yes", " yes"]
            candidates_no = ["No", " No", "no", " no"]
            yes_id, no_id = None, None
            for c in candidates_yes:
                ids = tokenizer.encode(c, add_special_tokens=False)
                if len(ids) == 1: yes_id = ids[0]; break
            for c in candidates_no:
                ids = tokenizer.encode(c, add_special_tokens=False)
                if len(ids) == 1: no_id = ids[0]; break
            
            processor.yes_token_id = yes_id or tokenizer.encode("Yes", add_special_tokens=False)[0]
            processor.no_token_id = no_id or tokenizer.encode("No", add_special_tokens=False)[0]
        except Exception:
            processor.yes_token_id, processor.no_token_id = None, None

        MODEL_CACHE[model_id] = (model, processor)
        return model, processor
    except Exception as e:
        print(f"❌ Failed to load model '{model_id}'. Error: {e}")
        raise

# --------------------------------------------------------------------
# 3. Input Preparation & Generation
# --------------------------------------------------------------------

def _prepare_inputs_for_vlm(prompts, processor, device, model_id):
    mid = (model_id or "").lower()
    is_qwen = "qwen" in mid
    is_ovis = "ovis" in mid
    is_medgemma = "medgemma" in mid
    is_llava = "llava" in mid
    
    if is_medgemma:
        batch_inputs = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt") for c in prompts]
        if len(batch_inputs) == 1: return {k: v.to(device) for k, v in batch_inputs[0].items()}
        padded = processor.tokenizer.pad({"input_ids": [x["input_ids"][0] for x in batch_inputs], "attention_mask": [x["attention_mask"][0] for x in batch_inputs]}, padding=True, return_tensors="pt")
        out = {"input_ids": padded["input_ids"].to(device), "attention_mask": padded["attention_mask"].to(device)}
        has_pv = ["pixel_values" in x for x in batch_inputs]
        if all(has_pv): out["pixel_values"] = torch.cat([x["pixel_values"] for x in batch_inputs], dim=0).to(device)
        elif any(has_pv): out["pixel_values_list"] = [x["pixel_values"].to(device) if "pixel_values" in x else None for x in batch_inputs]
        return out

    elif is_llava:
        prompt_texts = []
        images_grouped = []  
        for conversation in prompts:
            try: prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            except Exception: prompt = str(conversation)
            prompt_texts.append(prompt)
            conv_images = [item.get("image") for msg in conversation if isinstance(msg.get("content"), list) for item in msg["content"] if item.get("type") == "image" and item.get("image") is not None]
            images_grouped.append(conv_images if conv_images else None)
        images_arg = images_grouped if any(imgs is not None for imgs in images_grouped) else None
        return processor(text=prompt_texts, images=images_arg, return_tensors="pt", padding=True).to(device)

    elif is_qwen:
        try:
            from qwen_vl_utils import process_vision_info
            texts = [processor.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
            image_inputs, video_inputs = process_vision_info(prompts)
            return processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        except ImportError: pass

    elif is_ovis:
        formatted_texts = []
        batch_pixel_values = [None] * len(prompts) 
        for i, conversation in enumerate(prompts):
            flattened_conv = []
            patient_images = [] 
            for msg in conversation:
                if isinstance(msg['content'], list):
                    text_part = "".join(item['text'] for item in msg['content'] if item['type'] == 'text')
                    for item in msg['content']:
                        if item['type'] == 'image': patient_images.append(item['image'])
                    img_count = len(patient_images)
                    if img_count > 0 and "<image>" not in text_part: text_part = ("<image>\n" * img_count) + text_part
                    flattened_conv.append({"role": msg['role'], "content": text_part})
                else: flattened_conv.append(msg)
            txt = processor.tokenizer.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True) if hasattr(processor, "tokenizer") else processor.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(txt)
            if patient_images: batch_pixel_values[i] = processor.image_processor(images=patient_images, return_tensors="pt").pixel_values.to(device)
        inputs = processor(text=formatted_texts, return_tensors="pt", padding=True).to(device)
        inputs["pixel_values"] = batch_pixel_values
        return inputs

    else:
        formatted_texts = []
        all_images = []
        has_images = False
        for conversation in prompts:
            flattened_conv = []
            for msg in conversation:
                if isinstance(msg['content'], list):
                    text_part = "".join(item['text'] for item in msg['content'] if item['type'] == 'text')
                    img_count = sum(1 for item in msg['content'] if item['type'] == 'image')
                    for item in msg['content']:
                        if item['type'] == 'image': all_images.append(item['image'])
                    if img_count > 0 and "<image>" not in text_part: text_part = ("<image>\n" * img_count) + text_part
                    flattened_conv.append({"role": msg['role'], "content": text_part})
                else: flattened_conv.append(msg)
            txt = processor.tokenizer.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True) if hasattr(processor, "tokenizer") else processor.apply_chat_template(flattened_conv, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(txt)
        inputs = processor(text=formatted_texts, return_tensors="pt", padding=True)
        if all_images and hasattr(processor, "image_processor"): inputs["pixel_values"] = processor.image_processor(images=all_images, return_tensors="pt").pixel_values
        return inputs.to(device)

def generate_response(prompts, model, processor, max_tokens, **generation_kwargs):
    model_id = getattr(model, "model_id", "") or ""
    if _is_internvl_model(model_id): return [_internvl_generate_one(c, model, processor, max_tokens, generation_kwargs) for c in prompts]

    inputs = _prepare_inputs_for_vlm(prompts, processor, model.device, model_id=model_id)
    inputs.pop("pixel_values_list", None)

    try:
        with torch.inference_mode():
            if "medgemma" in model_id:
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, **generation_kwargs)
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                return [processor.decode(outputs[j][int(input_lens[j]):], skip_special_tokens=True) for j in range(outputs.shape[0])]
            elif "Ovis" in model.__class__.__name__:
                gen_args = {"inputs": inputs.get("input_ids"), "attention_mask": inputs.get("attention_mask"), "max_new_tokens": max_tokens, **generation_kwargs}
                gen_args["pixel_values"] = inputs.get("pixel_values") if inputs.get("pixel_values") is not None else [None] * inputs.get("input_ids").shape[0]
                outputs = model.generate(**gen_args)
            else:
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, **generation_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
            torch.cuda.empty_cache(); gc.collect()
        raise

    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

def generate_yes_no_probability(prompts, model, processor, max_tokens=1):
    model_id = getattr(model, "model_id", "") or ""
    if _is_internvl_model(model_id):
        res = [_internvl_yes_no_one(c, model, processor) for c in prompts]
        return [r[0] for r in res], np.asarray([r[1] for r in res], dtype=np.float32)

    inputs = _prepare_inputs_for_vlm(prompts, processor, model.device, model_id=model_id)

    try:
        with torch.inference_mode():
            if "Ovis" in model.__class__.__name__:
                forward_args = {"input_ids": inputs.get("input_ids"), "attention_mask": inputs.get("attention_mask"), "labels": None}
                forward_args["pixel_values"] = inputs.get("pixel_values") if inputs.get("pixel_values") is not None else [None] * inputs.get("input_ids").shape[0]
                outputs = model(**forward_args)
            else:
                outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); gc.collect()
        raise

    yes_probs = np.zeros(len(prompts), dtype=np.float32)
    if getattr(processor, 'yes_token_id', None) is not None:
        try:
            yes_score, no_score = next_token_logits[:, processor.yes_token_id], next_token_logits[:, processor.no_token_id]
            yes_probs = F.softmax(torch.stack([no_score, yes_score], dim=1), dim=-1)[:, 1].cpu().float().numpy()
            yes_probs = np.nan_to_num(yes_probs, nan=0.0)
        except Exception: pass

    try:
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        decoded_texts = tokenizer.batch_decode(torch.argmax(next_token_logits, dim=-1).unsqueeze(-1), skip_special_tokens=True)
    except Exception: decoded_texts = ["N/A"] * len(prompts)

    return [t.strip() for t in decoded_texts], yes_probs

# --------------------------------------------------------------------
# 4. Modality Parsing & Result Formatting
# --------------------------------------------------------------------

def _parse_allowed_modalities(args):
    if not hasattr(args, 'modalities') or not args.modalities:
        return ['ehr', 'cxr', 'rr', 'ps']
    return [m.strip().lower() for m in args.modalities.split('-')]

def calibrate_prob(prob, temperature):
    """Applies temperature scaling to a scalar probability."""
    if temperature == 1.0: return prob
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    logit = np.log(prob / (1.0 - prob))
    scaled_logit = logit / temperature
    return 1.0 / (1.0 + np.exp(-scaled_logit))

def _format_batch_results(batch, final_texts, final_probs, unimodal_preds=None, expert_traces=None):
    batch_results = []
    for i, p in enumerate(batch):
        exp_prob = float(final_probs[i])
        res = {
            'subject_id': p.get('subject_id'),
            'stay_id': p['stay_id'], 
            'ground_truth': p['labels'],
            'predictions': {
                'in_hospital_mortality_48hr': 1 if exp_prob >= 0.5 else 0, 
                'mortality_probability': exp_prob,
                'mortality_probability_text': final_texts[i]
            }
        }
        if unimodal_preds:
            res['unimodal_predictions'] = {mod: float(preds[i]) for mod, preds in unimodal_preds.items()}
        if expert_traces:
            res['expert_traces'] = expert_traces[i]
            
        batch_results.append(res)
    return batch_results

# --------------------------------------------------------------------
# 5. Core Prompt Builder
# --------------------------------------------------------------------

def _build_prompt_content(patient_data, ehr_text, allowed_modalities, prompt_setup="simple", rulebook=None, weights=None):
    """
    Constructs the input context. STRICTLY omits mention of missing modalities.
    If 'weights' dict is provided, it injects the reliability weight next to the modality header.
    """
    content = []
    has_image = False
    
    data_parts = ["--- PATIENT DATA ---"]
    
    def _get_w(mod):
        """Helper to format the weight string if available."""
        return f" (Reliability Weight: {weights[mod]:.3f})" if weights and mod in weights else ""
    
    if 'ps' in allowed_modalities and patient_data.get('patient_summary_text'):
        data_parts.append(f"Patient Summary{_get_w('ps')}:\n{patient_data['patient_summary_text']}")
        
    if 'rr' in allowed_modalities and patient_data.get('radiology_report_text') and patient_data.get('radiology_report_text') != 'Radiology report not available':
        data_parts.append(f"Radiology Reports{_get_w('rr')}:\n{patient_data['radiology_report_text']}")
        
    if 'cxr' in allowed_modalities:
        if 'pil_image' in patient_data and patient_data['pil_image']:
             content.append({"type": "image", "image": patient_data['pil_image']})
             has_image = True
             data_parts.append(f"Chest X-ray{_get_w('cxr')}: [Image Attached]")
        elif patient_data.get('cxr_image_path') and patient_data.get('cxr_image_path') != 'CXR not available':
            if os.path.exists(patient_data['cxr_image_path']):
                content.append({"type": "image", "image": Image.open(patient_data['cxr_image_path']).convert("RGB")})
                has_image = True
                data_parts.append(f"Chest X-ray{_get_w('cxr')}: [Image Attached]")
                
    if 'ehr' in allowed_modalities and ehr_text and ehr_text != "EHR Data Not Available":
        data_parts.append(f"Electronic Health Records{_get_w('ehr')}:\n{ehr_text}")

    data_block = "\n\n".join(data_parts)
    prompt_text = f"{data_block}\n\n--- DECISION ---\nDoes this patient die in the ICU?\n"

    if prompt_setup == "none":
        prompt_text += "Answer:"
    elif prompt_setup == "simple":
        prompt_text += "Answer using only one word - Yes or No, then provide your reasoning.\nAnswer:"
    elif prompt_setup == "moderate":
        prompt_text += "Output your response in strict JSON format: {\"prediction\": \"Yes/No\", \"reasoning\": \"...\"}.\nAnswer:"
    elif prompt_setup in ["complex", "bad"] and rulebook:
        rules = "\n".join(rulebook)
        prompt_text += f"Use the following diagnostic rulebook strictly to make your decision:\n{rules}\nAnswer using only one word - Yes or No, then provide your reasoning.\nAnswer:"
    else:
        prompt_text += "Answer:"

    content.append({"type": "text", "text": prompt_text})
    return content, has_image


# =========================================================================
# A. Single Agent (Zeroshot)
# =========================================================================
def run_single_agent(batch, model, processor, args, calibration_params=None):
    allowed_mods = _parse_allowed_modalities(args)
    # NEW
    is_temp = getattr(args, 'calibration_type', None) == 'temperature' and calibration_params is not None
    is_weighted = getattr(args, 'calibration_type', None) == 'weighted' and calibration_params is not None
    
    # If running weighted calibration, pass the weights down to inject into headers
    weights_to_pass = calibration_params if is_weighted else None
    
    prompts = []
    for p in batch:
        rulebook = args.rulebooks.get('single') if hasattr(args, 'rulebooks') else None
        content, _ = _build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, args.prompt_setup, rulebook, weights_to_pass)
        prompts.append([{"role": "user", "content": content}])
        
    # Phase 1: Binary Logit Probe
    _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
    
    # Apply Temperature Calibration
    if is_temp:
        t = calibration_params.get('single', 1.0)
        probs = np.array([calibrate_prob(p, t) for p in probs])
    
    # Phase 2: Reasoning Trace
    reasoning_prompts = copy.deepcopy(prompts)
    for i, prob in enumerate(probs):
        ans = "Yes" if prob >= 0.5 else "No"
        reasoning_prompts[i][0]['content'][-1]['text'] += f" {ans}\nReasoning:"
        
    reasonings = generate_response(reasoning_prompts, model, processor, max_tokens=150)
    
    return _format_batch_results(batch, reasonings, probs)


# =========================================================================
# B. Multi-Agent Independent (Majority Vote)
# =========================================================================
def run_majority_vote(batch, model, processor, args, calibration_params=None):
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    
    expert_analyses = ["" for _ in range(batch_size)]
    unimodal_predictions = {mod: np.zeros(batch_size) for mod in allowed_mods}
    weighted_probs = np.zeros(batch_size)
    total_weight = 0.0
    
    # NEW
    is_temp = getattr(args, 'calibration_type', None) == 'temperature' and calibration_params is not None
    is_weighted = getattr(args, 'calibration_type', None) == 'weighted' and calibration_params is not None

    for mod in allowed_mods:
        prompts = []
        for p in batch:
            rules = args.rulebooks.get(mod) if hasattr(args, 'rulebooks') else None
            content, _ = _build_prompt_content(p, p.get('ehr_text', ''), [mod], args.prompt_setup, rules)
            prompts.append([{"role": "user", "content": content}])
            
        # Step 1: Probe Yes/No
        _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
        
        # Step 2: Apply T-Scaling if active
        if is_temp:
            t = calibration_params.get(mod, 1.0)
            probs = np.array([calibrate_prob(p, t) for p in probs])
            
        unimodal_predictions[mod] = probs
        
        # Step 3: Mathmatically Apply Weights for final decision
        weight = calibration_params.get(mod, 1.0) if is_weighted else 1.0
        weighted_probs += probs * weight
        total_weight += weight
        
        # Step 4: Generate Trace Post-Answer
        reasoning_prompts = copy.deepcopy(prompts)
        for i, prob in enumerate(probs):
            ans = "Yes" if prob >= 0.5 else "No"
            reasoning_prompts[i][0]['content'][-1]['text'] += f" {ans}\nReasoning:"
            
        reasonings = generate_response(reasoning_prompts, model, processor, max_tokens=150)
        
        for i, reason in enumerate(reasonings):
            ans = "Yes" if probs[i] >= 0.5 else "No"
            expert_analyses[i] += f"--- {mod.upper()} EXPERT ---\nPrediction: {ans}\nReasoning: {reason}\n\n"

    final_probs = weighted_probs / total_weight if total_weight > 0 else np.zeros(batch_size)
    
    return _format_batch_results(batch, expert_analyses, final_probs, unimodal_predictions, expert_analyses)


# =========================================================================
# C. Multi-Agent Decentralized (Debate)
# =========================================================================
# =========================================================================
# C. Multi-Agent Decentralized (Debate)
# =========================================================================
def run_decentralized_debate(batch, model, processor, args, calibration_params=None):
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    n_rounds = 3
    
    is_temp = getattr(args, 'calibration_type', None) == 'temperature' and calibration_params
    is_weighted = getattr(args, 'calibration_type', None) == 'weighted' and calibration_params
    
    current_reasoning = {mod: ["No prior reasoning."] * batch_size for mod in allowed_mods}
    unimodal_predictions = {mod: np.zeros(batch_size) for mod in allowed_mods}
    
    for r in range(n_rounds):
        round_probs = {}
        for mod in allowed_mods:
            prompts = []
            for i, p in enumerate(batch):
                peer_txt = ""
                if r > 0:
                    peer_txt = "\n--- OTHER MODALITY SPECIALISTS OPINIONS ---\n"
                    for peer_mod in allowed_mods:
                        if mod != peer_mod: 
                            weight_str = f" (Reliability Weight: {calibration_params[peer_mod]:.3f})" if is_weighted else ""
                            peer_txt += f"Expert {peer_mod.upper()}{weight_str}: {current_reasoning[peer_mod][i]}\n"
                
                rules = args.rulebooks.get(mod) if hasattr(args, 'rulebooks') else None
                
                # Use the actual prompt_setup so rulebooks and JSON instructions trigger
                content, _ = _build_prompt_content(p, p.get('ehr_text', ''), [mod], args.prompt_setup, rules)
                
                # Inject peer text cleanly before the Decision block
                if peer_txt:
                    decision_split = content[-1]['text'].split("--- DECISION ---")
                    if len(decision_split) == 2:
                        content[-1]['text'] = f"{decision_split[0]}\n{peer_txt}\n--- DECISION ---{decision_split[1]}"
                
                prompts.append([{"role": "user", "content": content}])

            _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            
            if is_temp:
                t = calibration_params.get(mod, 1.0)
                probs = np.array([calibrate_prob(p, t) for p in probs])
                
            round_probs[mod] = probs
            unimodal_predictions[mod] = probs 
            
            reasoning_prompts = copy.deepcopy(prompts)
            for i, prob in enumerate(probs):
                ans = "Yes" if prob >= 0.5 else "No"
                reasoning_prompts[i][0]['content'][-1]['text'] += f" {ans}\nReasoning:"
                
            reasonings = generate_response(reasoning_prompts, model, processor, max_tokens=150)
            
            for i, reason in enumerate(reasonings):
                ans = "Yes" if probs[i] >= 0.5 else "No"
                current_reasoning[mod][i] = f"{ans} [{reason.strip()}]"

    avg_probs = np.mean(list(unimodal_predictions.values()), axis=0)
    
    final_logs = [""] * batch_size
    for i in range(batch_size):
        for mod in allowed_mods:
            final_logs[i] += f"{mod.upper()}: {current_reasoning[mod][i]}\n"

    return _format_batch_results(batch, final_logs, avg_probs, unimodal_predictions, final_logs)


# =========================================================================
# D. Multi-Agent Centralized (AgentiCDS Simple)
# =========================================================================
def run_centralized_judge(batch, model, processor, args, calibration_params=None):
    allowed_mods = _parse_allowed_modalities(args)
    batch_size = len(batch)
    
    expert_analyses = ["" for _ in range(batch_size)]
    unimodal_predictions = {mod: np.zeros(batch_size) for mod in allowed_mods}
    # NEW
    is_temp = getattr(args, 'calibration_type', None) == 'temperature' and calibration_params is not None
    is_weighted = getattr(args, 'calibration_type', None) == 'weighted' and calibration_params is not None
    
    # ---------------------------------------------
    # PHASE 1: Unimodal Experts
    # ---------------------------------------------
    for mod in allowed_mods:
        prompts = []
        for i, p in enumerate(batch):
            rules = args.rulebooks.get(mod) if hasattr(args, 'rulebooks') else None
            content, _ = _build_prompt_content(p, p.get('ehr_text', ''), [mod], args.prompt_setup, rules)
            prompts.append([{"role": "user", "content": content}])
            
        _, probs = generate_yes_no_probability(prompts, model, processor, max_tokens=1)
        
        if is_temp:
            t = calibration_params.get(mod, 1.0)
            probs = np.array([calibrate_prob(p, t) for p in probs])
            
        unimodal_predictions[mod] = probs
        
        reasoning_prompts = copy.deepcopy(prompts)
        for i, prob in enumerate(probs):
            ans = "Yes" if prob >= 0.5 else "No"
            reasoning_prompts[i][0]['content'][-1]['text'] += f" {ans}\nReasoning:"
            
        reasonings = generate_response(reasoning_prompts, model, processor, max_tokens=150)
        
        for i, reason in enumerate(reasonings):
            ans = "Yes" if probs[i] >= 0.5 else "No"
            weight_str = f" (Reliability Weight: {calibration_params[mod]:.3f})" if is_weighted else ""
            expert_analyses[i] += f"--- {mod.upper()} EXPERT{weight_str} ---\nPrediction: {ans}\nReasoning: {reason}\n\n"

    # ---------------------------------------------
    # PHASE 2: Lead Judge Synthesis (COMPLIANCE PATCHED)
    # ---------------------------------------------
    judge_prompts = []
    
    # Extract rules and formatting for the Judge (using the 'single' rulebook)
    judge_rulebook = args.rulebooks.get('single') if hasattr(args, 'rulebooks') else None
    
    instruction_suffix = "Answer:"
    if args.prompt_setup == "simple":
        instruction_suffix = "Answer using only one word - Yes or No, then provide your reasoning.\nAnswer:"
    elif args.prompt_setup == "moderate":
        instruction_suffix = "Output your response in strict JSON format: {\"prediction\": \"Yes/No\", \"reasoning\": \"...\"}.\nAnswer:"
    elif args.prompt_setup in ["complex", "bad"] and judge_rulebook:
        rules_text = "\n".join(judge_rulebook)
        instruction_suffix = f"Use the following diagnostic rulebook strictly to make your decision:\n{rules_text}\nAnswer using only one word - Yes or No, then provide your reasoning.\nAnswer:"

    for i, p in enumerate(batch):
        txt = (
            f"You are the Lead Judge.\n"
            f"--- EXPERT REPORTS ---\n{expert_analyses[i]}\n"
            f"--- DECISION ---\n"
            f"Based strictly on these reports, does this patient die in the ICU?\n{instruction_suffix}"
        )
        judge_prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}]}])

    # Judge Probe Yes/No
    _, final_probs = generate_yes_no_probability(judge_prompts, model, processor, max_tokens=1)
    
    if is_temp:
        t_judge = calibration_params.get('judge', 1.0)
        final_probs = np.array([calibrate_prob(p, t_judge) for p in final_probs])
    
    # Judge Reasoning Post-Answer
    for i, prob in enumerate(final_probs):
        ans = "Yes" if prob >= 0.5 else "No"
        judge_prompts[i][0]['content'][0]['text'] += f" {ans}\nReasoning:"
    
    final_reasonings = generate_response(judge_prompts, model, processor, max_tokens=150)
    
    final_logs = [f"{expert_analyses[i]}\n--- JUDGE ---\n{final_reasonings[i]}" for i in range(batch_size)]

    return _format_batch_results(batch, final_reasonings, final_probs, unimodal_predictions, final_logs)
    
# =========================================================================
# E. Dynamic Complexity-Routed Framework
# =========================================================================
def run_dynamic_framework(batch, model, processor, args, calibration_params=None):
    tier1, tier2, tier3, tier4 = [], [], [], []
    
    for p in batch:
        # 1. Presence features
        has_cxr = 1.0 if p.get('cxr_image_path') and p.get('cxr_image_path') != 'CXR not available' else 0.0
        has_rr = 1.0 if p.get('radiology_report_text') and p.get('radiology_report_text') != 'Radiology report not available' else 0.0
        
        # 2. Normalized Length (Capped at 1.0)
        lengths = p.get('raw_lengths', {'ps':0, 'rr':0, 'ehr':0})
        norm_ps = min(1.0, lengths['ps'] / args.normalization_bounds['ps'])
        norm_rr = min(1.0, lengths['rr'] / args.normalization_bounds['rr'])
        norm_ehr = min(1.0, lengths['ehr'] / args.normalization_bounds['ehr'])
        
        # Calculate average of available modalities
        avail_mods = 0
        total_norm = 0
        if lengths['ps'] > 0 or p.get('patient_summary_text'):
            avail_mods += 1
            total_norm += norm_ps
        if has_rr > 0:
            avail_mods += 1
            total_norm += norm_rr
        if lengths['ehr'] > 0 or p.get('ehr_timeseries_path'):
            avail_mods += 1
            total_norm += norm_ehr
            
        l_norm = (total_norm / avail_mods) if avail_mods > 0 else 0.0
        
        # 3. Compute final complexity score
        c_score = (args.w_cxr * has_cxr) + (args.w_rr * has_rr) + (args.w_len * l_norm)
        
        # Attach for debugging/tracing later
        p['complexity_score'] = c_score 
        
        # 4. Route to Tier
        if c_score < 0.25: tier1.append(p)
        elif c_score < 0.50: tier2.append(p)
        elif c_score < 0.75: tier3.append(p)
        else: tier4.append(p)
        
    all_results = []
    
    # 5. Execute Sub-Batches Sequentially
    if tier1:
        t1_args = copy.deepcopy(args)
        t1_args.prompt_setup = "none"
        all_results.extend(run_single_agent(tier1, model, processor, t1_args, calibration_params))
        
    if tier2:
        t2_args = copy.deepcopy(args)
        t2_args.prompt_setup = "simple"
        all_results.extend(run_single_agent(tier2, model, processor, t2_args, calibration_params))
        
    if tier3:
        t3_args = copy.deepcopy(args)
        t3_args.prompt_setup = "simple"
        all_results.extend(run_centralized_judge(tier3, model, processor, t3_args, calibration_params))
        
    if tier4:
        t4_args = copy.deepcopy(args)
        t4_args.prompt_setup = "complex"
        all_results.extend(run_centralized_judge(tier4, model, processor, t4_args, calibration_params))
        
    return all_results