import json
import logging
import os
import pathlib
import re
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, resize_pos_embed, get_cast_dtype,set_model_preprocess_cfg,resize_text_pos_embed
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import PreprocessCfg,merge_preprocess_dict,merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs
_MODEL_CKPT_PATHS = {'biomedclip_local': Path(__file__).parent / "ckpt/open_clip_pytorch_model.bin"}

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS

    # 查找配置文件,定义支持的配置文件扩展名（.json）
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    # 读取配置文件
    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if "model_cfg" in model_cfg:
                model_cfg = model_cfg["model_cfg"]
            # print(model_cfg['model_cfg'])
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())

def get_model_config(model_name):
    # print(_MODEL_CONFIGS)
    if model_name in _MODEL_CONFIGS:
        # print('herehere')
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        **kwargs,
):

    config = get_model_config(model_name)
    assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        tokenizer = HFTokenizer(
            text_config['hf_tokenizer_name'],
            context_length=context_length,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):

    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        adapter = False,
        cache_dir: Optional[str] = None,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    checkpoint_path = None
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')
    
    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size
    

    is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
    if pretrained_image:
        if is_timm_model:
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert False, 'pretrained image towers currently only supported for timm models'
    
    
    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = get_cast_dtype(precision)
    is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
    if is_hf_model:
        # load pretrained weights for HF text model IFF no CLIP weights being loaded
        model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
    custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

    if custom_text:
        model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
    else:
        model = CLIP(**model_cfg, cast_dtype=cast_dtype)
    
    model.to(device=device)


    # 加载预训练模型权重
    logging.info(f'Loading pretrained {model_name} weight from {_MODEL_CKPT_PATHS}.')
    pretrained_loaded = False
    if pretrained:
        checkpoint_path = pretrained
     
        if checkpoint_path:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
        else:
            error_str = (
                f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
            logging.warning(error_str)
            raise RuntimeError(error_str)
        pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
        # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
            f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')
        
    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))
    tokenizer = get_tokenizer('biomedclip_local')

    return model,tokenizer
