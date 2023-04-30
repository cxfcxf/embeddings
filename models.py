import logging

from pathlib import Path
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

# logging
LOG = logging.getLogger(__name__)

def load_quantized(model_name, args):
    path_to_model = Path(f'{args.model_dir}/{model_name}')
    LOG.debug(f"model path: {path_to_model}")

    model = AutoGPTQForCausalLM.from_quantized(path_to_model,  device="cuda:0", use_safetensors=args.use_safetensors)

    return model

def load_normal(model_name, args):\
    # VRAM warning
    path_to_model = Path(f'{args.model_dir}/{model_name}')
    LOG.debug(f"model path: {path_to_model}")

    model = AutoModelForCausalLM.from_pretrained(path_to_model, device=0)

    return model
