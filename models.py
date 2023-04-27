import re
import sys
import logging
from pathlib import Path

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

# in order to load 4 bits model (GPTQ), we need to install a special GPTQ library (setup_cuda.py install)
# then we load it in the sys path
sys.path.insert(0, str("GPTQ-for-LLaMa"))

from quant import make_quant

#from modelutils import find_layers
import torch
import torch.nn as nn

# logging
LOG = logging.getLogger(__name__)

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def _load_quant(model, checkpoint, wbits, groupsize=-1, exclude_layers=['lm_head']):
    config = AutoConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    # del layers
    
    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model

def load_quantized_gptq(model_name, args):
    if not "model_type" in args:
        # Try to determine model type from model name
        name = model_name.lower()
        if any((k in name for k in ['llama', 'alpaca'])):
            model_type = 'llama'
        elif any((k in name for k in ['opt-', 'galactica'])):
            model_type = 'opt'
        elif any((k in name for k in ['gpt-j', 'pygmalion-6b'])):
            model_type = 'gptj'
        else:
            print("Can't determine model type from model name. Please specify it manually using --model_type "
                  "argument")
            exit()
    else:
        model_type = args.model_type.lower()

    if model_type == 'llama' and args.pre_layer:
        load_quant = llama_inference_offload.load_quant
    elif model_type in ('llama', 'opt', 'gptj'):
        load_quant = _load_quant
    else:
        print("Unknown pre-quantized model type specified. Only 'llama', 'opt' and 'gptj' are supported")
        exit()

    #Now we are going to try to locate the quantized model file.
    path_to_model = Path(f'{args.model_dir}/{model_name}')
    LOG.debug(f"model path: {path_to_model}")
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]
    else:
        if path_to_model.name.lower().startswith('llama-7b'):
            pt_model = f'llama-7b-{args.wbits}bit'
        elif path_to_model.name.lower().startswith('llama-13b'):
            pt_model = f'llama-13b-{args.wbits}bit'
        elif path_to_model.name.lower().startswith('llama-30b'):
            pt_model = f'llama-30b-{args.wbits}bit'
        elif path_to_model.name.lower().startswith('llama-65b'):
            pt_model = f'llama-65b-{args.wbits}bit'
        else:
            pt_model = f'{model_name}-{args.wbits}bit'

        #Try to find the .safetensors or .pt both in models/ and in the subfolder
        for path in [Path(p+ext) for ext in ['.safetensors', '.pt'] for p in [f"models/{pt_model}", f"{path_to_model}/{pt_model}"]]:
            if path.exists():
                print(f"Found {path}")
                pt_path = path
                break

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

    # qwopqwop200's offload
    if args.pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), args.wbits, args.groupsize, args.pre_layer)
    else:
        LOG.info(f"{path_to_model}.. {pt_path}")
        model = load_quant(str(path_to_model), str(pt_path), args.wbits, args.groupsize)

        # accelerate offload (doesn't work properly)
        if args.gpu_memory:
            memory_map = list(map(lambda x : x.strip(), args.gpu_memory))
            max_cpu_memory = args.cpu_memory.strip() if args.cpu_memory else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory

            device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
            print("Using the following device map for the 4-bit model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        #No offload
        elif not args.cpu:
            model = model.to(torch.device('cuda:0'))

    return model

def load_quantized(model_name, args):

    path_to_model = Path(f'{args.model_dir}/{model_name}')
    LOG.debug(f"model path: {path_to_model}")

    model = AutoModelForCausalLM.from_pretrained(path_to_model, torch_dtype=torch.bfloat16)

    if not args.cpu:
        model = model.to(torch.device('cuda:0'))

    return model
