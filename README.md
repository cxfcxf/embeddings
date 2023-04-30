# embeddings
This binds to redis stack server for presistent storage, so if you dont have redis running, it will not work

# pre-requests
You will need cuda extention installed which be compatiable with pytorch version 2.0.0

# Usage
the basic idea of this is to load a gptq model and run embedding against it instead of requiring openAI connection

the gptq model is 4 bits with 128 group size model which loses some precision but allow you to fit a larger model in VRAM, for reference [GTPQ](https://arxiv.org/pdf/2210.17323.pdf)

# Example chat of stat of the union
![Chat Example](https://github.com/cxfcxf/embeddings/blob/main/example.png?raw=true)

## storing documents into vector store
```bash
❯ python embeddings.py --index-name state_of_the_union store --docs state_of_the_union.txt
INFO    - Loading encoding model sentence-transformers/all-MiniLM-L6-v2...
INFO    - Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
INFO    - Storing vector data to redis...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.38it/s]
INFO    - Index already exists
INFO    - finished, exiting...
```

--docs take multiple files, and currently only does txt and pdf

## loading vector store with model
```bash
❯ python embeddings.py --index-name state_of_the_union run --model-dir /home/siegfried/model-gptq --model-name vicuna-13b-4bits
INFO    - Loading encoding model sentence-transformers/all-MiniLM-L6-v2...
INFO    - Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
INFO    - Index already exists
INFO    - Loading Tokenizer from /home/siegfried/model-gptq/vicuna-13b-4bits...
INFO    - Loading the model from /home/siegfried/model-gptq/vicuna-13b-4bits...
INFO    - Loading gptq quantized models...
WARNING - CUDA extension not installed.
INFO    - creating transformer pipeline...
The model 'LlamaGPTQForCausalLM' is not supported for text-generation. Supported models are ['BartForCausalLM', 'BertLMHeadModel', 'BertGenerationDecoder', 'BigBirdForCausalLM', 'BigBirdPegasusForCausalLM', 'BioGptForCausalLM', 'BlenderbotForCausalLM', 'BlenderbotSmallForCausalLM', 'BloomForCausalLM', 'CamembertForCausalLM', 'CodeGenForCausalLM', 'CpmAntForCausalLM', 'CTRLLMHeadModel', 'Data2VecTextForCausalLM', 'ElectraForCausalLM', 'ErnieForCausalLM', 'GitForCausalLM', 'GPT2LMHeadModel', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 'GPTNeoForCausalLM', 'GPTNeoXForCausalLM', 'GPTNeoXJapaneseForCausalLM', 'GPTJForCausalLM', 'LlamaForCausalLM', 'MarianForCausalLM', 'MBartForCausalLM', 'MegaForCausalLM', 'MegatronBertForCausalLM', 'MvpForCausalLM', 'OpenAIGPTLMHeadModel', 'OPTForCausalLM', 'PegasusForCausalLM', 'PLBartForCausalLM', 'ProphetNetForCausalLM', 'QDQBertLMHeadModel', 'ReformerModelWithLMHead', 'RemBertForCausalLM', 'RobertaForCausalLM', 'RobertaPreLayerNormForCausalLM', 'RoCBertForCausalLM', 'RoFormerForCausalLM', 'Speech2Text2ForCausalLM', 'TransfoXLLMHeadModel', 'TrOCRForCausalLM', 'XGLMForCausalLM', 'XLMWithLMHeadModel', 'XLMProphetNetForCausalLM', 'XLMRobertaForCausalLM', 'XLMRobertaXLForCausalLM', 'XLNetLMHeadModel', 'XmodForCausalLM'].
INFO    - creating chain...
INFO    - Loading Q&A chain...
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

# About loading models
there are two way of loading models, normally its expecting a model which is quantized with GPTQ 4bits 128group_size

you can specify --no-gptq, it would load model normally (you can prob fit a 13b model with 8 bits for 24GB VRAM)

# About converting model to 4bits
Please use [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) to quantlized it, a 13b model will need about 35GB DRAM

# Current issues
* it does seems the gptq quantized model done by [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) which from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) is not very performant, somehow the old cuda branch [GPTQ-for-LLaMa](https://github.com/oobabooga/GPTQ-for-LLaMa/) is performant. but AutoGPTQ makes it really easy to use, so i stick with that

^ the issue is solved by force the model load with `use_triton=True` which loads the whole model into VRAM
