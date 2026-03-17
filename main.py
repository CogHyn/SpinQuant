
import transformers
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
import torch

config = transformers.AutoConfig.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct"
)

print(config)

model = LlamaForCausalLMQuant.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
    config=config,
    torch_dtype=torch.bfloat16,
)

from train_utils.main import prepare_model

prepare_model(model)

print(model)