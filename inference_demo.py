import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from flash_gemv import gather_transposed_gemv_flag_3d as scap_fc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TextStreamer,
    set_seed,
)

import utils


@dataclass
class Args:
    model_id: str = 'meta-llama/Llama-2-7b-hf'
    model_load_dtype: str = field(default='float32', metadata={'choices': ['float32']})
    calibrated_thresholds_json_path: str = field(
        default='results/scap/meta-llama/Llama-2-7b-hf/up,zero,0.35,gate,zero,0.35,down,zero,0.55/calibrated_thresholds.json'
    )
    prompt: str = field(default='Once upon a time, there lived')


class SCAPLinearRealSparse(nn.Module):
    def __init__(self, linear: nn.Linear, mode: float, threshold: float, layer_name: str = None):
        super().__init__()
        assert linear.weight.dtype == torch.float32, 'SCAP kerenl only supports float32 for now.'
        self.layer_name = layer_name or f'scap_linear_id_{id(self)}'
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_t = linear.weight.t().contiguous().data
        self.bias = linear.bias
        self.decode_bias = linear.bias
        self.mode = mode
        self.threshold = threshold

        if abs(mode) < 1e-7:
            self.decode_forward_fn = self.decode_forward_no_mode_no_bias \
                if self.decode_bias is None else self.decode_forward_no_mode_with_bias
        else:
            extra_bias = F.linear(
                torch.ones([1, self.in_features], dtype=linear.weight.dtype, device=linear.weight.device) * self.mode,
                linear.weight
            ).reshape(-1)
            self.decode_bias = extra_bias if self.decode_bias is None else self.decode_bias + extra_bias
            self.decode_bias = self.decode_bias.contiguous()
            self.decode_forward_fn = self.decode_forward_demode_with_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.prod(torch.tensor(x.shape[:-1])) == 1:
            return self.decode_forward_fn(x)
        return F.linear(x, self.weight_t.t(), self.bias)

    def decode_forward_no_mode_no_bias(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.abs() > self.threshold
        return scap_fc(x, self.weight_t, mask)

    def decode_forward_no_mode_with_bias(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_forward_no_mode_no_bias(x) + self.decode_bias

    def decode_forward_demode_with_bias(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_forward_no_mode_no_bias(x - self.mode) + self.decode_bias


def apply_scap_real_sparse(model, calibrated_thresholds: dict[str, utils.ThresholdDict]):
    for layer_name, sparse_config in calibrated_thresholds.items():
        old_linear = utils.get_layer(model, layer_name)
        assert isinstance(old_linear, nn.Linear)
        new_linear = SCAPLinearRealSparse(
            old_linear, mode=sparse_config['mode'], threshold=sparse_config['threshold'],
            layer_name=layer_name,
        )
        utils.replace_layer(model, layer_name, new_linear)
    return model


def main(args: Args):
    print(args)
    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=getattr(torch, args.model_load_dtype),
        device_map='cuda',
        trust_remote_code=True,
    ).eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    with Path(args.calibrated_thresholds_json_path).open('r', encoding='utf-8') as f:
        calibrated_thresholds: dict[str, utils.ThresholdDict] = json.load(f)
    model = apply_scap_real_sparse(model, calibrated_thresholds)
    print(model)
    pipeline = transformers.pipeline(
        'text-generation', model=model, tokenizer=tokenizer, torch_dtype=getattr(torch, args.model_load_dtype)
    )
    streamer = TextStreamer(tokenizer)
    print('Start to generate text. It would be slow at the beginning because of triton compiling.')
    if args.prompt:
        pipeline(args.prompt, max_new_tokens=100, streamer=streamer)[0]
    while True:
        prompt = input('\nEnter a prompt (`exit` to stop): ').strip()
        if prompt == 'exit' or prompt == '':
            return
        pipeline(prompt, max_new_tokens=100, streamer=streamer)[0]


if __name__ == '__main__':
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
