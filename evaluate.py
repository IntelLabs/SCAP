import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import rich
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

import utils

print = rich.print


@dataclass
class Args:
    model_id: str = 'meta-llama/Llama-2-7b-hf'
    model_load_dtype: str = field(default='float32', metadata={'choices': ['float32', 'float16', 'bfloat16']})
    calibrated_thresholds_json_path: str = field(default='')
    evaluation_tasks: str = field(
        default='winogrande,piqa,sciq,hellaswag,boolq,arc_easy,arc_challenge',
        metadata={'help': 'Comma-separated list of task names to evaluate on with zero shot.'})
    evaluation_limit: int = field(
        default=None,
        metadata={'help': 'If specified, will only evaluate on this many samples. For debugging purpose.'}
    )
    evaluation_results_json_path: str = field(
        default='./evaluation_results.json',
        metadata={'help': 'The output path of evaluation results.'}
    )


class SCAPLinear(nn.Module):
    sparsity_info = defaultdict(utils.AverageMeter)

    def __init__(self, linear: nn.Linear, mode: float, threshold: float, layer_name: str = None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.mode = mode
        self.threshold = threshold
        extra_bias = F.linear(
            torch.ones([1, self.in_features], dtype=self.weight.dtype, device=self.weight.device) * self.mode,
            self.weight
        ).reshape(-1)
        self.bias = extra_bias if self.bias is None else self.bias + extra_bias
        self.layer_name = layer_name or f'scap_linear_id_{id(self)}'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_shifted: torch.Tensor = input - self.mode
        mask = torch.le(x_shifted.abs(), self.threshold)
        sparse_input = x_shifted.masked_fill_(mask, value=0.)

        sparsity = mask.float().mean().item()
        self.__class__.sparsity_info[self.layer_name].update(sparsity)
        return F.linear(sparse_input, self.weight, self.bias)


def apply_scap(model, calibrated_thresholds: dict[str, utils.ThresholdDict]):
    for layer_name, sparse_config in calibrated_thresholds.items():
        old_linear = utils.get_layer(model, layer_name)
        assert isinstance(old_linear, nn.Linear)
        new_linear = SCAPLinear(
            old_linear, mode=sparse_config['mode'], threshold=sparse_config['threshold'],
            layer_name=layer_name,
        )
        utils.replace_layer(model, layer_name, new_linear)
    return model


def main(args: Args):
    print(args)
    if Path(args.evaluation_results_json_path).exists():
        print(f'Evaluation results already exist at: {args.evaluation_results_json_path}')
        return

    set_seed(42)
    Path(args.evaluation_results_json_path).parent.mkdir(parents=True, exist_ok=True)
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
    model = apply_scap(model, calibrated_thresholds)

    result = {'args': args.__dict__}
    tasks = args.evaluation_tasks.split(',')
    result['evaluation'] = utils.run_lm_eval(
        model, tokenizer, args.model_id,
        device='cuda', tasks=tasks, limit=args.evaluation_limit
    )
    result['sparsity'] = {k: v.avg for k, v in SCAPLinear.sparsity_info.items()}
    with Path(args.evaluation_results_json_path).open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == '__main__':
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    main(args)
