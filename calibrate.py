import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import accelerate
import accelerate.hooks
import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.logging
import seaborn as sns
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

import utils

print = rich.print


@dataclass
class Args:
    model_id: str = 'meta-llama/Llama-2-7b-hf'
    model_load_dtype: str = field(default='float32', metadata={'choices': ['float32', 'float16', 'bfloat16']})
    target_sparsity_config: str = field(default='', metadata={
        "help": '''
        The configuration consists of one or more triplets, separated by commas. Each triplet includes:
            1. layer type: Type of the linear layer; here we accept "up"/"gate"/"down" for the FFN layers in LLM.
            2. demode method: Method to do mode centering:
                "zero": do not apply mode cenetering, i.e., mode=0;
                "median": use the activation median as the mode;
                "kde": estimate the KDE peak as the mode.
            3. target sparsity: Target sparsity value in the range (0, 1)
        Example:
            "up,zero,0.3,gate,zero,0.3,down,median,0.7" means conducting 30% target sparsity on up/gate projectors
                without mode centering, and 70% target sparsity on down projectors with median value as mode.
        '''
    })
    calibration_file_path: str = field(default=None, metadata={
        'help': 'A json file containing a list of strings. If not specified, will use allenai/c4 by default.'
    })
    n_calibration_samples: int = 64
    calibrated_thresholds_json_path: str = field(
        default='./calibrated_thresholds.json',
        metadata={'help': 'The output path of calibrated thresholds that meet the target sparsity config.'}
    )


def parse_target_sparsity_config(args: Args) -> dict:
    config_by_layer_type = {}
    fields = args.target_sparsity_config.replace(' ', '').split(',')
    assert len(fields) % 3 == 0
    for item in [fields[i:i + 3] for i in range(0, len(fields), 3)]:
        layer_type, demode_method, target_sparsity = item
        config_by_layer_type[layer_type] = dict(
            layer_type=layer_type,
            target_sparsity=float(target_sparsity),
            demode_method=demode_method,
        )
    config_dict = {}
    model_config = AutoConfig.from_pretrained(args.model_id)
    for layer_idx in range(model_config.num_hidden_layers):
        for layer_type in config_by_layer_type:
            layer_name = utils.infer_module_name(args.model_id, layer_idx, layer_type)
            config_dict[layer_name] = dict(**config_by_layer_type[layer_type])
    return config_dict


class SCAPCalibrationHook(accelerate.hooks.ModelHook):
    threshold_info: dict[str, utils.ThresholdDict] = {}

    def __init__(
        self, module_name: str = None,
        target_sparsity: float = 0.0, demode_method: str = 'zero',
    ) -> None:
        self.module_name = module_name
        self.target_sparsity = target_sparsity
        self.demode_method = demode_method

        self.is_calibrated = False
        self.mode = None
        self.threshold = None
        self.extra_bias = None

    def pre_forward(self, module: nn.Linear, *args, **kwargs):
        # avoid impacting other layers that references this tensor, e.g., up/gate share the same inputs
        x = args[0].clone()
        flag = False
        if not self.is_calibrated:
            self._calibrate(module, x)
            flag = True
        shifted_sparsified_x = self._get_sparse_shifted_x(x, self.mode, self.threshold)

        if flag:
            actual_calibrated_sparsity = (shifted_sparsified_x == 0).float().mean().item()
            if abs(actual_calibrated_sparsity - self.target_sparsity) > 0.1:
                print(f'WARNING: actual_calibrated_sparsity={actual_calibrated_sparsity}')
            print(dict(
                **self.__class__.threshold_info[self.module_name],
                actual_calibrated_sparsity=actual_calibrated_sparsity,
                calibration_x_shape=list(x.shape),
                calibration_x_dtype=x.dtype,
                calibration_x_device=x.device,
            ))
        return (shifted_sparsified_x,), kwargs

    def post_forward(self, module, output):
        assert self.is_calibrated is True
        if self.extra_bias is not None:
            output = output + self.extra_bias.to(dtype=output.dtype, device=output.device)
        return output

    def _calibrate(self, module: nn.Linear, x: torch.Tensor):
        candidate_modes = self._calc_candidate_modes(x, demode_method=self.demode_method)
        if len(candidate_modes) == 1:
            mode = candidate_modes[0]
        else:
            dense_outputs = F.linear(x, module.weight)

            def get_similarity(candidate_mode: float):
                threshold = self._calc_threshold(x, candidate_mode, self.target_sparsity)
                sparse_x = self._get_sparse_shifted_x(x, candidate_mode, threshold) + candidate_mode
                sparse_outputs = F.linear(sparse_x, module.weight)
                return torch.dist(dense_outputs, sparse_outputs, p=2).item()

            modes_pairs = sorted(zip(map(get_similarity, candidate_modes), candidate_modes))
            mode = modes_pairs[0][1]
            print('KDE debugging:', modes_pairs)
        threshold = self._calc_threshold(x, mode, target_sparsity=self.target_sparsity)

        self.mode = mode
        self.threshold = threshold
        if abs(mode) > 1e-6:
            self.extra_bias = F.linear(
                torch.ones([1, x.shape[-1]], dtype=x.dtype, device=x.device) * mode,
                module.weight
            ).cpu()
        self.is_calibrated = True
        assert self.module_name not in self.__class__.threshold_info
        self.__class__.threshold_info[self.module_name] = dict(
            module_name=self.module_name,
            demode_method=self.demode_method,
            mode=self.mode,
            threshold=self.threshold,
            target_sparsity=self.target_sparsity,
            hook_type='pre_hook',
        )
        return mode, threshold

    def _quantile(self, x: torch.Tensor, q: float) -> float:
        x = x.detach().cpu().view(-1).to(torch.float64).numpy()
        return np.quantile(x, q=q, keepdims=False).item()

    def _calc_candidate_modes(self, x: torch.Tensor, demode_method: str) -> list[float]:
        if demode_method == 'zero':
            return [0.]
        elif demode_method == 'median':
            return [x.median().item()]
        elif demode_method == 'kde':
            np.random.seed(42)

            def get_sampled(x, N):
                x = x.view(-1)
                sep = (x.numel() // N)
                return x[::sep]
            candidates = []
            for N in [100000, 200000, 300000]:
                samples = get_sampled(x.cpu(), N).numpy()
                plt.ioff()
                xy = sns.kdeplot(samples, bw_adjust=0.5, gridsize=2000, cut=0).get_lines()[0].get_xydata()
                mode = float(xy[np.argmax(xy[:, 1]), 0])
                plt.close()
                candidates.append(mode)
            return candidates

    def _calc_threshold(self, x: torch.Tensor, mode: float, target_sparsity: float):
        x = x.view(-1)
        shifted_abs = (x - mode).abs()
        assert x.dtype == shifted_abs.dtype
        return self._quantile(shifted_abs, q=target_sparsity)

    def _get_sparse_shifted_x(self, x: torch.Tensor, mode: float, threshold: float):
        x_shifted = x - mode
        mask = torch.le(x_shifted.abs(), threshold)
        result = torch.masked_fill(x_shifted, mask, value=0.)
        assert result.dtype == x.dtype
        return result


@torch.no_grad()
def add_scap_calibration_hook(model: nn.Module, target_sparsity_config: dict):
    for module_name, module in model.named_modules():
        if module_name in target_sparsity_config:
            config = target_sparsity_config.pop(module_name)
            hook = SCAPCalibrationHook(
                module_name=module_name,
                target_sparsity=config['target_sparsity'],
                demode_method=config['demode_method'],
            )
            accelerate.hooks.add_hook_to_module(module, hook)
            print(f'Adding hook at {module_name}\t: {config}')
    assert len(target_sparsity_config) == 0, f'Unprocessed layers: {list(target_sparsity_config.keys())}'
    return model


@torch.no_grad()
def main(args: Args):
    print(args)
    if Path(args.calibrated_thresholds_json_path).exists():
        print(f'Calibrated thresholds already exist at {args.calibrated_thresholds_json_path}')
        return

    set_seed(42)
    Path(args.calibrated_thresholds_json_path).parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=getattr(torch, args.model_load_dtype),
        device_map='cpu',
        trust_remote_code=True,
    ).eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if args.calibration_file_path is not None:
        with open(args.calibration_file_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
    else:
        print('Use calibration set from "allenai/c4".')
        texts = utils.get_calibration_texts()
    input_ids = tokenizer(
        texts[:args.n_calibration_samples], truncation=True,
        return_tensors='pt', max_length=256, padding=False,
    )['input_ids']

    target_sparsity_config = parse_target_sparsity_config(args)
    model = add_scap_calibration_hook(model, target_sparsity_config)
    print('Starting calibration...')
    model(input_ids)

    with Path(args.calibrated_thresholds_json_path).open('w', encoding='utf-8') as f:
        json.dump(SCAPCalibrationHook.threshold_info, f, indent=2)


if __name__ == '__main__':
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    main(args)
