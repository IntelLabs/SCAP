from typing import Optional, Union

import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import AutoCausalLM, HuggingFaceAutoLM
from transformers import AutoTokenizer, PreTrainedModel


class LMEvalModel(AutoCausalLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device: Optional[Union[int, str]] = "cuda",
    ):
        super(HuggingFaceAutoLM, self).__init__()  # do the BaseLM init
        self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = model.config

        self._add_special_tokens = add_special_tokens
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.max_length

        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)
        self._device = device


def run_lm_eval(model, tokenizer, model_id: str, device: str, tasks: list[str], limit=None):
    tokenizer.pad_token = tokenizer.eos_token
    max_length = None
    if any([
        'llama-3' in model_id.lower(),
        'mistral' in model_id.lower(),
        'mixtral' in model_id.lower(),
    ]):
        max_length = LMEvalModel._DEFAULT_MAX_LENGTH
        print(f'Manually setting max_length={max_length} for {model_id} to avoid potential OOM.')

    lm_eval_model = LMEvalModel(model, tokenizer, batch_size=1, max_length=max_length, device=device)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        no_cache=True,
        limit=limit,
        device=device,
    )
    return results


class AverageMeter:
    def __init__(self):
        self.last_val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.last_val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
