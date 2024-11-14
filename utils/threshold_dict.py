from typing import Literal, TypedDict


class ThresholdDict(TypedDict):
    module_name: str
    demode_method: Literal['zero', 'kde', 'median']
    mode: float
    threshold: float
    target_sparsity: float
    hook_type: Literal['pre_hook', 'post_hook']
