import torch.nn as nn


def infer_module_name(model_id: str, layer_idx: int, layer_type: str, expert_id: int | None = None):
    model_id = model_id.lower()
    if '/' in model_id:
        author, model_id = model_id.split('/')[-2:]
    else:
        author, model_id = '', model_id
    layer_type = layer_type.lower()
    if author == 'state-spaces' and model_id.startswith('mamba2-'):
        if layer_type in ['in', 'out']:
            return f'backbone.layers.{layer_idx}.mixer.{layer_type}_proj'
    if 'mpt' in model_id:
        if layer_type in ['up', 'down']:
            return f'transformer.blocks.{layer_idx}.ffn.{layer_type}_proj'
        if layer_type in ['act']:
            return f'transformer.blocks.{layer_idx}.ffn.act'
    if 'llama' in model_id or 'mistral' in model_id or 'gemma' in model_id:
        if layer_type in ['q', 'k', 'v', 'o']:
            return f'model.layers.{layer_idx}.self_attn.{layer_type}_proj'
        if layer_type in ['up', 'gate', 'down']:
            return f'model.layers.{layer_idx}.mlp.{layer_type}_proj'
        if layer_type in ['act']:
            return f'model.layers.{layer_idx}.mlp.act_fn'
    if 'mixtral' in model_id:
        if layer_type in ['q', 'k', 'v', 'o']:
            return f'model.layers.{layer_idx}.self_attn.{layer_type}_proj'
        if layer_type == 'up':
            return f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w3'
        if layer_type == 'gate':
            return f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w1'
        if layer_type == 'down':
            return f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w2'
        if layer_type == 'act':
            return f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.act_fn'
    if 'falcon' in model_id:
        if layer_type in ['fc1', 'up']:
            return f'transformer.h.{layer_idx}.mlp.dense_h_to_4h'
        if layer_type in ['fc2', 'down']:
            return f'transformer.h.{layer_idx}.mlp.dense_4h_to_h'
        if layer_type in ['act']:
            return f'transformer.h.{layer_idx}.mlp.act'
    if 'timm' in author and 'deit' in model_id:
        if layer_type == 'o':
            return f'blocks.{layer_idx}.attn.proj'
        if layer_type == 'qkv':
            return f'blocks.{layer_idx}.attn.qkv'
        if layer_type == 'up':
            return f'blocks.{layer_idx}.mlp.fc1'
        if layer_type == 'down':
            return f'blocks.{layer_idx}.mlp.fc2'
    raise NotImplementedError(model_id, layer_type, layer_idx, expert_id)


def replace_layer(model: nn.Module, layer_name: str, new_layer: nn.Module):
    names = layer_name.split('.')
    parent_module = model
    for name in names[:-1]:
        if name.isdigit():
            parent_module = parent_module[int(name)]
        else:
            parent_module = getattr(parent_module, name)
    if names[-1].isdigit():
        parent_module[int(names[-1])] = new_layer
    else:
        setattr(parent_module, names[-1], new_layer)
    return model


def get_layer(model: nn.Module, layer_name: str):
    names = layer_name.split('.')
    target_layer = model
    for name in names:
        if name.isdigit():
            target_layer = target_layer[int(name)]
        else:
            target_layer = getattr(target_layer, name)
    return target_layer
