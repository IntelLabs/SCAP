from typing import Callable, List, Union

import torch

import datasets
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def get_calibration_texts(num_examples: int = 512):
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    texts = []

    for data in tqdm(dataset):
        inputs = llama_tokenizer(data['text'], add_special_tokens=False)
        input_ids = inputs['input_ids']
        if not 384 <= len(input_ids) <= 512:
            continue
        texts.append(data['text'])
        if len(texts) >= num_examples:
            break

    return texts


def get_input_texts_of_given_length(
    input_ids_getter: Callable[[Union[str, List[str]]], Union[List[int], List[List[str]]]],
    batch_size: int,
    seq_length: int,
    seed: int = 42,
) -> List[str]:
    """
    Generates a list of dummy input texts that match the specified sequence length.
    Args:
        input_ids_getter (Callable[[Union[str, List[str]]], Union[List[int], List[List[str]]]]): 
            A function that converts text or a list of texts into input IDs.
        batch_size (int): 
            The number of dummy input texts to generate.
        seq_length (int): 
            The desired sequence length for each dummy input text.
        seed (int, optional): 
            The random seed for reproducibility. Defaults to 42.
    Returns:
        List[str]:
            A list of dummy input texts, each with the specified sequence length.
    """

    dummy_input_ids = input_ids_getter('42')
    if isinstance(dummy_input_ids, torch.Tensor):
        _ori_input_ids_getter = input_ids_getter
        def input_ids_getter(text): return _ori_input_ids_getter(text).cpu().tolist()
    assert isinstance(input_ids_getter('42'), list)

    dataset = datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-v1', split='test')
    dataset = dataset.train_test_split(
        test_size=max(0.001, np.random.default_rng(seed).random()),
        seed=seed, shuffle=False,
    )
    dataset_iter = iter(datasets.concatenate_datasets([dataset['test'], dataset['train']]))

    texts: List[str] = []
    for _ in range(batch_size):
        success = False
        while not success:
            text = '<s> '
            while len(input_ids_getter(text)) < seq_length:
                new_text = next(dataset_iter)['text'].replace('\n\n', '\n').strip()
                while not new_text:
                    new_text = next(dataset_iter)['text'].replace('\n\n', '\n').strip()
                text = text + new_text + '\n'
            while len(input_ids_getter(text)) > seq_length and len(text) > 3:
                text = text[:-3]
            success = len(input_ids_getter(text)) == seq_length

        texts.append(text)

    input_ids = input_ids_getter(texts)
    assert len(input_ids) == batch_size
    for sample in input_ids:
        assert len(sample) == seq_length, f'{len(sample)} != {seq_length}'
    return texts
