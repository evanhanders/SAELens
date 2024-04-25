from collections.abc import Iterable
from math import ceil

import pytest
import torch
from datasets import Dataset, IterableDataset
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import LMActivationsStore, ToyActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.toy_models import Config, ReluOutputModel
from tests.unit.helpers import build_sae_cfg, load_model_cached


def tokenize_with_bos(model: HookedTransformer, text: str) -> list[int]:
    assert model.tokenizer is not None
    assert model.tokenizer.bos_token_id is not None
    return [model.tokenizer.bos_token_id] + model.tokenizer.encode(text)


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
            "prepend_bos": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.attn.hook_z",
            "hook_point_layer": 1,
            "d_in": 64,
            "prepend_bos": True,
        },
        {
            "model_name": "gelu-2l",
            "dataset_path": "NeelNanda/c4-tokenized-2b",
            "tokenized": True,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 512,
            "prepend_bos": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
            "tokenized": True,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 768,
            "prepend_bos": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "Skylion007/openwebtext",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 768,
            "prepend_bos": True,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-attn-out",
        "gelu-2l-tokenized",
        "gpt2-tokenized",
        "gpt2",
    ],
)
def cfg(request: pytest.FixtureRequest) -> LanguageModelSAERunnerConfig:
    # This function will be called with each parameter set
    params = request.param
    return build_sae_cfg(**params)


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig):
    return load_model_cached(cfg.model_name)


@pytest.fixture(
    params=[
        {
            "n_instances": 1,
            "n_features": 5,
            "n_hidden": 2,
        },
        pytest.param(
            {
                "n_instances": 2,
                "n_features": 5,
                "n_hidden": 2,
            },
            marks=pytest.mark.xfail(
                reason="Multiple toy model instances not supported yet in ToyActivationsStore."
            ),
        ),
    ],
    ids=[
        "toy-1x5x2",
        "toy-2x5x2",
    ],
)
def toy_cfg(request: pytest.FixtureRequest) -> Config:
    params = request.param
    return Config(**params)


@pytest.fixture
def toy_model(toy_cfg: Config):
    return ReluOutputModel(toy_cfg)


@pytest.fixture
def toy_sae_config():
    return LanguageModelSAERunnerConfig(
        hook_point="hook_hidden",
        hook_point_eval="hook_hidden",
        dtype=torch.float32,
        cached_activations_path=None,
    )


def test_toy_activations_store__shapes_look_correct(
    toy_cfg: Config,
    toy_model: ReluOutputModel,
    toy_sae_config: LanguageModelSAERunnerConfig,
):
    toy_sae_config.d_in = toy_cfg.n_hidden
    store = ToyActivationsStore.from_config(toy_model, toy_sae_config)
    assert store.model == toy_model
    assert store.d_in == toy_cfg.n_hidden

    # 50% is stored in buffer, the rest is in the dataloader.
    expected_size = (
        toy_sae_config.store_batch_size * toy_sae_config.n_batches_in_buffer // 2
    )
    assert store.storage_buffer.shape == (expected_size, 1, store.d_in)

    # --- Next, get batch tokens and assert they look correct ---

    batch = store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (store.store_batch_size, toy_cfg.n_features)
    assert batch.device == store.device

    # --- Next, get activations and assert they look correct ---

    activations = store.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        store.store_batch_size,
        1,  # a toy model could have multiple layers
        store.d_in,
    )
    assert activations.device == store.device

    # --- Next, get buffer and assert it looks correct ---

    n_batches_in_buffer = 3
    buffer = store.get_buffer(n_batches_in_buffer)

    assert isinstance(buffer, torch.Tensor)
    buffer_size_expected = store.store_batch_size * n_batches_in_buffer

    assert buffer.shape == (buffer_size_expected, 1, store.d_in)
    assert buffer.device == store.device


# tests involving loading real models / real datasets are very slow
# so do lots of stuff in this one test to make each load of model / data count
def test_activations_store__shapes_look_correct_with_real_models_and_datasets(
    cfg: LanguageModelSAERunnerConfig, model: HookedTransformer
):
    # --- first, test initialisation ---

    store: LMActivationsStore = LMActivationsStore.from_config(model, cfg)  # type: ignore

    assert store.model == model

    assert isinstance(store.dataset, IterableDataset)
    assert isinstance(store.iterable_dataset, Iterable)

    # the rest is in the dataloader.
    expected_size = (
        cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer // 2
    )
    assert store.storage_buffer.shape == (expected_size, 1, cfg.d_in)

    # --- Next, get batch tokens and assert they look correct ---

    batch = store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (
        store.store_batch_size,
        store.context_size,
    )
    assert batch.device == store.device

    # --- Next, get activations and assert they look correct ---

    activations = store.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        store.store_batch_size,
        store.context_size,
        1,
        store.d_in,
    )
    assert activations.device == store.device

    # --- Next, get buffer and assert it looks correct ---

    n_batches_in_buffer = 3
    buffer = store.get_buffer(n_batches_in_buffer)

    assert isinstance(buffer, torch.Tensor)
    buffer_size_expected = (
        store.store_batch_size * store.context_size * n_batches_in_buffer
    )

    assert buffer.shape == (buffer_size_expected, 1, store.d_in)
    assert buffer.device == store.device


def test_activations_store__get_activations_head_hook(ts_model: HookedTransformer):
    cfg = build_sae_cfg(
        hook_point="blocks.0.attn.hook_q",
        hook_point_head_index=2,
        hook_point_layer=1,
        d_in=4,
    )
    activation_store_head_hook = LMActivationsStore.from_config(ts_model, cfg)
    batch = activation_store_head_hook.get_batch_tokens()
    activations = activation_store_head_hook.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        activation_store_head_hook.store_batch_size,
        activation_store_head_hook.context_size,
        1,
        activation_store_head_hook.d_in,
    )
    assert activations.device == activation_store_head_hook.device


# 12 is divisible by the length of "hello world", 11 and 13 are not
@pytest.mark.parametrize("context_size", [11, 12, 13])
def test_activations_store__get_batch_tokens__fills_the_context_separated_by_bos(
    ts_model: HookedTransformer, context_size: int
):
    assert ts_model.tokenizer is not None
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )
    cfg = build_sae_cfg(
        store_batch_size=2,
        context_size=context_size,
    )

    activation_store = LMActivationsStore.from_config(ts_model, cfg, dataset=dataset)
    encoded_text = tokenize_with_bos(ts_model, "hello world")
    tokens = activation_store.get_batch_tokens()
    assert tokens.shape == (2, context_size)  # batch_size x context_size
    all_expected_tokens = (encoded_text * ceil(2 * context_size / len(encoded_text)))[
        : 2 * context_size
    ]
    expected_tokens1 = all_expected_tokens[:context_size]
    expected_tokens2 = all_expected_tokens[context_size:]
    if expected_tokens2[0] != ts_model.tokenizer.bos_token_id:
        expected_tokens2 = [ts_model.tokenizer.bos_token_id] + expected_tokens2[:-1]
    assert tokens[0].tolist() == expected_tokens1
    assert tokens[1].tolist() == expected_tokens2


def test_activations_store__get_next_dataset_tokens__tokenizes_each_example_in_order(
    ts_model: HookedTransformer,
):
    cfg = build_sae_cfg()
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = LMActivationsStore.from_config(ts_model, cfg, dataset=dataset)

    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world1"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world2"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world3"
    )
