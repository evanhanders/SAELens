from typing import Any, cast

import wandb
from transformer_lens.hook_points import HookedRootModule

from sae_lens.training.activations_store import ActivationsStore, ToyActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sae_group import SparseAutoencoderDictionary
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.toy_models import ReluOutputModel, ToyConfig
from sae_lens.training.train_sae_on_language_model import (
    train_sae_group_on_language_model,
)


def sae_runner(
    cfg: LanguageModelSAERunnerConfig,
    model: HookedRootModule,
    activations_loader: ActivationsStore,
    sparse_autoencoder: SparseAutoencoderDictionary,
):
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)

    # train SAE
    train_sae_group_on_language_model(
        model,
        sparse_autoencoder,
        activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_window=cfg.feature_sampling_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
    )

    if cfg.log_to_wandb:
        wandb.finish()

    return sparse_autoencoder


def language_model_sae_runner(cfg: LanguageModelSAERunnerConfig):
    """ """

    if cfg.from_pretrained_path is not None:
        (
            model,
            sparse_autoencoder,
            activations_loader,
        ) = LMSparseAutoencoderSessionloader.load_pretrained_sae(
            cfg.from_pretrained_path
        )
        cfg = sparse_autoencoder.cfg
    else:
        loader = LMSparseAutoencoderSessionloader(cfg)
        (
            model,
            sparse_autoencoder,
            activations_loader,
        ) = loader.load_sae_training_group_session()

    return sae_runner(cfg, model, activations_loader, sparse_autoencoder)


def toy_model_sae_runner(cfg: LanguageModelSAERunnerConfig, model_cfg: ToyConfig):
    """ """

    # TODO: put model type into runner config
    model = ReluOutputModel(cfg=model_cfg)
    model.optimize()

    activations_loader = ToyActivationsStore.from_config(model, cfg)
    sparse_autoencoder = SparseAutoencoderDictionary(cfg)

    return model, sae_runner(cfg, model, activations_loader, sparse_autoencoder)
