"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Example usage:
python train.py --config path_to_your_config.yaml your.overrides=here

Or run with Lightning Fabric:
fabric run --strategy ddp --devices 4 --precision bf16-mixed train.py --config path_to_your_config.yaml your.overrides=here
"""

import datetime
import os
import secrets
import time
import torch
import wandb
import yaml
import lightning as L
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from lightning.fabric.loggers import CSVLogger
from wandb.integration.lightning.fabric import WandbLogger

from core_train import Trainer, initialize_model, shard_model
from data.loop_stargraph import LoopStarGraphDataModule

DATAMODULES = {
    "loop_stargraph": LoopStarGraphDataModule,
}


def do_train(
    config: DictConfig,
    hide_progress_bar: bool = False,
    use_sharding: bool = False,
    notes: str = None,
):
    # Build the run-specific output directory: <out_dir>/<notes>/<run_name>/
    # Where <run_name> is either an explicit trainer.run_name override, or auto-
    # generated as run_<YYYY-MM-DD_HH-MM-SS>_<short-hash>.
    notes_group = (config.trainer.notes or "run").replace("/", "_")
    if config.trainer.get("run_name", None):
        experiment_name = config.trainer.run_name
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_hash = secrets.token_hex(3)
        experiment_name = f"run_{stamp}_{short_hash}"
    experiment_out_dir = os.path.join(config.trainer.out_dir, notes_group, experiment_name)
    config.trainer.experiment_name = experiment_name
    config.trainer.run_dir = experiment_out_dir
    # Emit a machine-parseable line so callers (orchestrator, README recipes) can
    # capture the exact run directory without re-discovering it via globs.
    print(f"RUN_DIR={os.path.abspath(experiment_out_dir)}", flush=True)

    # Initialize logging
    loggers = []
    if config.trainer.log_to_file:
        loggers.append(
            CSVLogger(
                root_dir=os.path.join(config.trainer.out_dir, notes_group),
                name=experiment_name,
                flush_logs_every_n_steps=1,
            )
        )
    if config.trainer.log_to_wandb:
        # Ensure we are authenticated with Weights & Biases BEFORE creating the WandbLogger
        # so that the initial upsert/run creation doesn't fail with a 401.
        if "WANDB_API_KEY" in os.environ:
            # Login on every rank is harmless; wandb will only create a single run (others attach or noop).
            print("hello")
            wandb.login(
                key=os.environ.get("WANDB_API_KEY"),
                host=os.environ.get("WANDB_HOST"),
                timeout=0,
                relogin=True,
            )
        else:
            # If the user explicitly asked to log to wandb but provided no key, fail fast with a clear message.
            raise RuntimeError(
                "config.trainer.log_to_wandb is True but WANDB_API_KEY is not set in the environment."
            )
        # Pass notes directly so W&B sets them at init robustly
        wandb_logger = WandbLogger(
            project=config.trainer.wandb_project,
            name=experiment_name,
            notes=notes,
            settings=wandb.Settings(code_dir="."),  # Upload all code in the current directory recursively
        )
        loggers.append(wandb_logger)


    # Initialize Fabric
    if use_sharding:
        # Sharding must be defined before creating Fabric
        strategy = ModelParallelStrategy(
            parallelize_fn=shard_model, save_distributed_checkpoint=False
        )
        fabric = L.Fabric(loggers=loggers, strategy=strategy)
        fabric.print("Training with sharded model")
    else:
        # GPUs, parallelism, and precision are set by the "fabric run" command
        fabric = L.Fabric(loggers=loggers)

    # Calculate per device batch size
    assert (
        config.data.effective_batch_size % fabric.world_size == 0
    ), f"effective_batch_size {config.data.effective_batch_size} must be divisible by DDP world size {fabric.world_size}"
    config.data.device_batch_size = (
        config.data.effective_batch_size // fabric.world_size
    )

    # Calculate micro batch size
    assert (
        config.data.device_batch_size % config.data.gradient_accum_steps == 0
    ), f"device_batch_size {config.data.device_batch_size} must be divisible by gradient_accum_steps {config.data.gradient_accum_steps}"
    config.data.micro_batch_size = (
        config.data.device_batch_size // config.data.gradient_accum_steps
    )

    # Print config
    fabric.print(yaml.dump(OmegaConf.to_container(config)))

    if config.trainer.log_to_wandb and fabric.global_rank == 0:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb_logger.experiment.config.update(config_dict, allow_val_change=True)

    tokens_per_update = config.data.effective_batch_size * config.model.block_size
    fabric.print(f"World size: {fabric.world_size}")
    fabric.print(f"Effective batch size: {config.data.effective_batch_size}")
    fabric.print(f"Per-GPU device batch size: {config.data.device_batch_size}")
    fabric.print(
        f"Batch size per gradient accumulation step: {config.data.micro_batch_size}"
    )
    fabric.print(f"Tokens per gradient update: {tokens_per_update}")

    # Initialize PyTorch settings
    seed_offset = fabric.global_rank
    fabric.seed_everything(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 16  # allow more recompiles

    # Load dataset
    assert (
        config.data.dataset in DATAMODULES
    ), f"Dataset {config.data.dataset} not supported"
    DataModuleClass = DATAMODULES[config.data.dataset]
    datamodule = DataModuleClass(fabric, config)
    datamodule.update_config(config)
    tokenizer = datamodule.get_tokenizer()

    # Create output directory (already computed above as experiment_out_dir)
    if fabric.global_rank == 0:
        os.makedirs(experiment_out_dir, exist_ok=True)
    fabric.print(f"Output directory: {experiment_out_dir}")

    # dump config to out_dir
    if fabric.global_rank == 0:
        OmegaConf.save(
            config, os.path.join(experiment_out_dir, "materialized_config.yaml")
        )
    # (wandb login already occurred before WandbLogger creation)

    # Create model and trainer
    model = initialize_model(fabric, config, tokenizer, initialize_optimizer=True)
    trainer = Trainer(
        fabric=fabric,
        config=config,
        model=model,
        show_progress_bar=(fabric.local_rank == 0 and not hide_progress_bar),
    )

    # Start training
    trainer.train(datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="train",
        description="Train a GPT on synthetic star-graph data.",
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="Optional path to a YAML config that overrides config/defaults.yaml. "
             "If omitted, config/defaults.yaml + any CLI key=value overrides are used.",
    )
    parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar")
    parser.add_argument("--shard", action="store_true", help="Shard the model")
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="Optional Weights & Biases run notes to attach at init.",
    )
    args, conf_cli = parser.parse_known_args()

    default = OmegaConf.load("config/defaults.yaml")
    overrides = OmegaConf.load(args.config) if args.config else OmegaConf.create({})
    cli = OmegaConf.from_dotlist(conf_cli)

    config = OmegaConf.merge(default, overrides, cli)

    do_train(
        config,
        hide_progress_bar=args.no_pbar,
        use_sharding=args.shard,
        notes=args.wandb_notes,
    )
