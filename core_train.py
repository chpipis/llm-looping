"""contains core training/inference logic used for multiple datasets/tasks"""

import gc
import inspect
import math
import os
import torch
import wandb
import lightning as L
from datetime import datetime
from dataclasses import asdict
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple

from model_base import ModelBase
from model_gpt import GPT, GPTConfig, Block


def initialize_model(
    fabric: L.Fabric,
    config,
    tokenizer,
    initialize_optimizer=True,
    checkpoint_path: Optional[str] = None,
):
    fabric.print(f"Initializing model with config: {config}")

    ModelClass = GPT
    ModelConfigClass = GPTConfig
    gpt_modes = ["next_token", "fim"]
    assert (
        config.model.gpt_mode in gpt_modes
    ), f"GPT mode must be one of {gpt_modes}"

    pad_token_id = tokenizer.pad_token_id
    if not 0 <= pad_token_id < config.model.vocab_size:
        raise ValueError("Tokenizer padding ID must be inside the model vocabulary")
    if pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Padding and EOS must use distinct token IDs")

    # start with model_args from command line
    model_args = dict(
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        block_size=config.model.block_size,
        bias=config.model.bias,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        loss_on_response_only=config.model.loss_on_response_only,
        use_fused=config.trainer.use_fused_kernels,
    )

    # GPT-specific parameters
    model_args = {
        **model_args,
        "fim_token_id": (
            -1
            if config.model.gpt_mode != "fim"
            else tokenizer.convert_tokens_to_ids("<|fim|>")
        ),
        "is_fim_mode": config.model.gpt_mode == "fim",
    }
    model_config = ModelConfigClass(**model_args)
    fabric.print(f"Effective GPTConfig: {asdict(model_config)}")

    # Load a checkpoint file
    if checkpoint_path:
        assert os.path.isfile(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist"

        # Create an empty model because we will load the weights from checkpoint
        with fabric.init_module(empty_init=True):
            model = ModelClass(model_config)

    # Resume from a previous training run
    elif config.trainer.init_from == "resume":
        # Sentinels live inside the run-specific dir (config.trainer.run_dir),
        # not the parent out_dir, so concurrent runs never collide. The caller
        # must point at the existing run via trainer.run_name=<name>.
        recovery_ckpt_pointer = os.path.join(config.trainer.run_dir, "recovery_ckpt")
        latest_ckpt_pointer = os.path.join(config.trainer.run_dir, "latest_ckpt")

        # Use the recovery checkpoint if it exists
        if os.path.isfile(recovery_ckpt_pointer):
            with open(recovery_ckpt_pointer, "r") as f:
                checkpoint_path = f.read().strip()
            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file {checkpoint_path} does not exist"
            fabric.print(f"Resuming from recovery checkpoint {checkpoint_path}")

        # Otherwise, use the latest validation checkpoint if it exists
        elif os.path.isfile(latest_ckpt_pointer):
            with open(latest_ckpt_pointer, "r") as f:
                checkpoint_path = f.read().strip()
            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file {checkpoint_path} does not exist"
            fabric.print(
                f"Resuming from previous validation checkpoint {checkpoint_path}"
            )

        # If no checkpoint file is found, initialize a new model
        else:
            fabric.print(f"Could not find checkpoint file {recovery_ckpt_pointer}")
            fabric.print(f"Could not find checkpoint file {latest_ckpt_pointer}")
            checkpoint_path = None

        # Empty init if we have found a checkpoint
        with fabric.init_module(empty_init=(checkpoint_path is not None)):
            model = ModelClass(model_config)

    # Initialize a new model from scratch
    elif config.trainer.init_from == "scratch":
        with fabric.init_module():
            model = ModelClass(model_config)

    # Initialize from OpenAI GPT-2 weights
    elif config.trainer.init_from.startswith("gpt2"):  # currently broken @dayan
        fabric.print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
        override_args = dict(dropout=config.model.dropout)
        model = ModelClass.from_pretrained(config.trainer.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)

    else:
        raise ValueError(
            f"Invalid init_from value: {config.trainer.init_from}. Must be 'resume', 'scratch', or 'gpt2'."
        )

    # crop down the model block size if desired, using model surgery
    # so that the checkpoint will have the right value
    if config.model.block_size < model.config.block_size:
        model.crop_block_size(config.model.block_size)
        model_args["block_size"] = config.model.block_size

    fabric.print(
        f"Number of parameters (including embedding): {model.get_num_params(non_embedding=False):,}"
    )
    fabric.print(
        f"Number of parameters (excluding embedding): {model.get_num_params(non_embedding=True):,}"
    )

    # Compile must occur before fabric.setup()
    if config.trainer.compile:
        fabric.print("Compiling model")
        model.compile()

    # FSDP requires model to be setup before initializing the optimizer
    if isinstance(fabric.strategy, ModelParallelStrategy):
        model.setup_fabric(fabric)

    # Initialize optimizer
    if initialize_optimizer:
        is_device_cuda = fabric.device.type == "cuda"
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # This is native PyTorch fused optimizer, not use_fused_kernels in config
        use_fused = fused_available and is_device_cuda

        model.configure_optimizers(
            weight_decay=config.optimizer.weight_decay,
            learning_rate=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            use_fused=use_fused,
        )

    # Setup model and optimizer
    # In the FSDP case, model is already setup and this sets up only the optimizer
    model.setup_fabric(fabric)

    # Load checkpoint file
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
        fabric.print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        fabric.print("Initialized a new model from scratch")

    return model


def shard_model(
    module: torch.nn.Module, device_mesh: torch.distributed.device_mesh.DeviceMesh
):
    """
    Function to define the sharding strategy for the model.

    Given a 2D device mesh of (nodes, gpus per node), fully_shard will:
        - Replicate across nodes (data parallel)
        - Shard across GPUs within a node (FSDP)
    """

    # Function to shard individual transformer layers recursively
    # This lets us only gather full weights one layer at a time
    def _shard_recursive(module: torch.nn.Module):
        for submodule in module.children():
            if isinstance(submodule, Block):
                submodule = fully_shard(
                    submodule,
                    mesh=device_mesh,
                    reshard_after_forward=True,
                )
            else:
                _shard_recursive(submodule)

    # Shard the submodules
    _shard_recursive(module)

    # Shard the top level module
    fully_shard(module, mesh=device_mesh, reshard_after_forward=False)

    return module


class Trainer:
    """Trainer class for training/validating GPT models"""

    def __init__(
        self,
        fabric: L.Fabric,
        config,
        model: ModelBase,
        show_progress_bar: bool = True,
    ):
        # Check that model has optimizer
        assert (
            hasattr(model, "optimizer") and model.optimizer is not None
        ), "Model must be initialized with optimizer"

        # Initialization state
        self.fabric = fabric
        self.config = config
        self.model = model
        self.show_progress_bar = show_progress_bar

        # Training loop state
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.tokenizer = None
        self.prepare_batch_func: Optional[Callable] = None
        self.epoch: int = 0
        self.step: int = self.model.training_steps
        self.last_train_loss: Optional[float] = None
        self.best_val_loss: Optional[float] = None
        # Store most recent validation logs for final artifact upload
        self.last_validation_logs: Dict[str, Any] = {}

        # Checkpointing state
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.recovery_checkpoint_path = None
        self.checkpoints_to_always_keep = set()
        # Track whether training completed successfully so we can mark run state appropriately
        self.successfully_finished = False

    def train(self, datamodule):
        """
        Call this to start training.
        """
        # Initialize dataloaders
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.tokenizer = datamodule.get_tokenizer()

        # Check if datamodule has a prepare_batch function
        if hasattr(datamodule, "prepare_batch"):
            self.prepare_batch_func = datamodule.prepare_batch
            self.fabric.print(
                f"Using prepare batch function {type(datamodule).__name__}.prepare_batch()"
            )
        else:
            self.prepare_batch_func = None

        # Reset success flag before starting training/validation
        self.successfully_finished = False

        # Do training or validation
        try:
            if self.config.trainer.val_only:
                self.fabric.print("Running validation only (no training)")
                self.last_validation_logs = self._validation_loop()
                self.fabric.print(self.last_validation_logs)
            else:
                self.fabric.print(f"Starting training from step={self.step}")
                self._train_loop()
                self.fabric.print("Training complete, running final validation")
                self.last_validation_logs = self._validation_loop()
            # If we reach this point, everything completed successfully
            self.successfully_finished = True
        finally:
            # Final artifact upload (only once, at end or interrupt)
            if (
                self.config.trainer.log_to_wandb
                and self.fabric.global_rank == 0
            ):
                # Prefer uploading best first, then latest
                if self.best_checkpoint_path and os.path.isfile(self.best_checkpoint_path):
                    self._upload_checkpoint_to_wandb(
                        self.best_checkpoint_path, "best", self.last_validation_logs
                    )
                if self.latest_checkpoint_path and os.path.isfile(self.latest_checkpoint_path):
                    self._upload_checkpoint_to_wandb(
                        self.latest_checkpoint_path, "latest", self.last_validation_logs
                    )

                # Explicitly finish the wandb run with non-zero exit code if not successful.
                # wandb marks runs with a non-zero exit code as failed/aborted.
                if self.successfully_finished:
                    wandb.finish()
                else:
                    self.fabric.print(
                        "Marking wandb run as failed (non-zero exit code) due to error or interruption"
                    )
                    wandb.finish(exit_code=255)

    def _upload_checkpoint_to_wandb(self, upload_ckpt_path, name_prefix, val_logs):
        # Initialize wandb artifact
        artifact = wandb.Artifact(
            name=name_prefix + "_" + self.config.trainer.experiment_name,
            type="model",
            metadata={
                "config": self.config,
                "validation_logs": val_logs,
                "last_train_loss": self.last_train_loss,
                "best_val_loss": self.best_val_loss,
            },
        )
        # Get the latest checkpoint filename
        # latest_ckpt_pointer = os.path.join(
        #     self.config.trainer.out_dir, "latest_ckpt"
        # )
        if os.path.isfile(upload_ckpt_path):
            # Upload the checkpoint
            self.fabric.print(f"Uploading {upload_ckpt_path} to wandb artifact")
            artifact.add_file(
                upload_ckpt_path,
                name=os.path.basename(upload_ckpt_path),
            )
            # Use name_prefix as the sole alias (e.g., :best or :latest)
            wandb.run.log_artifact(artifact, aliases=[name_prefix])
        else:
            # Checkpoint file not found
            self.fabric.print(
                f"Warning: {upload_ckpt_path} file not found, not saving wandb artifact"
            )

    def _train_loop(self):
        """
        Main training loop.
        Iterate over the train dataloader for the configured number of batches.

        For each batch:
            - If we have reached the validation interval, run validation
            - Split batch into micro batches and accumulate gradients
            - Run optimizer step
            - Do logging
        """
        # Set model to train mode
        self.model.train()

        # Sum of train loss over log_interval batches
        running_train_loss = torch.zeros(1, device=self.fabric.device)

        # Initialize progress bar
        if self.show_progress_bar:
            pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

        # If resuming from a previous checkpoint, fast-forward data
        if self.step > 0:
            do_fast_forward = True
            fast_forward_steps = 0
        else:
            do_fast_forward = False

        # print(f"[DEBUG] BEFORE WHILE LOOP: do_fast_forward = {do_fast_forward} and self.step = {self.step}")

        if do_fast_forward:
            num_batches_in_train_loader = 0
            for batch in self.train_dataloader:
                num_batches_in_train_loader += 1
            fast_forward_steps = num_batches_in_train_loader * int(self.step / num_batches_in_train_loader)

        while True:
            # Do one epoch over the entire training dataloader
            for batch in self.train_dataloader:
                # Fast forward data
                if do_fast_forward:
                    # print("[DEBUG] inside fast forward loop")
                    # Increment fast forward counter until we reach the current step
                    fast_forward_steps += 1
                    if fast_forward_steps >= self.step:
                        self.fabric.print(
                            f"Fast forwarded data to step {fast_forward_steps}"
                        )
                        do_fast_forward = False
                    # Skip this batch
                    continue

                # Garbage collect to free up memory
                if (
                    self.config.trainer.garbage_collect > 0
                    and self.step % self.config.trainer.garbage_collect == 0
                ):
                    # print("[DEBUG] Inside garbage collection")
                    gc.collect()

                # Validation
                validation_logs = {}
                if (
                    self.step % self.config.trainer.val_interval == 0 and self.step > 0
                ) or self.config.trainer.val_only:
                    # Close the previous training progress bar
                    if self.show_progress_bar:
                        pbar.close()

                    validation_logs = self._validation_loop()
                    # Store latest validation logs for final upload (upload happens after training completes or on interrupt)
                    self.last_validation_logs = validation_logs
                    self.model.train()

                    if self.config.trainer.garbage_collect > 0:
                        gc.collect()

                    # Initialize a new progress bar
                    if self.show_progress_bar:
                        pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

                # Prepare batch
                if self.prepare_batch_func is not None:
                    batch = self.prepare_batch_func(batch)

                # Expect batch to always be a tuple: (tokens, prompt_len_or_None)
                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    raise TypeError(
                        "Training dataloader must return a 2-tuple (tokens, prompt_len_or_None). "
                        f"Got type={type(batch)} with length={getattr(batch, '__len__', lambda: 'n/a')()}"
                    )
                tokens, prompt_len = batch

                # print("[DEBUG] BEFORE ACCUM_STEP FOR LOOP")
                # Accumulate gradients over gradient_accum_steps
                for accum_step in range(self.config.data.gradient_accum_steps):
                    start_idx = accum_step * self.config.data.micro_batch_size
                    end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                    sub_tokens = tokens[start_idx:end_idx]
                    sub_prompt_len = None if prompt_len is None else prompt_len[start_idx:end_idx]

                    # Only sync gradients for the last step
                    no_sync = accum_step < self.config.data.gradient_accum_steps - 1
                    loss = self.model.compute_loss(
                        sub_tokens,
                        backpropagate=True,
                        loss_div=self.config.data.gradient_accum_steps,
                        no_sync=no_sync,
                        prompt_len=sub_prompt_len,
                    )

                    # loss is already divided by gradient_accum_steps
                    running_train_loss += loss

                # Optimizer step
                # This should increment model.training_steps
                lr = self._get_lr()
                grad_clip = self.config.optimizer.grad_clip
                grad_norm = self.model.optimizer_step(
                    learning_rate=lr, grad_clip=grad_clip
                )

                # Logging
                if self.step % self.config.trainer.log_interval == 0 and self.step > 0:
                    # Train loss has been accumulated over log_interval batches
                    self.last_train_loss = (
                        running_train_loss.item() / self.config.trainer.log_interval
                    )
                    running_train_loss.zero_()

                    custom_metrics = self.model.get_custom_metrics()
                    log_dict = {
                        "epoch": self.epoch,
                        "lr": lr,
                        "train_loss": self.last_train_loss,
                        **validation_logs,
                        **custom_metrics,
                    }
                    if grad_norm is not None:
                        log_dict["grad_norm"] = grad_norm

                    self.fabric.log_dict(log_dict, self.step)

                # Update progress bar
                progress_str = f"{datetime.now()} Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_train_loss or 0.0:.4f}"
                if self.show_progress_bar:
                    pbar.set_description(progress_str)
                    pbar.update(1)
                else:
                    self.fabric.print(progress_str)

                # Increment batch count
                self.step += 1
                assert (
                    self.step == self.model.training_steps
                ), "Bug in training loop: trainer step count does not match model step count"
                if self.step > self.config.trainer.train_batches:
                    # End of training
                    return

                # Save recovery checkpoint
                if (
                    self.config.trainer.save_recovery_checkpoint > 0
                    and self.step % self.config.trainer.save_recovery_checkpoint == 0
                ):
                    self._save_recovery_checkpoint()

            # End of one full iteration over train dataloader
            self.epoch += 1

    @torch.inference_mode()
    def _validation_loop(self) -> Dict[str, Any]:
        """
        Main validation loop.
        This is called every val_interval steps during training.

        For validation, we do the following:
            - Compute validation loss
            - Save checkpoint
        """
        # Set model to eval mode
        self.model.eval()

        if self.config.trainer.val_batches > 0:
            # Divide total validation batches by world size
            n_val_batches = self.config.trainer.val_batches // self.fabric.world_size
        else:
            # If validation batches are not specified, validate over entire dataloader
            n_val_batches = None

        validation_logs = {}
        batch_count = 0
        total_loss = torch.zeros(1, device=self.fabric.device)

        disable_pbar = (
            True if not self.show_progress_bar else None
        )  # None means automatic

        # Main validation loop
        for batch in tqdm(
            self.val_dataloader,
            desc="Validation",
            total=n_val_batches or len(self.val_dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            if self.prepare_batch_func is not None:
                batch = self.prepare_batch_func(batch)

            # Expect validation batch to always be a tuple: (tokens, prompt_len_or_None)
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise TypeError(
                    "Validation dataloader must return a 2-tuple (tokens, prompt_len_or_None). "
                    f"Got type={type(batch)} with length={getattr(batch, '__len__', lambda: 'n/a')()}"
                )
            tokens, prompt_len = batch

            # Accumulate loss over gradient_accum_steps
            for accum_step in range(self.config.data.gradient_accum_steps):
                start_idx = accum_step * self.config.data.micro_batch_size
                end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                sub_tokens = tokens[start_idx:end_idx]
                sub_prompt_len = None if prompt_len is None else prompt_len[start_idx:end_idx]

                loss = self.model.compute_loss(
                    sub_tokens,
                    backpropagate=False,
                    loss_div=self.config.data.gradient_accum_steps,
                    prompt_len=sub_prompt_len,
                )
                # loss is already divided by gradient_accum_steps
                total_loss += loss

            # End loop if limit_batches is reached
            batch_count += 1
            if n_val_batches is not None and batch_count >= n_val_batches:
                break

        # Sync average loss across all GPUs by uniform mean of per-device averages
        # This assumes that each GPU has roughly the same number of batches
        device_avg_loss = total_loss / batch_count
        global_avg_loss = self.fabric.all_reduce(device_avg_loss, reduce_op="mean")
        val_loss = global_avg_loss.item()

        # Add validation loss to log
        validation_logs["val_loss"] = val_loss

        self.fabric.print(
            f"Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_train_loss} Validation loss: {val_loss}"
        )

        # Update best validation loss
        is_new_best = (
            True if self.best_val_loss is None else (val_loss < self.best_val_loss)
        )
        if is_new_best:
            self.best_val_loss = val_loss

        # Save checkpoint
        if not self.config.trainer.val_only and (
            self.config.trainer.save_last_checkpoint
            or (self.config.trainer.save_best_checkpoint and is_new_best)
            or (
                self.config.trainer.keep_checkpoint_steps
                and self.step in self.config.trainer.keep_checkpoint_steps
            )
        ):
            filename = f"ckpt_iter_{self.step}_loss_{val_loss}.pt"
            new_ckpt_path = self._save_checkpoint(filename)

            # Delete outdated checkpoints
            if not self.config.trainer.always_save_checkpoint:
                old_checkpoints = set()
                if (
                    self.config.trainer.save_last_checkpoint
                    and self.latest_checkpoint_path is not None
                    and (
                        self.latest_checkpoint_path != self.best_checkpoint_path
                        or is_new_best
                    )
                ):
                    old_checkpoints.add(self.latest_checkpoint_path)
                if (
                    self.config.trainer.save_best_checkpoint
                    and is_new_best
                    and self.best_checkpoint_path is not None
                ):
                    old_checkpoints.add(self.best_checkpoint_path)
                if self.fabric.global_rank == 0:
                    for ckpt in old_checkpoints:
                        if ckpt not in self.checkpoints_to_always_keep:
                            os.remove(ckpt)

            # Update latest and best checkpoint paths
            if self.config.trainer.save_last_checkpoint:
                self.latest_checkpoint_path = new_ckpt_path
            if self.config.trainer.save_best_checkpoint and is_new_best:
                self.best_checkpoint_path = new_ckpt_path
            if (
                self.config.trainer.keep_checkpoint_steps
                and self.step in self.config.trainer.keep_checkpoint_steps
            ):
                self.checkpoints_to_always_keep.add(new_ckpt_path)

        return validation_logs

    def _save_checkpoint(self, filename: str) -> str:
        """
        Save the model checkpoint to the given file name.
        Then save the checkpoint file path to the latest_ckpt file (sibling to
        the checkpoint, inside the run dir).
        """
        ckpt_dir = self.config.trainer.run_dir
        ckpt_path = os.path.join(ckpt_dir, filename)
        self.model.save_checkpoint(ckpt_path)

        # Save the file path to the latest checkpoint, inside the run dir so
        # concurrent runs in the same group don't clobber each other's pointers.
        if self.fabric.global_rank == 0:
            with open(os.path.join(ckpt_dir, "latest_ckpt"), "w") as f:
                f.write(ckpt_path)

        return ckpt_path

    def _save_recovery_checkpoint(self):
        """
        Save a checkpoint for recovery from crashes.
        """
        ckpt_dir = self.config.trainer.run_dir
        ckpt_path = os.path.join(ckpt_dir, f"recovery_ckpt_iter_{self.step}.pt")
        self.model.save_checkpoint(ckpt_path)

        # Save the most recent file path to the recovery checkpoint pointer
        # file, sibling to the checkpoint inside the run dir.
        if self.fabric.global_rank == 0:
            with open(os.path.join(ckpt_dir, "recovery_ckpt"), "w") as f:
                f.write(ckpt_path)

        # Delete the old recovery checkpoint file if it exists
        if self.recovery_checkpoint_path is not None:
            if self.fabric.global_rank == 0:
                os.remove(self.recovery_checkpoint_path)

        # Update the recovery checkpoint path
        self.recovery_checkpoint_path = ckpt_path

    def _get_lr(self) -> float:
        """
        Cosine learning rate decay with warmup
        """
        iter_num = self.step
        warmup_iters = self.config.lr_scheduler.warmup_iters
        decay_iters = self.config.lr_scheduler.lr_decay_iters
        max_lr = self.config.optimizer.learning_rate
        min_lr = self.config.lr_scheduler.min_lr

        # If decay_lr is False, always use max_lr
        if not self.config.lr_scheduler.decay_lr:
            return max_lr

        # 1) linear warmup for warmup_iters steps
        if iter_num < warmup_iters:
            return max_lr * iter_num / warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if iter_num > decay_iters:
            return min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter_num - warmup_iters) / (decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)
