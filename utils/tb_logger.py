# your_project_name/utils/tb_logger.py
from __future__ import annotations
import logging
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

# Profiler-specific imports if enabled
try:
    from torch.profiler import profile, record_function, ProfilerActivity, schedule
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    profile = None # Make it a no-op if not available

logger = logging.getLogger(__name__)

def denormalize_image(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """Denormalizes a tensor image."""
    if not isinstance(tensor, torch.Tensor):
        return tensor # Or raise error
    
    # Clone to avoid modifying original tensor
    tensor = tensor.clone()

    # Ensure mean and std are tensors and reshaped correctly for broadcasting
    # Assuming tensor is NCHW or CHW
    if tensor.ndim == 3: # CHW
        _mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
        _std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    elif tensor.ndim == 4: # NCHW
        _mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
        _std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    else:
        logger.warning(f"Denormalize: Unexpected tensor ndim {tensor.ndim}. Returning original.")
        return tensor
        
    tensor.mul_(_std).add_(_mean)
    return torch.clamp(tensor, 0, 1) # Clamp to valid image range

class TensorBoardLogger:
    def __init__(self, log_dir: str | Path, experiment_config: dict, train_loader_len: int):
        self.cfg_main = experiment_config
        self.cfg_tb = self.cfg_main.get("tensorboard_logging", {})
        self.writer = None
        self.global_train_step = 0
        self.global_val_step = 0
        self.train_loader_len = train_loader_len

        if not self.cfg_tb.get("enable", False):
            logger.info("TensorBoard logging is disabled in the configuration.")
            return

        self.writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard logger initialized. Logging to: {log_dir}")

        # Image logging config
        self.img_log_cfg = self.cfg_tb.get("image_logging", {})
        self.img_log_enabled = self.img_log_cfg.get("enable", False)
        self.img_denormalize = self.img_log_cfg.get("denormalize", True)
        self.img_num_samples = self.img_log_cfg.get("num_samples", 4)
        self.img_log_at_epochs = self.img_log_cfg.get("log_at_epochs", [])
        if isinstance(self.img_log_at_epochs, int): # Handle single int case
            self.img_log_at_epochs = [self.img_log_at_epochs]


        # Profiler config
        self.profiler_cfg = self.cfg_tb.get("profiler", {})
        self.profiler_enabled_for_epoch = False # Will be set per epoch
        self.profiler_instance = None

        # Memory logging config
        self.mem_log_cfg = self.cfg_tb.get("memory_logging", {})
        self.mem_log_enabled = self.mem_log_cfg.get("enable", False)
        
        # Scalar logging intervals
        self.log_interval_train_batch = self.cfg_tb.get("log_interval_batches_train", 0)
        self.log_interval_val_batch = self.cfg_tb.get("log_interval_batches_val", 0)


    def _should_log_batch(self, interval: int, batch_idx: int) -> bool:
        return interval > 0 and (batch_idx + 1) % interval == 0

    def log_train_batch_metrics(self, loss: float, acc: float, lr: float, epoch: int, batch_idx: int, imgs: torch.Tensor | None = None):
        if not self.writer: return
        self.global_train_step = epoch * self.train_loader_len + batch_idx

        if self._should_log_batch(self.log_interval_train_batch, batch_idx):
            self.writer.add_scalar("Loss/train_batch", loss, self.global_train_step)
            self.writer.add_scalar("Accuracy/train_batch", acc, self.global_train_step)
            # LR is usually per epoch, but if it changes per batch (custom scheduler), log here.
            # self.writer.add_scalar("LearningRate/batch", lr, self.global_train_step)

        if self.mem_log_enabled and torch.cuda.is_available() and \
           self._should_log_batch(self.mem_log_cfg.get("log_interval_batches", 0), batch_idx):
            self.writer.add_scalar("Memory/Allocated_MB_batch", torch.cuda.memory_allocated() / 1024**2, self.global_train_step)
            self.writer.add_scalar("Memory/Reserved_MB_batch", torch.cuda.memory_reserved() / 1024**2, self.global_train_step)
        
        # Image logging (training) - typically done at epoch end or specific epochs for less clutter
        # But if needed per batch for debugging, can be adapted. Here, tied to epoch check.
        if self.img_log_enabled and self.img_log_cfg.get("log_train_input", False) and \
           epoch in self.img_log_at_epochs and batch_idx == 0: # Log first batch of specified epochs
            self._log_images(imgs, "Train/Input_Samples", epoch)


    def log_val_batch_metrics(self, loss: float, acc: float, epoch: int, batch_idx: int, imgs: torch.Tensor | None = None):
        if not self.writer: return
        # global_val_step could be epoch * val_loader_len + batch_idx if needed for finer Grained val batch logging
        
        if self._should_log_batch(self.log_interval_val_batch, batch_idx):
            # Using epoch as step for batch val metrics to overlay them per epoch rather than continuous global step
            step = epoch * 1000 + batch_idx # Hacky step to group by epoch, or use a different tag structure
            self.writer.add_scalar(f"Loss/val_batch_E{epoch}", loss, batch_idx) # Log per batch within an epoch
            self.writer.add_scalar(f"Accuracy/val_batch_E{epoch}", acc, batch_idx)

        # Image logging (validation) - typically done at epoch end or specific epochs
        if self.img_log_enabled and self.img_log_cfg.get("log_val_input", False) and \
           epoch in self.img_log_at_epochs and batch_idx == 0: # Log first batch of specified epochs
            self._log_images(imgs, "Val/Input_Samples", epoch)
            # TODO: Add self.img_log_cfg.get("log_val_predictions", False) logic if applicable


    def _log_images(self, images_tensor: torch.Tensor | None, tag: str, step: int):
        if not self.writer or images_tensor is None or not self.img_log_enabled:
            return
        
        try:
            num_to_log = min(self.img_num_samples, images_tensor.size(0))
            imgs_to_log = images_tensor[:num_to_log].cpu()

            if self.img_denormalize:
                norm_mean = self.cfg_main.get("NORM_MEAN", [0.485, 0.456, 0.406]) # Get from main config
                norm_std = self.cfg_main.get("NORM_STD", [0.229, 0.224, 0.225])
                imgs_to_log = denormalize_image(imgs_to_log, norm_mean, norm_std)
            
            grid = make_grid(imgs_to_log, nrow=int(np.sqrt(num_to_log)))
            self.writer.add_image(tag, grid, step)
            logger.debug(f"Logged {num_to_log} images to TensorBoard with tag '{tag}' at step {step}.")
        except Exception as e:
            logger.error(f"Failed to log images for tag '{tag}': {e}", exc_info=True)


    def log_epoch_summary(self, metrics: dict, epoch: int):
        if not self.writer or not self.cfg_tb.get("log_epoch_summary", True):
            return

        for key, value in metrics.items():
            if value is not None: # Ensure value is not None
                self.writer.add_scalar(key, value, epoch)
        
        if self.cfg_tb.get("log_lr", True) and "lr" in metrics: # LR is already in metrics dict
            pass # self.writer.add_scalar("LearningRate/epoch", metrics["lr"], epoch)

        if self.cfg_tb.get("log_throughput", True) and "throughput/train_samples_per_sec" in metrics:
            pass # self.writer.add_scalar("Throughput/train_samples_per_sec", metrics["throughput/train_samples_per_sec"], epoch)
        
        if self.cfg_tb.get("log_gpu_time_epoch", True) and torch.cuda.is_available() and "time/gpu_ms_per_train_epoch" in metrics:
            pass # self.writer.add_scalar("Time/GPU_ms_per_train_epoch", metrics["time/gpu_ms_per_train_epoch"], epoch)

        if self.mem_log_enabled and self.mem_log_cfg.get("log_epoch_summary", True) and torch.cuda.is_available():
            self.writer.add_scalar("Memory/Epoch_Peak_Allocated_MB", torch.cuda.max_memory_allocated() / 1024**2, epoch)
            self.writer.add_scalar("Memory/Epoch_Peak_Reserved_MB", torch.cuda.max_memory_reserved() / 1024**2, epoch)
            # Reset peak stats for the next epoch (already done in train_one_fold, but good to be aware)
            # torch.cuda.reset_peak_memory_stats() # This is typically done at start of epoch in training loop


    def setup_profiler(self, current_epoch: int, profiler_log_dir: str | Path) -> torch.profiler.profile | None:
        if not self.writer or not PROFILER_AVAILABLE:
            self.profiler_instance = None
            return None

        self.profiler_enabled_for_epoch = (
            self.profiler_cfg.get("enable", False) and
            current_epoch == self.profiler_cfg.get("profile_epoch", -1)
        )

        if self.profiler_enabled_for_epoch:
            trace_dir = Path(profiler_log_dir) / "profiler_traces" # Use the fold-specific log_dir
            trace_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"PyTorch Profiler ENABLED for epoch {current_epoch}. Traces will be saved to {trace_dir}")
            
            prof_schedule = schedule(
                wait=self.profiler_cfg.get("wait_steps", 1),
                warmup=self.profiler_cfg.get("warmup_steps", 1),
                active=self.profiler_cfg.get("active_steps", 3),
                repeat=self.profiler_cfg.get("repeat_cycles", 0) # 0 means one full cycle
            )
            
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            self.profiler_instance = profile(
                activities=activities,
                schedule=prof_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=self.profiler_cfg.get("record_shapes", True),
                profile_memory=self.profiler_cfg.get("profile_memory", True),
                with_stack=self.profiler_cfg.get("with_stack", True),
                # with_flops=self.profiler_cfg.get("with_flops", False) # Requires fvcore
            )
            self.profiler_instance.start() # Start the profiler
            return self.profiler_instance
        
        self.profiler_instance = None
        return None

    def step_profiler(self):
        if self.profiler_instance and self.profiler_enabled_for_epoch:
            self.profiler_instance.step()

    def stop_and_process_profiler(self):
        if self.profiler_instance and self.profiler_enabled_for_epoch:
            self.profiler_instance.stop() # Stop the profiler
            logger.info("Processing profiler results...")
            sort_by = self.profiler_cfg.get("sort_by", "self_cuda_time_total")
            row_limit = self.profiler_cfg.get("row_limit", 20)
            group_by_input_shape = self.profiler_cfg.get("group_by_input_shape", False)
            
            try:
                # Print to console
                key_avg_table = self.profiler_instance.key_averages(group_by_input_shape=group_by_input_shape).table(sort_by=sort_by, row_limit=row_limit)
                print(f"\n--- PyTorch Profiler Summary (Epoch {self.profiler_cfg.get('profile_epoch', -1)}, sorted by {sort_by}) ---")
                print(key_avg_table)

                # Manual Chrome trace export (optional, as tensorboard_trace_handler already does it)
                if self.profiler_cfg.get("export_chrome_trace_manual", False):
                    manual_trace_path = Path(self.writer.log_dir) / "profiler_traces" / f"manual_trace_e{self.profiler_cfg.get('profile_epoch', -1)}.json"
                    self.profiler_instance.export_chrome_trace(str(manual_trace_path))
                    logger.info(f"Manually exported Chrome trace to {manual_trace_path}")

            except Exception as e:
                logger.error(f"Failed during profiler result processing: {e}", exc_info=True)
            
            logger.info("Profiler traces (if any) should be available in TensorBoard under the PROFILER tab "
                        f"in directory: {Path(self.writer.log_dir) / 'profiler_traces'}")
            self.profiler_instance = None # Reset for next potential use (though typically one epoch)
            self.profiler_enabled_for_epoch = False


    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.flush()
            self.writer.close()
            logger.info("TensorBoard writer closed.")