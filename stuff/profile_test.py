import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import time
from pathlib import Path
import multiprocessing # Import multiprocessing

# --- Define classes and functions at the top level ---
# These are safe to define globally as they are just definitions.

# --- 1. Define a Simple Model ---
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = models.resnet18(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.network(x)

# --- 2. Define a Dummy Dataset ---
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=224, num_classes=10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, 3, img_size, img_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Function to encapsulate your training and profiling logic ---
def run_training_and_profiling():
    # --- 3. Training Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}") # This will now only print once from the main process

    model = SimpleModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2 if device.type == 'cuda' else 0 # Set to > 0 to test multiprocessing
    pin_memory_enabled = True if device.type == 'cuda' and NUM_WORKERS > 0 else False

    dataset = DummyDataset(num_samples=BATCH_SIZE * 20, img_size=IMG_SIZE, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory_enabled) # Add this line)

    # --- 4. Profiler Configuration ---
    PROFILER_ENABLED = True
    PROFILE_EPOCH = 0
    WAIT_STEPS = 1
    WARMUP_STEPS = 1
    ACTIVE_STEPS = 3
    REPEAT_CYCLES = 0

    PROFILE_DIR = Path("./profile_traces")
    PROFILE_DIR.mkdir(exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    num_epochs = 1
    print(f"Starting training for {num_epochs} epoch(s)...") # This will now only print once

    schedule = torch.profiler.schedule(
        wait=WAIT_STEPS,
        warmup=WARMUP_STEPS,
        active=ACTIVE_STEPS,
        repeat=REPEAT_CYCLES
    )

    profiler_kwargs = {
        "schedule": schedule,
        "on_trace_ready": torch.profiler.tensorboard_trace_handler(str(PROFILE_DIR)),
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
        "with_modules": True,
        "activities": activities
    }
    if pin_memory_enabled:
        print(f"DataLoader initialized with pin_memory=True and num_workers={NUM_WORKERS}")

    if PROFILER_ENABLED:
        profiler_context = torch.profiler.profile(**profiler_kwargs)
    else:
        from contextlib import nullcontext
        profiler_context = nullcontext()

    model.train()
    global_step = 0

    # --- 5. Training Loop with Profiler ---
    # THIS IS THE LINE 124 from your original traceback context
    with profiler_context as prof: # profiler_context is either the profiler or nullcontext
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}") # Printed by main process
            # The error occurs when this loop starts and DataLoader tries to spawn workers
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if PROFILER_ENABLED and batch_idx >= (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * (REPEAT_CYCLES + 1):
                    print("Profiling active steps completed for all cycles. Breaking loop.")
                    break

                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Global Step: {global_step}") # Printed by main
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if PROFILER_ENABLED and hasattr(prof, 'step'): # Check if profiler_context is the actual profiler
                    prof.step()

                global_step += 1
            
            if PROFILER_ENABLED and batch_idx >= (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * (REPEAT_CYCLES + 1) :
                break


    print("\nTraining loop finished.")

    # --- 6. Accessing and Printing Profiler Results (if enabled) ---
    if PROFILER_ENABLED and hasattr(prof, 'key_averages'): # Check if actual profiler ran
        print("\n--- Profiler Results (Key Averages, Grouped by Operator) ---")
        sort_by_key = "self_cuda_time_total" if device.type == 'cuda' else "self_cpu_time_total"
        try:
            print(prof.key_averages(group_by_input_shape=False, group_by_stack_n=0).table(sort_by=sort_by_key, row_limit=20))
        except Exception as e:
            print(f"Could not print table sorted by {sort_by_key}, trying 'self_cpu_time_total': {e}")
            print(prof.key_averages(group_by_input_shape=False, group_by_stack_n=0).table(sort_by="self_cpu_time_total", row_limit=20))

        print(f"\nProfiler traces (for TensorBoard) were saved in directory: {PROFILE_DIR}")
    else:
        print("\nProfiler was disabled or no results. No profiling results to show.")


# --- Main execution guard ---
if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows + spawn, and generally good practice
    run_training_and_profiling()      # Call the main function