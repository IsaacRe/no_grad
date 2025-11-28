import os.path
import sys
from torchvision.models import resnet18
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import contextlib
try:
    import wandb
    from wandb import Run as WandbRun
except ImportError:
    wandb = None

from no_grad.optim import get_optimizer, OptimizerConfig, ESOptimizer
from no_grad.utils import Logger


@dataclass
class TrainConfig:
    batch_size: int = 128
    dataloader_workers: int = 2
    model_checkpoint_path: str = "tmp.pt"
    model_eval_mode: bool = False
    cifar_path: str = "./data"
    es_updates_per_batch: int = 1
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb_project_name: str = "no_grad-cifar"
    report_to_wandb: bool = False


def load_model(checkpoint_path: str, eval_mode: bool = False):
    model = resnet18().to(0)
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint at {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print(f"Checkpoint not found at {checkpoint_path}, initializing new model.")
        torch.save(model.state_dict(), checkpoint_path)
    if eval_mode:
        model = model.eval()
    else:
        model = model.train()
    return model


def load_cifar(cifar_path: str = "./data", batch_size: int = 128, n_workers: int = 2):
    # Standard normalization for CIFAR-10
    NORM_MEAN = [0.4914, 0.4822, 0.4465]
    NORM_STD = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    train_dataset = datasets.CIFAR10(
        root=cifar_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=n_workers,
    )

    val_dataset = datasets.CIFAR10(
        root=cifar_path, 
        train=False, 
        download=True, 
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=n_workers,
    )

    return train_loader, val_loader


def train_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    val_loader: DataLoader | None = None,
    iters_per_batch: int = 1,
    use_tqdm: bool = False,
    wandb_run: "WandbRun | None" = None,
):
    val_iter = iter(val_loader)

    criterion = nn.CrossEntropyLoss()

    sample_param = next(model.parameters())
    dtype = sample_param.dtype
    device = sample_param.device

    step_count = 0
    log_metrics = [("batch", f"{{}}/{len(train_loader)}"), ("step", "{}"), ("train loss", "{:.6f}")]
    if val_loader is not None:
        log_metrics.append(("val loss", "{:.6f}"))
    logger = Logger(*log_metrics, column_width=12)
    if use_tqdm:
        pbar = tqdm(total=len(train_loader) * iters_per_batch)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device=device, dtype=dtype), targets.to(device)
        logger.print_header()

        for j in range(iters_per_batch):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            log_vals = [i+1, step_count, loss.item()]

            loss.backward()

            optimizer.step()

            if val_loader is not None:
                try:
                    val_inputs, val_targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_inputs, val_targets = next(val_iter)
                val_inputs, val_targets = val_inputs.to(device=device, dtype=dtype), val_targets.to(device=device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                log_vals.append(val_loss.item())

            step_count += 1
            logger.print_metrics(*log_vals)

            if wandb_run is not None:
                # map 'train loss' to 'avg loss' and to 'min loss' so wandb can overlay plots
                vals = {k: v for k, v in zip([m[0] for m in log_metrics], log_vals)}
                vals.update({"avg loss": vals["train loss"], "min loss": vals["train loss"]})
                # map 'step_count' to 'forwards' (one forward pass per training step)
                vals["forwards"] = step_count
                wandb_run.log(vals)

            if use_tqdm:
                pbar.update(1)

    if use_tqdm:
        pbar.close()


@torch.no_grad()
def train_es(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: ESOptimizer,
    val_loader: DataLoader | None = None,
    iters_per_batch: int = 1,
    use_tqdm: bool = False,
    wandb_run: "WandbRun | None" = None,
):
    val_iter = iter(val_loader)

    criterion = nn.CrossEntropyLoss()
    
    sample_param = next(model.parameters())
    dtype = sample_param.dtype
    device = sample_param.device

    avg_loss = 0
    min_loss = 1000
    step_count = 0
    log_metrics = [("batch", f"{{}}/{len(train_loader)}"), ("step", "{}"), ("avg loss", "{:.6f}"), ("min loss", "{:.6f}"), ("lr", "{:.6f}")]
    if val_loader is not None:
        log_metrics.append(("val loss", "{:.6f}"))
    logger = Logger(*log_metrics, column_width=12)
    if use_tqdm:
        pbar = tqdm(total=len(train_loader) * iters_per_batch)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device=device, dtype=dtype), targets.to(device=device)
        logger.print_header()

        for _ in range(iters_per_batch):
            optimizer.mutate()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets).item()
            avg_loss += loss
            min_loss = loss if loss < min_loss else min_loss

            # if final mutation, mutation_index will reset
            optimizer.reward_step(-loss)
            
            # report aggregate training loss across all mutations
            # (this will be higher than loss for aggregated model)
            if optimizer.mutation_index == -1:
                avg_loss /= optimizer.population_size
                log_vals = [i+1, step_count, avg_loss, min_loss, optimizer.lr]

                if val_loader is not None:
                    try:
                        val_inputs, val_targets = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_inputs, val_targets = next(val_iter)
                    val_inputs, val_targets = val_inputs.to(device=device, dtype=dtype), val_targets.to(device=device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
                    log_vals.append(val_loss.item())

                avg_loss = 0
                min_loss = 1000
                step_count += 1
                logger.print_metrics(*log_vals)

                if wandb_run is not None:
                    vals = {k: v for k, v in zip([m[0] for m in log_metrics], log_vals)}
                    # track forward pass count
                    vals["forwards"] = step_count * optimizer.population_size
                    wandb_run.log(vals)

            if use_tqdm:
                pbar.update(1)

    if use_tqdm:
        pbar.close()


def main():
    try:
        cfg = OmegaConf.load(sys.argv[1])
    except FileNotFoundError:
        print("Usage: python scripts/cifar.py <config_file> [additional args...]")
        sys.exit(1)

    cfg: TrainConfig = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg)

    if len(sys.argv) > 2:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(sys.argv[2:]))

    model = load_model(cfg.model_checkpoint_path, eval_mode=cfg.model_eval_mode)
    train_loader, val_loader = load_cifar(cfg.cifar_path, cfg.batch_size, cfg.dataloader_workers)
    optimizer = get_optimizer(
        model.parameters(),
        cfg.optimizer_config,
    )
    if wandb is not None and cfg.report_to_wandb:
        run_ctx = wandb.init(
            project=cfg.wandb_project_name,
            config=OmegaConf.to_container(cfg.optimizer_config, resolve=True),
            id=OptimizerConfig.make_run_id(cfg.optimizer_config),
        )
    else:
        run_ctx = contextlib.nullcontext()
    with run_ctx as maybe_wandb_run:
        if isinstance(optimizer, ESOptimizer):
            train_es(
                model,
                train_loader,
                optimizer,
                val_loader=val_loader,
                iters_per_batch=cfg.es_updates_per_batch * cfg.optimizer_config.es.population_size,
                wandb_run=maybe_wandb_run,
            )
        else:
            train_sgd(
                model,
                train_loader,
                optimizer,
                val_loader=val_loader,
                iters_per_batch=1,
                wandb_run=maybe_wandb_run,
            )


if __name__ == "__main__":
    main()
