"""
step3_train.py

Script d'entrainement (supervision faible par CT).

Pipeline:
1) Charger tous les fichiers .nc de DATA_DIR avec step1_data_loader.AI4ArcticWeaklyLabeledDataset
2) ConcatDataset -> DataLoader
3) Entrainement d'un petit U-Net (step2_model.UNetWeak)
4) Logging de ressources (RAM / VRAM) et sauvegarde de quelques images de debug.

Notes pratiques:
- Sur GPU 8 Go, on utilise souvent: AMP + petite batch + accumulation de gradient.
- On ignore un warning CuPy (CUDA_PATH) qui n'est pas bloquant pour ce script.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

# Configuration simple (tu peux modifier ici).
DATA_DIR = "../Northwest"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5
LAMBDA_REG = 0.01
USE_AMP = True
ACCUM_STEPS = 4
PATCH_SIZE = 256
AUGMENT = True
NUM_WORKERS = 0
SHUFFLE = True
DROP_LAST = True
DEBUG_VIZ = True
SAVE_DIR = "."
DEVICE = "auto"  # "auto" | "cpu" | "cuda" | "mps"
SEED: int | None = None
MAX_FILES: int | None = None


# Filtre un warning CuPy bruyant (CuPy est installe mais n'est pas requis ici).
warnings.filterwarnings(
    "ignore",
    message=r"CUDA path could not be detected\..*CuPy fails to load\.",
    category=UserWarning,
)

# Cache quelques warnings connus pour ne garder que l'info utile.
warnings.filterwarnings(
    "ignore",
    message=r"expandable_segments not supported on this platform.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\..* is deprecated.*",
    category=FutureWarning,
)

# Limite la fragmentation CUDA (utile sur petites VRAM).
# PyTorch recent deprecie PYTORCH_CUDA_ALLOC_CONF au profit de PYTORCH_ALLOC_CONF.
_ALLOC_KEY = "PYTORCH_ALLOC_CONF"
_ALLOC_KEY_DEPRECATED = "PYTORCH_CUDA_ALLOC_CONF"

# Si l'utilisateur a defini l'ancienne variable, on la copie puis on la supprime
# pour eviter le warning de deprecation.
if (_ALLOC_KEY not in os.environ) and (_ALLOC_KEY_DEPRECATED in os.environ):
    os.environ[_ALLOC_KEY] = os.environ[_ALLOC_KEY_DEPRECATED]
    del os.environ[_ALLOC_KEY_DEPRECATED]

if _ALLOC_KEY not in os.environ:
    # Sur Windows, "expandable_segments" n'est pas supporte par certaines builds PyTorch,
    # donc on utilise une config plus conservative.
    if os.name == "nt":
        os.environ[_ALLOC_KEY] = "max_split_size_mb:128"
    else:
        os.environ[_ALLOC_KEY] = "expandable_segments:True,max_split_size_mb:128"

# IMPORTANT:
# Le warning "PYTORCH_CUDA_ALLOC_CONF is deprecated" est emis pendant l'import de torch
# si la variable est presente dans l'environnement. On fait donc ce parametrage AVANT
# d'importer torch.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from step1_data_loader import AI4ArcticWeaklyLabeledDataset
from step2_model import UNetWeak


def _format_bytes(n_bytes: int | None) -> str:
    """Formatage lisible (MiB/GiB)."""
    if n_bytes is None:
        return "n/a"
    x = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024.0 or unit == "TiB":
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}TiB"


def _try_psutil():
    """psutil est optionnel (meilleur monitoring CPU/RAM si present)."""
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _get_process_rss_bytes() -> int | None:
    """Memoire RSS du process (meilleur effort)."""
    psutil = _try_psutil()
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass

    # Fallback Windows sans dependance.
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            pass

    return None


def _get_system_ram_bytes() -> tuple[int | None, int | None]:
    """(total_bytes, avail_bytes) pour la RAM systeme."""
    psutil = _try_psutil()
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            pass

    # Fallback Windows sans dependance.
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            if ok:
                return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        except Exception:
            pass

    return None, None


def _get_gpu_mem_str(device: torch.device) -> str:
    """Snapshot memoire GPU (CUDA/MPS)."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            alloc = int(torch.cuda.memory_allocated())
            reserv = int(torch.cuda.memory_reserved())
            max_alloc = int(torch.cuda.max_memory_allocated())
            max_reserv = int(torch.cuda.max_memory_reserved())

            free, total = torch.cuda.mem_get_info()
            free = int(free)
            total = int(total)

            return (
                f"cuda_alloc={_format_bytes(alloc)} "
                f"cuda_res={_format_bytes(reserv)} "
                f"cuda_peak_alloc={_format_bytes(max_alloc)} "
                f"cuda_peak_res={_format_bytes(max_reserv)} "
                f"cuda_free={_format_bytes(free)}/{_format_bytes(total)}"
            )
        except Exception:
            return "cuda_mem=n/a"

    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            current = int(torch.mps.current_allocated_memory()) if hasattr(torch.mps, "current_allocated_memory") else None
            driver = int(torch.mps.driver_allocated_memory()) if hasattr(torch.mps, "driver_allocated_memory") else None
            parts = []
            if current is not None:
                parts.append(f"mps_alloc={_format_bytes(current)}")
            if driver is not None:
                parts.append(f"mps_driver={_format_bytes(driver)}")
            return " ".join(parts) if parts else "mps_mem=n/a"
        except Exception:
            return "mps_mem=n/a"

    return "gpu_mem=n/a"


def _get_cpu_usage_str() -> str | None:
    """Usage CPU (process + systeme) si psutil est disponible."""
    psutil = _try_psutil()
    if psutil is None:
        return None
    try:
        p = psutil.Process(os.getpid())
        proc = p.cpu_percent(interval=None)
        sys = psutil.cpu_percent(interval=None)
        return f"cpu_proc={proc:.1f}% cpu_sys={sys:.1f}%"
    except Exception:
        return None


def _get_nvidia_smi_str() -> str | None:
    """Resume nvidia-smi (si present)."""
    try:
        import subprocess

        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return None
        lines = (res.stdout or "").strip().splitlines()
        if not lines:
            return None
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 6:
            return None
        util_gpu, util_mem, mem_used, mem_total, temp, pwr = parts[:6]
        return (
            f"nvsmi_util={util_gpu}% "
            f"mem_util={util_mem}% "
            f"vram={mem_used}/{mem_total}MiB "
            f"temp={temp}C "
            f"pwr={pwr}W"
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None


def print_resources(tag: str, device: torch.device) -> None:
    """Affiche un snapshot de ressources: RAM (process/system) + VRAM."""
    rss = _get_process_rss_bytes()
    total_ram, avail_ram = _get_system_ram_bytes()

    msg = f"proc_rss={_format_bytes(rss)}"
    if total_ram is not None and avail_ram is not None:
        msg += f" ram_avail={_format_bytes(avail_ram)}/{_format_bytes(total_ram)}"

    cpu_str = _get_cpu_usage_str()
    if cpu_str:
        msg += f" {cpu_str}"

    msg += f" | {_get_gpu_mem_str(device)}"
    if device.type == "cuda":
        smi = _get_nvidia_smi_str()
        if smi:
            msg += f" | {smi}"

    print(f"[resources] {tag} | {msg}")


def debug_viz(
    inputs: torch.Tensor,
    pred_map: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    batch_idx: int,
    out_dir: str,
) -> None:
    """Sauvegarde une figure de debug pour verifier que le modele produit quelque chose."""
    sar_img = inputs[0, 0, :, :].detach().cpu().numpy()
    amsr_img = inputs[0, 3, :, :].detach().cpu().numpy()
    pred_mask = pred_map[0, 0, :, :].detach().cpu().numpy()
    true_conc = float(targets[0].item())
    pred_conc = float(np.mean(pred_mask))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sar_img, cmap="gray")
    plt.title("SAR HH")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(amsr_img, cmap="magma")
    plt.title("AMSR2")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="jet", vmin=0, vmax=1)
    plt.title(f"Pred: {pred_conc:.2f} / Cible: {true_conc:.2f}")
    plt.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch+1}_batch_{batch_idx}.png"))
    plt.close()


def train() -> None:
    # Seed (optionnel) pour rendre les runs plus reproductibles.
    if SEED is not None:
        np.random.seed(int(SEED))
        torch.manual_seed(int(SEED))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(SEED))

    # Selection device: CUDA -> MPS -> CPU.
    device_choice = str(DEVICE).lower().strip()
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda demande, mais CUDA n'est pas disponible.")
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cuda"
        print(f"Device: CUDA ({name})")
    elif device_choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("DEVICE=mps demande, mais MPS n'est pas disponible.")
        device = torch.device("mps")
        print("Device: Apple Metal (MPS)")
    elif device_choice == "cpu":
        device = torch.device("cpu")
        print("Device: CPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cuda"
        print(f"Device: CUDA ({name})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    print_resources("startup", device)

    nc_files = glob.glob(os.path.join(DATA_DIR, "*.nc"))
    nc_files.sort()
    if not nc_files:
        print(f"ERREUR: pas de fichiers .nc dans {DATA_DIR}")
        return

    if MAX_FILES is not None:
        nc_files = nc_files[: int(MAX_FILES)]

    print(f"Chargement de {len(nc_files)} fichiers...")
    datasets = [AI4ArcticWeaklyLabeledDataset(f, patch_size=int(PATCH_SIZE), augment=bool(AUGMENT)) for f in nc_files]
    mega_dataset = ConcatDataset(datasets)

    dataloader = DataLoader(
        mega_dataset,
        batch_size=BATCH_SIZE,
        shuffle=bool(SHUFFLE),
        drop_last=bool(DROP_LAST),
        pin_memory=(device.type == "cuda"),
        num_workers=int(NUM_WORKERS),
    )

    out_root = os.path.abspath(SAVE_DIR)
    debug_dir = os.path.join(out_root, "debug_plots")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"Dataset unifie : {len(mega_dataset)} patches")
    print(
        "Config: "
        f"data_dir={DATA_DIR}, save_dir={out_root}, patch_size={PATCH_SIZE}, "
        f"batch_size={BATCH_SIZE}, accum_steps={ACCUM_STEPS}, lr={LEARNING_RATE}, "
        f"epochs={EPOCHS}, amp={USE_AMP}, augment={AUGMENT}, workers={NUM_WORKERS}"
    )

    model = UNetWeak(n_channels=4, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_criterion = nn.MSELoss()

    use_amp = bool(USE_AMP and (device.type == "cuda"))
    # Nouvelle API AMP (PyTorch >= 2.0) : torch.amp.* (evite les FutureWarning)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    loss_history: list[float] = []

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        running_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        optimizer.zero_grad(set_to_none=True)
        for i, (inputs, targets) in enumerate(loop):
            try:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred_concentration, pred_map = model(inputs)
                    loss_mse = mse_criterion(pred_concentration, targets)
                    # Regularisation "anti-binaire": penalise une carte trop 0/1 trop vite.
                    loss_bin = torch.mean(pred_map * (1.0 - pred_map))
                    loss = loss_mse + (LAMBDA_REG * loss_bin)

                loss_for_backward = loss / max(1, ACCUM_STEPS)
                scaler.scale(loss_for_backward).backward()

                if ((i + 1) % max(1, ACCUM_STEPS) == 0) or ((i + 1) == len(dataloader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            except (torch.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.OutOfMemoryError) or ("out of memory" in str(exc).lower())
                if device.type == "cuda" and is_oom:
                    print(f"WARNING: CUDA OOM sur batch {i}, batch ignore")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            running_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

            if DEBUG_VIZ and i == 0:
                debug_viz(inputs, pred_map, targets, epoch, i, out_dir=debug_dir)

        avg_loss = running_loss / max(1, len(dataloader))
        loss_history.append(avg_loss)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

        dt = time.perf_counter() - t0
        print_resources(f"epoch {epoch+1}/{EPOCHS} end (loss={avg_loss:.6f}, dt={dt:.1f}s)", device)
        torch.save(model.state_dict(), os.path.join(out_root, f"weights_epoch_{epoch+1}.pth"))

    plt.figure()
    plt.plot(loss_history, marker="o")
    plt.title("Loss (MSE + Reg)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_root, "loss_curve.png"))

    print("Entrainement termine")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrainement weak supervision (CT) sur fichiers .nc (AI4Arctic / Northwest).",
    )
    p.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Dossier contenant les fichiers .nc (ex: ../Northwest_2019 ou ../Northwest).",
    )
    p.add_argument("--save-dir", default=SAVE_DIR, help="Dossier de sortie (weights, debug_plots, loss_curve.png).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (micro-batch).")
    p.add_argument("--accum-steps", type=int, default=ACCUM_STEPS, help="Accumulation de gradient (batch effectif = batch_size * accum_steps).")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'epoques.")
    p.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    p.add_argument("--lambda-reg", type=float, default=LAMBDA_REG, help="Poids de la regularisation (anti-binaire).")
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE, help="Taille des patches (ex: 256).")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Workers DataLoader (Windows: souvent 0).")
    p.add_argument("--max-files", type=int, default=MAX_FILES, help="Limiter le nombre de fichiers .nc (debug).")
    p.add_argument("--seed", type=int, default=SEED, help="Seed (reproductibilite partielle).")
    p.add_argument(
        "--device",
        default=DEVICE,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Choix du device (auto par defaut).",
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=USE_AMP, help="Activer AMP (CUDA uniquement).")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=AUGMENT, help="Augmentations simples (flips).")
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=SHUFFLE, help="Shuffle DataLoader.")
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=DROP_LAST, help="Drop dernier batch (utile BatchNorm).")
    p.add_argument("--debug-viz", action=argparse.BooleanOptionalAction, default=DEBUG_VIZ, help="Sauvegarder une image debug (1er batch / epoch).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    # Parametrage via CLI (sans casser l'usage en notebook via import).
    global DATA_DIR, SAVE_DIR, BATCH_SIZE, ACCUM_STEPS, EPOCHS, LEARNING_RATE, LAMBDA_REG, PATCH_SIZE, NUM_WORKERS
    global USE_AMP, AUGMENT, SHUFFLE, DROP_LAST, DEBUG_VIZ, DEVICE, SEED, MAX_FILES
    DATA_DIR = str(args.data_dir)
    SAVE_DIR = str(args.save_dir)
    BATCH_SIZE = int(args.batch_size)
    ACCUM_STEPS = int(args.accum_steps)
    EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.lr)
    LAMBDA_REG = float(args.lambda_reg)
    PATCH_SIZE = int(args.patch_size)
    NUM_WORKERS = int(args.num_workers)
    USE_AMP = bool(args.amp)
    AUGMENT = bool(args.augment)
    SHUFFLE = bool(args.shuffle)
    DROP_LAST = bool(args.drop_last)
    DEBUG_VIZ = bool(args.debug_viz)
    DEVICE = str(args.device)
    SEED = args.seed if args.seed is None else int(args.seed)
    MAX_FILES = args.max_files if args.max_files is None else int(args.max_files)

    train()


if __name__ == "__main__":
    main()
