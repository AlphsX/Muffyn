# %% [markdown]
# # Muffin vs Chihuahua — SOTA Classification Pipeline (Kaggle Version)
# **Architecture**: EVA-02-Large + Swin-V2-Base + ConvNeXt-Base Hybrid Ensemble
# **Techniques**: SAM Optimizer | EMA | CutMix/Mixup | 5-Fold CV | Consensus Pseudo-Labeling | AMP | 12-Pass TTA
# **References**: Google Research (SAM), Meta (EVA-02, DINOv2), NVIDIA (Training Best Practices)

# %% [markdown]
# ## Step 0 — Install Dependencies

# %%
import subprocess
import sys

def install_if_missing(package, pip_name=None):
    """Install package if not already available."""
    try:
        __import__(package)
        print(f"[Install] {package} already installed")
    except ImportError:
        print(f"[Install] Installing {pip_name or package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or package])
        print(f"[Install] {package} installed successfully")

# Install timm for EVA-02 model
install_if_missing("timm", "timm")

# %% [markdown]
# ## Step 1 — Imports & Configuration

# %%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import gc
import json
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import timm
    TIMM_AVAILABLE = True
    print(f"[Import] timm {timm.__version__}")
except ImportError:
    TIMM_AVAILABLE = False
    print("[Import] timm not available — EVA-02 will be skipped")

# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
#  Device (Kaggle: CUDA GPU or CPU)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    AMP_ENABLED, PIN_MEMORY = True, True
    print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    AMP_ENABLED, PIN_MEMORY = False, False
    print("[Device] CPU — WARNING: Training will be very slow!")

AMP_DTYPE = "cuda" if AMP_ENABLED else "cpu"

# ---------------------------------------------------------------------------
#  Data Paths (Kaggle)
# ---------------------------------------------------------------------------
BASE_DIR  = "/kaggle/input/cs-460-muffin-vs-chihuahua-classification-challenge"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "kaggle_test_final")
CKPT_DIR  = "/kaggle/working/checkpoints"
OUTPUT_DIR = "/kaggle/working"

os.makedirs(CKPT_DIR, exist_ok=True)

print(f"[Data] TRAIN    : {TRAIN_DIR} (exists={os.path.exists(TRAIN_DIR)})")
print(f"[Data] TEST     : {TEST_DIR}  (exists={os.path.exists(TEST_DIR)})")
print(f"[Data] CKPT     : {CKPT_DIR}")
print(f"[Data] OUTPUT   : {OUTPUT_DIR}")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training data not found at {TRAIN_DIR}")

# ---------------------------------------------------------------------------
#  Hyperparameters (Optimized for Kaggle GPU)
# ---------------------------------------------------------------------------
IMG_SIZE         = 384
BATCH_SIZE       = 8
GRAD_ACCUM       = 4         # Effective batch = 32
P1_EPOCHS        = 5         # Phase 1: Head-only warmup
P2_EPOCHS        = 30        # Phase 2: Full SAM fine-tune
P1_LR            = 1e-3
P2_LR            = 5e-5
WEIGHT_DECAY     = 0.05
LABEL_SMOOTHING  = 0.1
MIXUP_ALPHA      = 0.2
CUTMIX_ALPHA     = 1.0
MIX_PROB         = 0.5
GRAD_CLIP        = 1.0
PATIENCE         = 8
N_FOLDS          = 5
NUM_WORKERS      = 2         # Kaggle Linux supports multi-workers
SAM_RHO          = 0.05
PSEUDO_THRESHOLD = 0.98      # Minimum confidence for pseudo-labeling
EMA_DECAY        = 0.999
WARMUP_EPOCHS    = 2         # Linear LR warmup at start of Phase 2

# Model selection: EVA-02-Large requires timm
ARCHITECTURES = ["swin_v2_b", "convnext_base"]
if TIMM_AVAILABLE:
    ARCHITECTURES.append("eva02_large")

print(f"\n[Config] Architectures : {ARCHITECTURES}")
print(f"[Config] Ensemble size : {len(ARCHITECTURES) * N_FOLDS} base models")

# %% [markdown]
# ## Step 2 — Augmentation & Mixing Engine

# %%
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FixedCrop:
    """Deterministic crop at a fixed pixel position for TTA."""
    def __init__(self, top, left, height, width):
        self.top, self.left, self.height, self.width = top, left, height, width

    def __call__(self, img):
        return transforms.functional.crop(img, self.top, self.left, self.height, self.width)


_SZ, _LG = IMG_SIZE, int(IMG_SIZE * 1.15)

def _compose(*tfms):
    """Helper to build transform pipeline with normalization."""
    return transforms.Compose(list(tfms) + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

tta_transforms_list = [
    val_transforms,                                                                          # 1  Standard
    _compose(transforms.Resize((_SZ, _SZ)), transforms.RandomHorizontalFlip(p=1.0)),        # 2  H-Flip
    _compose(transforms.Resize((_LG, _LG)), transforms.CenterCrop(_SZ)),                    # 3  Zoom
    _compose(transforms.Resize((_LG, _LG)), transforms.CenterCrop(_SZ),
             transforms.RandomHorizontalFlip(p=1.0)),                                        # 4  Zoom+Flip
    _compose(transforms.Resize((_SZ, _SZ)), transforms.RandomRotation((10, 10))),            # 5  Rot+10
    _compose(transforms.Resize((_SZ, _SZ)), transforms.RandomRotation((-10, -10))),          # 6  Rot-10
    _compose(transforms.Resize((_SZ, _SZ)), transforms.ColorJitter(0.2, 0.2)),               # 7  Color
    _compose(transforms.Resize((_LG, _LG)), FixedCrop(0,       0,       _SZ, _SZ)),         # 8  TL
    _compose(transforms.Resize((_LG, _LG)), FixedCrop(0,       _LG-_SZ, _SZ, _SZ)),        # 9  TR
    _compose(transforms.Resize((_LG, _LG)), FixedCrop(_LG-_SZ, 0,       _SZ, _SZ)),        # 10 BL
    _compose(transforms.Resize((_LG, _LG)), FixedCrop(_LG-_SZ, _LG-_SZ, _SZ, _SZ)),       # 11 BR
    _compose(transforms.Resize((_SZ, _SZ)), transforms.GaussianBlur(3, sigma=0.5)),         # 12 Blur
]


def _rand_bbox(W, H, lam):
    """Random bounding box for CutMix."""
    cut = np.sqrt(1.0 - lam)
    cw, ch = int(W * cut), int(H * cut)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (int(np.clip(cx - cw // 2, 0, W)), int(np.clip(cy - ch // 2, 0, H)),
            int(np.clip(cx + cw // 2, 0, W)), int(np.clip(cy + ch // 2, 0, H)))


def apply_mixup_cutmix(images, labels):
    """Apply either Mixup or CutMix with 50/50 probability."""
    use_cutmix = random.random() > 0.5
    alpha = CUTMIX_ALPHA if use_cutmix else MIXUP_ALPHA
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)

    if use_cutmix:
        mixed = images.clone()
        x1, y1, x2, y2 = _rand_bbox(images.size(3), images.size(2), lam)
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3))
    else:
        mixed = lam * images + (1.0 - lam) * images[perm]

    return mixed, labels, labels[perm], lam


print(f"[Augmentation] {len(tta_transforms_list)} TTA passes configured.")

# %% [markdown]
# ## Step 3 — Datasets & Stratified K-Fold

# %%
class TransformSubset(Dataset):
    """Wraps an ImageFolder subset with a custom transform."""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """Loads test images from a flat directory."""
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.filenames = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image = Image.open(os.path.join(self.test_dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, name


class PseudoLabelDataset(Dataset):
    """Combines labeled training data with pseudo-labeled test images."""
    def __init__(self, original_dataset, original_indices, pseudo_paths, pseudo_labels, transform):
        self.original_dataset = original_dataset
        self.original_indices = original_indices
        self.pseudo_paths = pseudo_paths
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.original_indices) + len(self.pseudo_paths)

    def __getitem__(self, idx):
        if idx < len(self.original_indices):
            image, label = self.original_dataset[self.original_indices[idx]]
        else:
            pi = idx - len(self.original_indices)
            image = Image.open(self.pseudo_paths[pi]).convert("RGB")
            label = self.pseudo_labels[pi]
        if self.transform:
            image = self.transform(image)
        return image, label


full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=None)
CLASS_NAMES  = full_dataset.classes
IDX_TO_CLASS = {v: k for k, v in full_dataset.class_to_idx.items()}

skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(np.zeros(len(full_dataset)), full_dataset.targets))

print(f"[Dataset] Classes     : {CLASS_NAMES}")
print(f"[Dataset] Train images: {len(full_dataset)}")
print(f"[Dataset] Test images : {len(TestDataset(TEST_DIR))}")
print(f"[Dataset] K-Folds     : {N_FOLDS} (stratified)")

# %% [markdown]
# ## Step 4 — SAM Optimizer, EMA & Model Factory

# %%
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (Foret et al., Google Research 2021).

    Two-step weight update that seeks flat minima for better generalization.
    Used by Kaggle Grandmaster winning solutions and Google's internal training.
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                perturbation = p.grad * scale
                p.add_(perturbation)
                self.state[p]["perturbation"] = perturbation
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["perturbation"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), 2)

    def step(self, closure=None):
        pass


class ModelEMA:
    """Exponential Moving Average of model weights for stable inference."""
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad = False

    def update(self, source_model):
        with torch.no_grad():
            for ema_val, model_val in zip(
                self.model.state_dict().values(),
                source_model.state_dict().values(),
            ):
                ema_val.copy_(self.decay * ema_val + (1.0 - self.decay) * model_val)


def create_model(architecture, num_classes=2, freeze_backbone=True):
    """
    Model factory supporting:
      - swin_v2_b     (torchvision, 88M params)
      - convnext_base (torchvision, 89M params)
      - eva02_large   (timm, 304M params — strongest backbone, 90.0% ImageNet top-1)
    """
    if architecture == "swin_v2_b":
        model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        backbone_params = list(model.features.parameters())
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif architecture == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        backbone_params = list(model.features.parameters())
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif architecture == "eva02_large" and TIMM_AVAILABLE:
        model = timm.create_model(
            "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
            pretrained=True, num_classes=num_classes, img_size=IMG_SIZE,
        )
        head_names = {"head", "head_drop", "fc_norm"}
        backbone_params = [
            p for n, p in model.named_parameters()
            if not any(h in n for h in head_names)
        ]

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if freeze_backbone:
        for param in backbone_params:
            param.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{architecture}] {n_total:,} params total, {n_train:,} trainable")

    return model.to(device)


print("[Components] SAM, EMA, Model Factory ready.")

# %% [markdown]
# ## Step 5 — Checkpoint Manager (Google/NVIDIA Pattern)

# %%
PROGRESS_FILE = os.path.join(CKPT_DIR, "progress.json")


def _load_progress():
    """Load training progress from disk."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_folds": {}, "histories": {}}


def _save_progress(progress):
    """Persist training progress to disk."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def _fold_key(architecture, fold):
    """Unique key for architecture + fold combination."""
    return f"{architecture}_f{fold}"


def save_training_checkpoint(architecture, fold, epoch, model, ema, optimizer,
                             scheduler, scaler, best_val_acc, best_weights,
                             no_improve, history):
    """
    Save full training state for mid-fold resume.

    Follows Google/NVIDIA best practice: saves model + optimizer + scheduler +
    scaler + RNG states + metadata for exact reproducibility on resume.
    """
    path = os.path.join(CKPT_DIR, f"{_fold_key(architecture, fold)}_resume.pth")
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": ema.model.state_dict(),
        "optimizer_state": (
            optimizer.base_optimizer.state_dict()
            if isinstance(optimizer, SAM)
            else optimizer.state_dict()
        ),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "best_weights": best_weights,
        "no_improve": no_improve,
        "history": history,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    torch.save(state, path)


def load_training_checkpoint(architecture, fold):
    """Load training state if a resume checkpoint exists."""
    path = os.path.join(CKPT_DIR, f"{_fold_key(architecture, fold)}_resume.pth")
    if os.path.exists(path):
        return torch.load(path, weights_only=False, map_location=device)
    return None


def clear_resume_checkpoint(architecture, fold):
    """Remove resume checkpoint after fold completes successfully."""
    path = os.path.join(CKPT_DIR, f"{_fold_key(architecture, fold)}_resume.pth")
    if os.path.exists(path):
        os.remove(path)


print("[Checkpoint] Manager ready.")

# %% [markdown]
# ## Step 6 — Training Engine (with Checkpoint/Resume)

# %%
@torch.no_grad()
def evaluate(model, dataloader, criterion=None):
    """Evaluate model on a dataloader. Returns (accuracy, avg_loss, predictions, labels)."""
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        with autocast(AMP_DTYPE, enabled=AMP_ENABLED):
            logits = model(images)
            if criterion:
                running_loss += criterion(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / total if criterion else 0.0
    return accuracy, avg_loss, np.array(all_preds), np.array(all_labels)


def train_fold(architecture, fold_idx, save_path):
    """
    Two-phase training pipeline with full checkpoint/resume support.

    Phase 1 — Head warmup with AdamW + Gradient Accumulation (frozen backbone)
    Phase 2 — Full SAM fine-tune with EMA, Mixup/CutMix, Warmup + CosineWarmRestarts

    Can be stopped at any time — will resume from the last completed epoch.

    Returns: (best_val_acc, history_dict)
    """
    fold_key = _fold_key(architecture, fold_idx)
    progress = _load_progress()

    # Skip if already completed
    if fold_key in progress.get("completed_folds", {}):
        best = progress["completed_folds"][fold_key]
        hist = progress.get("histories", {}).get(fold_key, {})
        print(f"\n  [SKIP] {architecture} Fold {fold_idx} — already completed (best={best:.2f}%)")
        return best, hist

    print(f"\n{'=' * 60}")
    print(f"  {architecture} | Fold {fold_idx}")
    print(f"{'=' * 60}")

    # --- DataLoaders ---
    train_idx, val_idx = FOLDS[fold_idx]
    train_loader = DataLoader(
        TransformSubset(full_dataset, train_idx, train_transforms),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        TransformSubset(full_dataset, val_idx, val_transforms),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # --- Model & Training Components ---
    model = create_model(architecture, freeze_backbone=True)
    ema = ModelEMA(model, decay=EMA_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler(enabled=AMP_ENABLED)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "phase": []}
    best_val_acc, best_weights, no_improve = 0.0, None, 0
    start_epoch_p1, start_epoch_p2 = 1, 1

    # --- Check for resume checkpoint ---
    ckpt = load_training_checkpoint(architecture, fold_idx)
    resume_phase = None
    if ckpt:
        history = ckpt["history"]
        best_val_acc = ckpt["best_val_acc"]
        best_weights = ckpt["best_weights"]
        no_improve = ckpt["no_improve"]
        torch.set_rng_state(ckpt["rng_torch"])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])

        last_phase = history["phase"][-1] if history["phase"] else 0
        last_epoch = ckpt["epoch"]

        if last_phase == 1:
            start_epoch_p1 = last_epoch + 1
            resume_phase = 1
        else:
            start_epoch_p1 = P1_EPOCHS + 1  # skip Phase 1
            start_epoch_p2 = last_epoch + 1
            resume_phase = 2
        print(f"  [RESUME] Phase {resume_phase}, epoch {last_epoch + 1}")

    # ==================================================================
    #  Phase 1: Head Warmup (frozen backbone)
    # ==================================================================
    if start_epoch_p1 <= P1_EPOCHS:
        head_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=P1_LR, weight_decay=WEIGHT_DECAY,
        )
        if resume_phase == 1 and ckpt:
            model.load_state_dict(ckpt["model_state"])
            ema.model.load_state_dict(ckpt["ema_state"])
            head_optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])

        print(f"\n  Phase 1: Head warmup (epoch {start_epoch_p1}-{P1_EPOCHS})")
        for epoch in range(start_epoch_p1, P1_EPOCHS + 1):
            model.train()
            head_optimizer.zero_grad()
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

            for step, (images, labels) in enumerate(tqdm(train_loader, leave=False, desc=f"P1 E{epoch}")):
                images, labels = images.to(device), labels.to(device)
                with autocast(AMP_DTYPE, enabled=AMP_ENABLED):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss / GRAD_ACCUM).backward()
                epoch_loss += loss.item() * labels.size(0)
                epoch_correct += (logits.argmax(1) == labels).sum().item()
                epoch_total += labels.size(0)

                if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(head_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(head_optimizer)
                    scaler.update()
                    head_optimizer.zero_grad()
                    ema.update(model)

            train_acc = 100.0 * epoch_correct / epoch_total
            train_loss = epoch_loss / epoch_total
            val_acc, val_loss, _, _ = evaluate(ema.model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["phase"].append(1)

            marker = " *" if val_acc > best_val_acc else ""
            print(f"    E{epoch}/{P1_EPOCHS}  t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  "
                  f"t_acc={train_acc:.2f}%  v_acc={val_acc:.2f}%{marker}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(ema.model.state_dict())

            # Save checkpoint after each epoch
            dummy_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(head_optimizer, T_0=10)
            save_training_checkpoint(
                architecture, fold_idx, epoch, model, ema, head_optimizer,
                dummy_scheduler, scaler, best_val_acc, best_weights, no_improve, history,
            )

    # ==================================================================
    #  Phase 2: Full SAM Fine-tune with Warmup + CosineWarmRestarts
    # ==================================================================
    for param in model.parameters():
        param.requires_grad = True

    sam_optimizer = SAM(
        model.parameters(), optim.AdamW,
        rho=SAM_RHO, lr=P2_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        sam_optimizer.base_optimizer, T_0=10, T_mult=1, eta_min=1e-7,
    )

    if resume_phase == 2 and ckpt:
        model.load_state_dict(ckpt["model_state"])
        ema.model.load_state_dict(ckpt["ema_state"])
        sam_optimizer.base_optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Phase 2: SAM fine-tune ({n_trainable:,} params, epoch {start_epoch_p2}-{P2_EPOCHS})")

    for epoch in range(start_epoch_p2, P2_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        # Linear warmup: scale LR from 10% to 100% over WARMUP_EPOCHS
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = P2_LR * (0.1 + 0.9 * (epoch / WARMUP_EPOCHS))
            for param_group in sam_optimizer.base_optimizer.param_groups:
                param_group["lr"] = warmup_lr

        for images, labels in tqdm(train_loader, leave=False, desc=f"P2 E{epoch:02d}"):
            images, labels = images.to(device), labels.to(device)

            # Pre-compute mixed data ONCE for SAM consistency
            use_mix = random.random() < MIX_PROB
            if use_mix:
                mixed, labels_a, labels_b, lam = apply_mixup_cutmix(images, labels)
            else:
                mixed, labels_a, labels_b, lam = images, labels, labels, 1.0

            # SAM Pass 1: Compute gradients and perturb weights
            with autocast(AMP_DTYPE, enabled=AMP_ENABLED):
                logits = model(mixed)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            loss.backward()
            sam_optimizer.first_step(zero_grad=True)

            # SAM Pass 2: Compute gradients at perturbed point and update
            with autocast(AMP_DTYPE, enabled=AMP_ENABLED):
                logits = model(mixed)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            sam_optimizer.second_step(zero_grad=True)
            ema.update(model)

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(1) == labels_a).sum().item()
            epoch_total += labels.size(0)

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        train_acc = 100.0 * epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        val_acc, val_loss, _, _ = evaluate(ema.model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["phase"].append(2)

        marker = " * BEST" if val_acc > best_val_acc else ""
        print(f"    E{epoch:02d}/{P2_EPOCHS}  t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  "
              f"t_acc={train_acc:.2f}%  v_acc={val_acc:.2f}%{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(ema.model.state_dict())
            torch.save(best_weights, save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}.")
                break

        save_training_checkpoint(
            architecture, fold_idx, epoch, model, ema, sam_optimizer,
            scheduler, scaler, best_val_acc, best_weights, no_improve, history,
        )

    # --- Mark fold as completed ---
    progress = _load_progress()
    if "completed_folds" not in progress:
        progress["completed_folds"] = {}
    if "histories" not in progress:
        progress["histories"] = {}
    progress["completed_folds"][fold_key] = best_val_acc
    progress["histories"][fold_key] = history
    _save_progress(progress)
    clear_resume_checkpoint(architecture, fold_idx)

    del model, ema, sam_optimizer, criterion, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Finished. Best val accuracy: {best_val_acc:.2f}%\n")
    return best_val_acc, history


def ensemble_inference(model_configs, tta_list, test_dir):
    """Memory-efficient sequential ensemble inference: one model at a time."""
    softmax_fn = nn.Softmax(dim=1)
    accumulated_probs, filenames = None, None
    total_passes = len(model_configs) * len(tta_list)

    for model_idx, (arch, weight_path) in enumerate(model_configs):
        print(f"  [{model_idx + 1}/{len(model_configs)}] {arch} <- {os.path.basename(weight_path)}")
        model = create_model(arch, freeze_backbone=False)
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
        model.eval()

        for tta_transform in tta_list:
            loader = DataLoader(
                TestDataset(test_dir, transform=tta_transform),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )
            batch_probs, batch_names = [], []
            with torch.no_grad():
                for images, names in loader:
                    with autocast(AMP_DTYPE, enabled=AMP_ENABLED):
                        probs = softmax_fn(model(images.to(device))).cpu().numpy()
                        batch_probs.append(probs)
                    if filenames is None:
                        batch_names.extend(names)

            pass_probs = np.concatenate(batch_probs, axis=0)
            if accumulated_probs is None:
                accumulated_probs, filenames = pass_probs, batch_names
            else:
                accumulated_probs += pass_probs

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accumulated_probs /= total_passes
    return accumulated_probs, filenames


print("[Engine] Training and inference ready (with checkpoint/resume).")

# %% [markdown]
# ## Step 7 — Visualization Utilities

# %%
def plot_training_history(history, title="Training History"):
    """Dual-panel plot: accuracy and loss curves with phase boundary."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    phase_boundary = sum(1 for p in history["phase"] if p == 1)

    for ax, suffix, ylabel, loc in [
        (ax_acc, "_acc", "Accuracy (%)", "lower right"),
        (ax_loss, "_loss", "Loss", "upper right"),
    ]:
        ax.plot(epochs, history["train" + suffix], "b-o", ms=3, label="Train")
        ax.plot(epochs, history["val" + suffix], "r-o", ms=3, label="Validation")
        if phase_boundary < len(epochs):
            ax.axvline(x=phase_boundary + 0.5, color="gray", ls="--", alpha=0.5, label="Phase 1|2")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc=loc)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, title.replace(" ", "_").lower() + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Heatmap confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.72, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, title.replace(" ", "_").lower() + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def print_classification_report(y_true, y_pred, class_names, title="Classification Report"):
    """Print sklearn classification report with header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


print("[Visualization] Ready.")

# %% [markdown]
# ## Step 8 — Train All Folds (Auto-Resumes)

# %%
all_results = {}
all_histories = {}

for arch in ARCHITECTURES:
    all_results[arch] = {}
    all_histories[arch] = {}

    for fold in range(N_FOLDS):
        save_name = os.path.join(CKPT_DIR, f"best_{arch}_f{fold}.pth")
        acc, hist = train_fold(arch, fold, save_name)
        all_results[arch][fold] = acc
        all_histories[arch][fold] = hist

    avg = np.mean(list(all_results[arch].values()))
    print(f"\n[{arch}] Average: {avg:.2f}%")
    if all_histories[arch].get(0):
        plot_training_history(all_histories[arch][0], f"{arch} Fold 0")

# Training Summary
print("\n" + "=" * 60)
print("  Training Summary")
print("=" * 60)
for arch in ARCHITECTURES:
    avg = np.mean(list(all_results[arch].values()))
    print(f"  {arch:20s}: {avg:.2f}% avg")

# %% [markdown]
# ## Step 9 — Classification Report & Confusion Matrix

# %%
print("\n" + "=" * 60)
print("  Evaluation on Fold 0 Validation Set")
print("=" * 60)

val_loader = DataLoader(
    TransformSubset(full_dataset, FOLDS[0][1], val_transforms),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

for arch in ARCHITECTURES:
    weight_path = os.path.join(CKPT_DIR, f"best_{arch}_f0.pth")
    if not os.path.exists(weight_path):
        print(f"  {arch}: checkpoint not found, skipping")
        continue
    model = create_model(arch, freeze_backbone=False)
    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
    acc, loss, preds, labels = evaluate(model, val_loader, criterion)
    print(f"\n  {arch} -> val_acc={acc:.2f}%  val_loss={loss:.4f}")
    print_classification_report(labels, preds, CLASS_NAMES, f"{arch}")
    plot_confusion_matrix(labels, preds, CLASS_NAMES, f"{arch} Confusion Matrix")
    del model
    gc.collect()

# %% [markdown]
# ## Step 10 — Round 1 Ensemble Inference

# %%
print("\n" + "=" * 60)
round1_configs = []
for arch in ARCHITECTURES:
    for fold in range(N_FOLDS):
        path = os.path.join(CKPT_DIR, f"best_{arch}_f{fold}.pth")
        if os.path.exists(path):
            round1_configs.append((arch, path))

n_passes = len(round1_configs) * len(tta_transforms_list)
print(f"  Round 1: {len(round1_configs)} models x {len(tta_transforms_list)} TTA = {n_passes} passes")
print("=" * 60)

r1_probs, r1_filenames = ensemble_inference(round1_configs, tta_transforms_list, TEST_DIR)
r1_pred_idx = np.argmax(r1_probs, axis=1)
r1_preds = [IDX_TO_CLASS[i] for i in r1_pred_idx]
r1_conf = np.max(r1_probs, axis=1)

print(f"\n  Confidence  mean={r1_conf.mean():.4f}  min={r1_conf.min():.4f}")
print(f"  High-conf (>{PSEUDO_THRESHOLD}): {(r1_conf > PSEUDO_THRESHOLD).sum()}/{len(r1_conf)}")

sub_r1 = pd.DataFrame({"ID": r1_filenames, "Predict": r1_preds})
sub_r1.to_csv(os.path.join(OUTPUT_DIR, "submission_r1.csv"), index=False)
print(f"\n  Round 1 saved -> submission_r1.csv ({len(sub_r1)} rows)")

# %% [markdown]
# ## Step 11 — Pseudo-Labeling

# %%
print("\n" + "=" * 60)
print("  Pseudo-Labeling")
print("=" * 60)

# Build pseudo-label candidates based on confidence threshold
pseudo_paths = []
pseudo_labels = []
n_high_conf = 0
class_to_idx = full_dataset.class_to_idx

for i in range(len(r1_filenames)):
    if r1_conf[i] <= PSEUDO_THRESHOLD:
        continue
    n_high_conf += 1

    pseudo_paths.append(os.path.join(TEST_DIR, r1_filenames[i]))
    pseudo_labels.append(class_to_idx[r1_preds[i]])

n_pseudo = len(pseudo_paths)
print(f"  High-confidence (>{PSEUDO_THRESHOLD}): {n_high_conf}/{len(r1_conf)}")
print(f"  Final pseudo-labels: {n_pseudo}")

if n_pseudo > 0:
    label_counts = dict(zip(*np.unique(pseudo_labels, return_counts=True)))
    print(f"  Label distribution: {label_counts}")

    for arch in ARCHITECTURES:
        base_path = os.path.join(CKPT_DIR, f"best_{arch}_f0.pth")
        pseudo_save_path = os.path.join(CKPT_DIR, f"best_{arch}_pseudo.pth")

        if not os.path.exists(base_path):
            continue

        # Skip if already completed
        if os.path.exists(pseudo_save_path):
            print(f"\n  [SKIP] {arch} pseudo-labeling — already completed")
            continue

        print(f"\n  Pseudo-retrain: {arch}")

        train_idx, val_idx = FOLDS[0]
        pseudo_train_loader = DataLoader(
            PseudoLabelDataset(full_dataset, train_idx, pseudo_paths, pseudo_labels, train_transforms),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        pseudo_val_loader = DataLoader(
            TransformSubset(full_dataset, val_idx, val_transforms),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        model = create_model(arch, freeze_backbone=False)
        model.load_state_dict(torch.load(base_path, weights_only=True, map_location=device))
        ema = ModelEMA(model, EMA_DECAY)
        crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        opt = optim.AdamW(model.parameters(), lr=P2_LR * 0.5, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=1, eta_min=1e-7)
        grad_scaler = GradScaler(enabled=AMP_ENABLED)
        best_pseudo_acc, no_improve_pseudo = 0.0, 0

        for ep in range(1, 11):
            model.train()
            for images, labels in tqdm(pseudo_train_loader, leave=False, desc=f"Pseudo E{ep}"):
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                use_mix = random.random() < MIX_PROB
                if use_mix:
                    mixed, la, lb, lam = apply_mixup_cutmix(images, labels)
                else:
                    mixed, la, lb, lam = images, labels, labels, 1.0
                with autocast(AMP_DTYPE, enabled=AMP_ENABLED):
                    logits = model(mixed)
                    loss = lam * crit(logits, la) + (1.0 - lam) * crit(logits, lb)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                grad_scaler.step(opt)
                grad_scaler.update()
                ema.update(model)
            sched.step()

            val_acc, val_loss, _, _ = evaluate(ema.model, pseudo_val_loader, crit)
            marker = " *" if val_acc > best_pseudo_acc else ""
            print(f"    E{ep:02d}/10  val_acc={val_acc:.2f}%  val_loss={val_loss:.4f}{marker}")

            if val_acc > best_pseudo_acc:
                best_pseudo_acc = val_acc
                torch.save(copy.deepcopy(ema.model.state_dict()), pseudo_save_path)
                no_improve_pseudo = 0
            else:
                no_improve_pseudo += 1
                if no_improve_pseudo >= 5:
                    print(f"    Early stopping at epoch {ep}.")
                    break

        del model, ema, opt, crit, grad_scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {arch} pseudo best: {best_pseudo_acc:.2f}%")
else:
    print("  No pseudo-label candidates. Skipping.")

# %% [markdown]
# ## Step 12 — Final Ensemble Submission

# %%
print("\n" + "=" * 60)
print("  Final Ensemble (All Models + Pseudo)")
print("=" * 60)

final_configs = []
for arch in ARCHITECTURES:
    for fold in range(N_FOLDS):
        path = os.path.join(CKPT_DIR, f"best_{arch}_f{fold}.pth")
        if os.path.exists(path):
            final_configs.append((arch, path))
    pseudo_path = os.path.join(CKPT_DIR, f"best_{arch}_pseudo.pth")
    if os.path.exists(pseudo_path):
        final_configs.append((arch, pseudo_path))

print(f"  Models in final ensemble: {len(final_configs)}")

final_probs, final_filenames = ensemble_inference(final_configs, tta_transforms_list, TEST_DIR)
final_preds = [IDX_TO_CLASS[i] for i in np.argmax(final_probs, axis=1)]
final_conf = np.max(final_probs, axis=1)

submission = pd.DataFrame({"ID": final_filenames, "Predict": final_preds})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

print(f"\n  submission.csv saved ({len(submission)} rows)")
print(f"  Confidence  mean={final_conf.mean():.4f}  min={final_conf.min():.4f}")
print(f"\n{submission['Predict'].value_counts().to_string()}")
print("\n  Pipeline complete. Upload submission.csv to Kaggle.")
