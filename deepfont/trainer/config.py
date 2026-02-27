from dataclasses import field, dataclass


@dataclass
class TrainerConfig:
    """Shared configuration for hardware, loop control, checkpointing, and logging.

    This config is inherited by both ``PretrainConfig`` and ``FinetuneConfig`` and
    covers everything that is common to both training regimes.
    """

    # --- Hardware / Fabric ---
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str = "32-true"
    num_workers: int = 4

    # --- Training loop ---
    max_epochs: int | None = 100
    batch_size: int = 64
    val_batch_size: int = 64
    train_ratio: float = 0.8
    grad_accum_steps: int = 1
    gradient_clip_val: float | None = None

    # --- Validation ---
    val_frequency: int = 1  # run a validation epoch every N training epochs

    # --- Checkpointing ---
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 1  # save a checkpoint every N epochs

    # --- Logging ---
    log_every_n_steps: int = 10  # log training metrics every N optimiser steps

    # --- Reproducibility ---
    seed: int | None = None


@dataclass
class PretrainConfig(TrainerConfig):
    """Configuration for the :class:`PretrainTrainer` autoencoder stage.

    Adds data paths, augmentation settings, loss choice, and optimiser
    hyper-parameters on top of the shared :class:`TrainerConfig`.
    """

    # --- Data ---
    bcf_store_file: str = ""
    data_folder_name: str | None = None  # directory of real images; None = synthetic only
    aug_prob: float = 0.5
    image_normalization: str = "0to1"  # "0to1" or "-1to1"
    upsample_real_images: bool = True  # match real count to synthetic count via resampling
    num_images_to_cache: int = 0  # pre-load this many images into RAM; 0 = disabled

    # --- Optimiser ---
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # --- LR Scheduler (optional) ---
    scheduler_type: str | None = None  # "cosine" | "step" | "reduce_on_plateau" | None
    scheduler_kwargs: dict = field(default_factory=dict)

    # --- Loss ---
    reconstruction_loss: str = "mse"  # "mse" or "l1"

    # --- Model ---
    output_activation: str | None = None  # None | "sigmoid" | "relu"


@dataclass
class FinetuneConfig(TrainerConfig):
    """Configuration for the :class:`FinetuneTrainer` classification stage.

    Extends :class:`TrainerConfig` with data paths, class count, optional
    transfer-learning from a pretrained encoder, and TTA evaluation settings.
    """

    # --- Train / val data ---
    bcf_store_file: str = ""
    label_file: str = ""
    aug_prob: float = 0.5
    image_normalization: str = "0to1"  # "0to1" or "-1to1"
    num_images_to_cache: int = 0

    # --- TTA evaluation data (used by evaluate(), not by fit()) ---
    eval_bcf_store_file: str = ""
    eval_label_file: str = ""
    num_image_crops: int = 15  # number of augmented crops per image during TTA

    # --- Optimiser ---
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # --- LR Scheduler (optional) ---
    scheduler_type: str | None = None  # "cosine" | "step" | "reduce_on_plateau" | None
    scheduler_kwargs: dict = field(default_factory=dict)

    # --- Model ---
    num_classes: int = 2383
    encoder_weights_path: str | None = None  # path to pretrained AE weights for transfer
