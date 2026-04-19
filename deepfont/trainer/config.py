"""Pydantic configuration classes for DeepFont trainers.

These configuration classes parameterize the training loop, optimizer,
scheduler, and trainer-level data handling settings. Data source and
model architecture settings live on their own dedicated config classes
(see ``deepfont.data.config`` and ``deepfont.models.config``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, BaseModel, ConfigDict, field_validator

OptimizerType = Literal["adam", "adamw", "sgd"]


class TrainerConfig(BaseModel):
    """Shared configuration for hardware, loop control, checkpointing, and logging.

    This config is inherited by both PretrainConfig and FinetuneConfig and
    covers everything that is common to both training regimes.
    """

    model_config = ConfigDict(frozen=True)

    # Hardware / Fabric
    accelerator: str = Field(default="auto")
    devices: int | str = Field(default="auto")
    strategy: str = Field(default="auto")
    precision: str = Field(default="32-true")
    num_workers: int = Field(default=4, ge=0)

    # Training loop
    max_epochs: int | None = Field(default=100)
    batch_size: int = Field(default=64, ge=1)
    val_batch_size: int = Field(default=64, ge=1)
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    grad_accum_steps: int = Field(default=1, ge=1)
    gradient_clip_val: float | None = Field(default=None)
    limit_train_batches: int | None = Field(
        default=None,
        ge=1,
        description="Cap the number of training batches per epoch. None = use all.",
    )
    limit_val_batches: int | None = Field(
        default=None,
        ge=1,
        description="Cap the number of validation batches per epoch. None = use all.",
    )

    # Validation
    val_frequency: int = Field(default=1, ge=1)  # run a validation epoch every N training epochs

    # Checkpointing
    checkpoint_dir: str = Field(default="./checkpoints")
    checkpoint_frequency: int = Field(default=1, ge=1)  # save a checkpoint every N epochs

    # Logging
    log_every_n_steps: int = Field(default=10, ge=1)  # log training metrics every N optimizer steps

    # Reproducibility
    seed: int | None = Field(default=None)

    @field_validator("max_epochs", mode="after")
    @classmethod
    def _max_epochs_positive_if_set(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_epochs must be >= 1 when set.")
        return v


class PretrainConfig(TrainerConfig):
    """Configuration for the PretrainTrainer autoencoder stage.

    Adds trainer-level data handling, optimizer hyper-parameters, and loss
    choice on top of the shared TrainerConfig. Data source and model
    architecture settings are provided separately via PretrainDataConfig
    and DeepFontAEConfig.
    """

    # Trainer-level data handling
    upsample_real_images: bool = Field(
        default=True,
        description="Match real image count to synthetic count via resampling.",
    )
    num_images_to_cache: int = Field(
        default=0,
        ge=0,
        description="Pre-load this many images into RAM; 0 = disabled.",
    )

    # Optimizer
    optimizer_type: OptimizerType = Field(default="sgd")
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer_kwargs: dict = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to the optimizer constructor. "
            "SGD: momentum, dampening, nesterov. "
            "Adam/AdamW: betas, eps, amsgrad."
        ),
    )

    # LR Scheduler (optional)
    scheduler_type: Literal["cosine", "step", "reduce_on_plateau", "linear"] | None = Field(
        default=None,
    )
    scheduler_kwargs: dict = Field(default_factory=dict)

    # Loss
    reconstruction_loss: Literal["mse", "l1"] = Field(default="mse")


class FinetuneConfig(TrainerConfig):
    """Configuration for the FinetuneTrainer classification stage.

    Extends TrainerConfig with trainer-level data handling, optional
    transfer-learning from a pretrained encoder, and optimizer
    hyper-parameters. Data source, model architecture, and evaluation
    settings are provided separately via FinetuneDataConfig,
    DeepFontConfig, and EvalDataConfig.
    """

    # Trainer-level data handling
    num_images_to_cache: int = Field(
        default=0,
        ge=0,
        description="Pre-load this many images into RAM; 0 = disabled.",
    )

    # Transfer learning
    encoder_weights_path: str | None = Field(
        default=None,
        description="Path to pretrained AE weights for transfer learning.",
    )

    # Optimizer
    optimizer_type: OptimizerType = Field(default="sgd")
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    optimizer_kwargs: dict = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to the optimizer constructor. "
            "SGD: momentum, dampening, nesterov. "
            "Adam/AdamW: betas, eps, amsgrad."
        ),
    )

    # LR Scheduler (optional)
    scheduler_type: Literal["cosine", "step", "reduce_on_plateau", "linear"] | None = Field(
        default=None,
    )
    scheduler_kwargs: dict = Field(default_factory=dict)

    # Evaluation
    limit_eval_batches: int | None = Field(
        default=None,
        ge=1,
        description="Cap the number of evaluation batches in evaluate(). None = use all.",
    )
