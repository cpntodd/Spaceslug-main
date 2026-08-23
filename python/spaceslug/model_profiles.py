"""Intended Spaceslug model-size profiles and resolved planning metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    target_parameters: int
    hidden_size: int
    layers: int
    attention_heads: int
    context_length: int
    training_mode: str = "full"
    dtype: str = "float32"
    lora_rank: int = 0

    @property
    def estimated_parameters(self) -> int:
        # Approximate decoder-only block + embeddings/LM head estimate.
        return self.target_parameters if self.model_id != "Spaceslug-Tiny" else self.hidden_size * self.hidden_size * 8 + self.hidden_size * 259 * 2

    def with_overrides(self, **overrides: object) -> "ModelProfile":
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown model overrides: {sorted(unknown)}")
        return replace(self, **overrides)

    def validate(self) -> None:
        if self.target_parameters <= 0 or self.hidden_size <= 0 or self.layers <= 0 or self.attention_heads <= 0 or self.context_length <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.training_mode not in ("full", "lora", "inference-only"):
            raise ValueError("unsupported training_mode")


_PROFILES = {
    "Spaceslug-Tiny": ModelProfile("Spaceslug-Tiny", 1_000_000, 64, 2, 4, 512, "lora", "float32", 4),
    "Spaceslug-0.1B": ModelProfile("Spaceslug-0.1B", 100_000_000, 768, 12, 12, 2048),
    "Spaceslug-0.5B": ModelProfile("Spaceslug-0.5B", 500_000_000, 1024, 16, 16, 2048, "lora"),
    "Spaceslug-1B": ModelProfile("Spaceslug-1B", 1_000_000_000, 2048, 16, 16, 4096, "lora"),
    "Spaceslug-3.5B": ModelProfile("Spaceslug-3.5B", 3_500_000_000, 3072, 28, 24, 4096, "inference-only"),
}


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_profile(model_id: str) -> ModelProfile:
    try:
        profile = _PROFILES[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model profile: {model_id}") from exc
    profile.validate()
    return profile


def resolve_profile(model_id: str, **overrides: object) -> dict:
    profile = get_profile(model_id).with_overrides(**overrides) if overrides else get_profile(model_id)
    profile.validate()
    return {**asdict(profile), "estimated_parameters": profile.estimated_parameters}
