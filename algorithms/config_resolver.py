import argparse
import json
from dataclasses import dataclass, fields, asdict, replace
from typing import Any
from typing import Callable, Any, ClassVar, Type
import numpy as np
from PIL import Image
import subprocess


@dataclass(kw_only=True)
class EngineConfig:
    algorithm: str
    json: str | None = None
    batch_size: int = 1
    master_seed: int | None = None
    image: str | None = None
    show_metadata: bool = False
    unlock: list[str] | None = None


@dataclass(kw_only=True)
class RendererConfig:
    to_video: bool = False
    duration: float = 30.0
    fps: int = 60
    background_image: str | None = None
    norm_method: int = 0
    norm_power: float = 1.0
    sig_steepness: float = 1.0
    sig_midpoint: float = 0.5
    blur_radius: int = 0
    blur_sigma: float = 0.0
    scaling_factor: int = 1


@dataclass(kw_only=True)
class AlgorithmBaseConfig:
    height: int = 1920
    width: int = 1080
    color_palette: int = 0
    background_color: int = 0

    COLOR_PALETTES: ClassVar[tuple[list[list[int]], ...]] = (
        [[0, 0, 0, 255], [255, 255, 255, 255]],
        [[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255]],
        [[0, 0, 50, 255], [0, 150, 255, 255], [200, 255, 255, 255]],
        [[20, 0, 40, 255], [255, 0, 255, 255], [0, 255, 255, 255]],
        [[10, 30, 15, 255], [0, 160, 70, 255], [215, 175, 55, 255]],
        [[92, 64, 51, 255], [52, 199, 89, 255]],
        [[0, 0, 0, 255]],
    )

    BG_COLORS: ClassVar[tuple[list[int], ...]] = (
        [0, 0, 0, 0],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 0],
    )

    def __post_init__(self):
        if not (0 <= self.color_palette < len(self.COLOR_PALETTES)):
            raise ValueError(f"Fatal: Invalid palette index '{self.color_palette}'. " f"Valid options: 0-{len(self.COLOR_PALETTES) - 1}")
        if not (0 <= self.background_color < len(self.BG_COLORS)):
            raise ValueError(f"Fatal: Invalid background index '{self.background_color}'. " f"Valid options: 0-{len(self.BG_COLORS) - 1}")

    def get_palette_array(self) -> np.ndarray:
        return np.array(self.COLOR_PALETTES[self.color_palette], dtype=np.uint8)

    def get_background_array(self) -> np.ndarray:
        return np.array(self.BG_COLORS[self.background_color], dtype=np.uint8)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class AlgorithmBaseFactory:
    _CONFIG_CLASS: ClassVar[Type[Any]]
    _RULES: ClassVar[dict[str, Callable[[np.random.Generator], Any]]] = {
        "height": lambda rng: int(rng.integers(250, 1001)) & ~1,
        "width": lambda rng: int(rng.integers(250, 1001)) & ~1,
        "color_palette": lambda rng: int(rng.integers(6)),
        "background_color": lambda rng: int(rng.integers(4)),
    }

    @classmethod
    def generate_random(cls, rng: np.random.Generator) -> Any:
        new_values = {key: rule(rng) for key, rule in cls._RULES.items()}
        return cls._CONFIG_CLASS(**new_values)

    @classmethod
    def unlock(cls, base_config: Any, rng: np.random.Generator, unlocked_keys: list[str]) -> Any:
        new_values = {key: cls._RULES[key](rng) for key in unlocked_keys}
        return replace(base_config, **new_values)


def extract_locks(parsed_args: argparse.Namespace, config_class: type) -> dict[str, Any]:
    locks: dict[str, Any] = {}

    for f in fields(config_class):
        raw_value = getattr(parsed_args, f.name, None)
        if raw_value is not None:
            locks[f.name] = raw_value

    return locks


def load_configs(parsed_args: argparse.Namespace, target_config: Any, target_factory: Any) -> tuple[EngineConfig, RendererConfig, list[Any]]:

    engine_config_locks = extract_locks(parsed_args, EngineConfig)
    renderer_config_locks = extract_locks(parsed_args, RendererConfig)
    algorithm_config_locks = extract_locks(parsed_args, target_config)

    config_unlocks: list[str] = getattr(parsed_args, "unlock", None) or []

    rng = np.random.default_rng(parsed_args.master_seed)

    base_engine_config: dict[str, Any] = {}
    base_renderer_config: dict[str, Any] = {}
    base_algorithm_config: dict[str, Any] = {}

    if parsed_args.video:
        result = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format_tags=description", "-of", "default=nw=1:nk=1", parsed_args.video], capture_output=True, text=True)
        full_config = json.loads(result.stdout.strip())
        base_engine_config = full_config.get("engine", {})
        base_renderer_config = full_config.get("renderer", {})
        base_algorithm_config = full_config.get("algorithm", {})

    elif parsed_args.image:
        with Image.open(parsed_args.image) as image:
            full_config = json.loads(image.info.get("MimikryConfig", "{}"))
            base_engine_config = full_config.get("engine", {})
            base_renderer_config = full_config.get("renderer", {})
            base_algorithm_config = full_config.get("algorithm", {})

    elif parsed_args.json:
        with open(parsed_args.json, "r") as f:
            full_config = json.load(f)
            base_engine_config = full_config.get("engine", {})
            base_renderer_config = full_config.get("renderer", {})
            base_algorithm_config = full_config.get("algorithm", {})

    base_engine_config.update(engine_config_locks)
    base_renderer_config.update(renderer_config_locks)

    for field in config_unlocks:
        base_algorithm_config.pop(field, None)

    base_algorithm_config.update(algorithm_config_locks)

    engine_config = EngineConfig(**base_engine_config)
    renderer_config = RendererConfig(**base_renderer_config)

    algorithm_configs: list[Any] = []
    for _ in range(getattr(engine_config, "batch_size", 1)):
        random_base = target_factory.generate_random(rng)
        final_config = replace(random_base, **base_algorithm_config)
        algorithm_configs.append(final_config)

    return engine_config, renderer_config, algorithm_configs
