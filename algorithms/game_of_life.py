import numpy as np
from numba import njit  # type: ignore
from dataclasses import dataclass
from typing import Any
from algorithms.base_config import AlgorithmBaseConfig, AlgorithmBaseFactory


@dataclass
class GameOfLifeConfig(AlgorithmBaseConfig):
    height: int
    width: int
    iterations: int
    simulation_seed: int
    seed_amount: int

    def __post_init__(self):
        super().__post_init__()
        if self.iterations <= 0:
            raise ValueError(f"Fatal: iterations must be > 0. Received {self.iterations}")


class GameOfLifeFactory(AlgorithmBaseFactory):
    _CONFIG_CLASS = GameOfLifeConfig
    _RULES = {
        "height": lambda rng: int(rng.integers(250, 1001)) * 2,
        "width": lambda rng: int(rng.integers(250, 1001)) * 2,
        "iterations": lambda rng: int(rng.integers(1000000, 2000000)),
        "simulation_seed": lambda rng: int(rng.integers(0, 4294967296)),
        "seed_amount": lambda rng: int(rng.integers(1, 11)),
    }


def run_game_of_life(
    i: int,
    config: GameOfLifeConfig,
    to_video: bool,
    duration: float,
    fps: int,
    batch_directory: str,
    background_image: str | None,
    blur_radius: int,
    blur_sigma: float,
    engine_config: dict[str, Any],
):
    task_rng = np.random.default_rng(config.simulation_seed)
    canvas = np.zeros((config.height, config.width), dtype=np.uint32)
    color_palette = config.get_palette_array()
    background_color = config.get_background_array()
    color_buffer = np.zeros((config.height, config.width, color_palette.shape[1]), dtype=np.uint8)
