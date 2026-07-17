import numpy as np
from numpy.typing import NDArray
from numba import njit  # type: ignore
from dataclasses import dataclass
from typing import Callable, Any, ClassVar
from renderer import RendererConfig
from algorithms.base_config import AlgorithmBaseConfig, AlgorithmBaseFactory, run_algorithm


@dataclass
class GameOfLifeConfig(AlgorithmBaseConfig):
    iterations: int
    simulation_seed: int
    living_ratio: float

    def __post_init__(self):
        super().__post_init__()
        if self.iterations <= 0:
            raise ValueError(f"Fatal: iterations must be > 0. Received {self.iterations}")


class GameOfLifeFactory(AlgorithmBaseFactory):
    _CONFIG_CLASS = GameOfLifeConfig
    _GAME_OF_LIFE_RULES: ClassVar[dict[str, Callable[[np.random.Generator], Any]]] = {
        "iterations": lambda rng: int(rng.integers(10, 250)),
        "simulation_seed": lambda rng: int(rng.integers(0, 4294967296)),
        "living_ratio": lambda rng: float(rng.uniform(0.1, 0.9)),
    }

    _RULES: ClassVar[dict[str, Callable[[np.random.Generator], Any]]] = AlgorithmBaseFactory._RULES | _GAME_OF_LIFE_RULES


def run_game_of_life(
    i: int,
    algorithm_config: GameOfLifeConfig,
    renderer_config: RendererConfig,
    batch_directory: str,
    engine_config: dict[str, Any],
):
    def build_generator(task_rng: np.random.Generator, canvas: np.ndarray, capture_interval: int):
        return game_of_life(
            task_rng,
            canvas,
            algorithm_config.iterations,
            algorithm_config.living_ratio,
            capture_interval,
        )

    run_algorithm(i, algorithm_config, renderer_config, batch_directory, engine_config, build_generator)


@njit(cache=True)  # type: ignore
def game_of_life(
    rng: np.random.Generator,
    canvas: NDArray[np.uint32],
    iterations: int,
    living_ratio: float,
    capture_interval: int,
):
    height = canvas.shape[0]
    width = canvas.shape[1]

    inital_alive = int(height * width * living_ratio)

    for _ in range(inital_alive):
        sy = rng.integers(0, height)  # type: ignore
        sx = rng.integers(0, width)  # type: ignore
        canvas[sy, sx] = 1

    next_canvas = np.empty((height, width), dtype=np.uint32)

    if capture_interval > 0:
        yield 0

    for i in range(1, iterations + 1):
        for y in range(height):
            for x in range(width):
                alive_neighbors = 0

                neighborhood_top_edge = max(0, y - 1)
                neighborhood_bottom_edge = min(height, y + 2)
                neighborhood_left_edge = max(0, x - 1)
                neighborhood_right_edge = min(width, x + 2)

                for ny in range(neighborhood_top_edge, neighborhood_bottom_edge):
                    for nx in range(neighborhood_left_edge, neighborhood_right_edge):
                        if ny == y and nx == x:
                            continue
                        if canvas[ny, nx] > 0:
                            alive_neighbors += 1

                if canvas[y, x] > 0:
                    if alive_neighbors == 2 or alive_neighbors == 3:
                        next_canvas[y, x] = 1
                    else:
                        next_canvas[y, x] = 0
                else:
                    if alive_neighbors == 3:
                        next_canvas[y, x] = 1
                    else:
                        next_canvas[y, x] = 0

        for y in range(height):
            for x in range(width):
                canvas[y, x] = next_canvas[y, x]

        if capture_interval > 0 and i % capture_interval == 0:
            yield 0

    yield 1
