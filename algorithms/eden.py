import numpy as np
from numpy.typing import NDArray
from numba import njit  # type: ignore
from dataclasses import dataclass
from typing import Callable, Any, ClassVar
from algorithms.base_config import AlgorithmBaseConfig, AlgorithmBaseFactory, run_algorithm


@dataclass
class EdenConfig(AlgorithmBaseConfig):
    iterations: int
    simulation_seed: int
    seed_amount: int
    perimeter_size: int
    perimeter_filled: bool
    bias: int
    bias_power: float
    max_neighbor_weirdness: bool
    minkowski_radius: int
    minkowski_power: float
    tournament_size: int

    def __post_init__(self):
        super().__post_init__()
        if self.minkowski_radius <= 0:
            raise ValueError(f"Fatal: minkowski_radius must be > 0. Received {self.minkowski_radius}")
        if self.iterations <= 0:
            raise ValueError(f"Fatal: iterations must be > 0. Received {self.iterations}")
        if self.tournament_size < 1:
            raise ValueError(f"Tournament size must be > 0. Received {self.tournament_size}")


class EdenFactory(AlgorithmBaseFactory):
    _CONFIG_CLASS = EdenConfig
    _EDEN_RULES: ClassVar[dict[str, Callable[[np.random.Generator], Any]]] = {
        "iterations": lambda rng: int(rng.integers(1000000, 2000000)),
        "simulation_seed": lambda rng: int(rng.integers(0, 4294967296)),
        "seed_amount": lambda rng: int(rng.integers(1, 11)),
        "perimeter_size": lambda rng: int(rng.integers(1, 4)),
        "perimeter_filled": lambda rng: bool(rng.choice([True, False])),
        "bias": lambda rng: int(rng.integers(0, 6)),
        "bias_power": lambda rng: float(rng.uniform(0.5, 3.0)),
        "max_neighbor_weirdness": lambda rng: bool(rng.choice([True, False])),
        "minkowski_radius": lambda rng: int(rng.integers(1, 51)),
        "minkowski_power": lambda rng: float(rng.uniform(0.5, 5.0)),
        "tournament_size": lambda rng: int(rng.integers(2, 21)),
    }

    _RULES: ClassVar[dict[str, Callable[[np.random.Generator], Any]]] = AlgorithmBaseFactory._RULES | _EDEN_RULES


def run_eden(
    i: int,
    config: EdenConfig,
    to_video: bool,
    duration: float,
    fps: int,
    batch_directory: str,
    background_image: str | None,
    blur_radius: int,
    blur_sigma: float,
    engine_config: dict[str, Any],
):
    def build_generator(task_rng: np.random.Generator, canvas: np.ndarray, capture_interval: int):
        return eden(
            task_rng,
            canvas,
            config.iterations,
            config.seed_amount,
            config.perimeter_size,
            config.perimeter_filled,
            config.bias,
            config.bias_power,
            config.max_neighbor_weirdness,
            config.minkowski_radius,
            config.minkowski_power,
            config.tournament_size,
            capture_interval,
        )

    run_algorithm(i, config, to_video, duration, fps, batch_directory, background_image, blur_radius, blur_sigma, engine_config, build_generator)


@njit(cache=True)  # type: ignore
def eden(
    rng: np.random.Generator,
    canvas: NDArray[np.uint32],
    iterations: int,
    seed_amount: int,
    perimeter_size: int,
    perimeter_filled: bool,
    bias: int,
    bias_power: float,
    max_neighbor_weirdness: bool,
    minkowski_radius: int,
    minkowski_power: float,
    tournament_size: int,
    capture_interval: int,
):
    height = canvas.shape[0]
    width = canvas.shape[1]

    max_candidates = height * width
    candidates = np.empty((max_candidates, 2), dtype=np.int32)
    num_candidates = 0
    perimeter = build_perimeter(perimeter_size, perimeter_filled)

    half_h = height // 2
    half_w = width // 2
    max_dist_sq = float(half_h**2 + half_w**2)
    canvas_max_dist_sq = float(height**2 + width**2)

    minkowski_mask = build_minkowski_mask(minkowski_radius, minkowski_power)
    if max_neighbor_weirdness:
        max_neighbors = (1 * 2 + 1) ** 2
    else:
        max_neighbors = float(np.sum(minkowski_mask))

    density_map = np.zeros((height, width), dtype=np.int32)

    initial_seeds = np.zeros((seed_amount, 2), dtype=np.int32)

    for s in range(seed_amount):
        if seed_amount == 1:
            sy = half_h
            sx = half_w
        else:
            sy = rng.integers(0, height, size=1)[0]
            sx = rng.integers(0, width, size=1)[0]
        initial_seeds[s, 0] = sy
        initial_seeds[s, 1] = sx

        num_candidates = push_candidate(sy, sx, num_candidates, candidates, canvas)

    static_fitness_map = precompute_static_fitness(
        height,
        width,
        bias,
        bias_power,
        half_w,
        half_h,
        max_dist_sq,
        seed_amount,
        initial_seeds,
        canvas_max_dist_sq,
    )

    if capture_interval > 0:
        yield 0

    for i in range(2, iterations + 2):
        if num_candidates == 0:
            break

        random_candidate = 0
        ny = 0
        nx = 0

        if bias == 0:
            random_candidate = rng.integers(0, num_candidates)
            ny = candidates[random_candidate, 0]
            nx = candidates[random_candidate, 1]
        else:
            best_idx = -1
            best_fitness = -1.0
            tournament = build_tournament(rng, num_candidates, min(tournament_size, num_candidates))
            for idx in tournament:
                cy = candidates[idx, 0]
                cx = candidates[idx, 1]

                if bias == 5:
                    filled_neighbors = density_map[cy, cx]
                    density = filled_neighbors / max_neighbors
                    fitness = (1.0 - density) ** bias_power
                else:
                    fitness = static_fitness_map[cy, cx]

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_idx = idx

            random_candidate = best_idx
            ny = candidates[random_candidate, 0]
            nx = candidates[random_candidate, 1]

        num_candidates -= 1
        candidates[random_candidate, 0] = candidates[num_candidates, 0]
        candidates[random_candidate, 1] = candidates[num_candidates, 1]

        update_density_map(ny, nx, minkowski_radius, density_map, minkowski_mask)

        canvas[ny, nx] = i

        for p in range(perimeter.shape[0]):
            cy = ny + perimeter[p, 1]
            cx = nx + perimeter[p, 0]
            if 0 <= cy < height and 0 <= cx < width:
                if canvas[cy, cx] == 0:
                    num_candidates = push_candidate(cy, cx, num_candidates, candidates, canvas)

        if capture_interval > 0 and i % capture_interval == 0:
            yield 0

    for y in range(height):
        for x in range(width):
            if canvas[y, x] > 0:
                canvas[y, x] -= 1

    yield 1


@njit(cache=True)  # type: ignore
def push_candidate(
    y: int,
    x: int,
    num_candidates: int,
    candidates: NDArray[np.int32],
    canvas: NDArray[np.uint32],
) -> int:

    candidates[num_candidates, 0] = y
    candidates[num_candidates, 1] = x
    canvas[y, x] = 1
    return num_candidates + 1


@njit(cache=True)  # type: ignore
def build_perimeter(size: int, filled: bool) -> NDArray[np.int32]:
    if filled:
        num_points = 2 * size * (size + 1)
        start_perimeter = 1
    else:
        num_points = size * 4
        start_perimeter = size

    perimeter = np.zeros((num_points, 2), dtype=np.int32)
    pointer_perimeter = 0
    for p in range(start_perimeter, size + 1):
        for i in range(p):
            perimeter[pointer_perimeter, 0] = i
            perimeter[pointer_perimeter, 1] = p - i
            pointer_perimeter += 1
            perimeter[pointer_perimeter, 0] = p - i
            perimeter[pointer_perimeter, 1] = -i
            pointer_perimeter += 1
            perimeter[pointer_perimeter, 0] = -i
            perimeter[pointer_perimeter, 1] = -(p - i)
            pointer_perimeter += 1
            perimeter[pointer_perimeter, 0] = -(p - i)
            perimeter[pointer_perimeter, 1] = i
            pointer_perimeter += 1
    return perimeter


@njit(fastmath=True, cache=True)  # type: ignore
def update_density_map(
    cy: int,
    cx: int,
    radius: int,
    density_map: NDArray[np.int32],
    mask: NDArray[np.int32],
):
    height, width = density_map.shape
    mask_size = radius * 2 + 1
    my_start = max(0, radius - cy)
    my_end = min(mask_size, height + radius - cy)
    mx_start = max(0, radius - cx)
    mx_end = min(mask_size, width + radius - cx)
    canvas_y_base = cy - radius
    canvas_x_base = cx - radius
    for my in range(my_start, my_end):
        canvas_y = canvas_y_base + my
        for mx in range(mx_start, mx_end):
            density_map[canvas_y, canvas_x_base + mx] += mask[my, mx]


@njit(cache=True)  # type: ignore
def build_minkowski_mask(radius: int, power: float) -> NDArray[np.int32]:
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.int32)
    center = radius
    radius_f = float(radius)

    for y in range(size):
        for x in range(size):
            dy = float(abs(y - center))
            dx = float(abs(x - center))
            if (dx / radius_f) ** power + (dy / radius_f) ** power <= 1.0:
                kernel[y, x] = 1

    return kernel


@njit(fastmath=True, cache=True)  # type: ignore
def precompute_static_fitness(
    height: int,
    width: int,
    bias: int,
    bias_power: float,
    half_w: int,
    half_h: int,
    max_dist_sq: float,
    seed_amount: int,
    initial_seeds: NDArray[np.int32],
    canvas_max_dist_sq: float,
) -> NDArray[np.float32]:

    if bias == 0 or bias == 5:
        return np.empty((0, 0), dtype=np.float32)

    fitness_map = np.empty((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            if bias == 1:
                fitness_map[y, x] = (y / height) ** bias_power
            elif bias == 2:
                fitness_map[y, x] = (abs(x - half_w) / half_w) ** bias_power
            elif bias == 3:
                dist_sq = float((y - half_h) ** 2 + (x - half_w) ** 2)
                fitness_map[y, x] = (dist_sq / max_dist_sq) ** (bias_power / 2.0)
            elif bias == 4:
                total_dist_sq = 0.0
                for s in range(seed_amount):
                    dist_sq = float((y - initial_seeds[s, 0]) ** 2 + (x - initial_seeds[s, 1]) ** 2)
                    total_dist_sq += dist_sq
                fitness_map[y, x] = (total_dist_sq / (seed_amount * canvas_max_dist_sq)) ** bias_power

    return fitness_map


@njit(cache=True)  # type: ignore
def build_tournament(rng: np.random.Generator, pool_size: int, tournament_size: int) -> NDArray[np.int32]:
    tournament = np.empty(tournament_size, dtype=np.int32)

    for i in range(tournament_size):
        j = pool_size - tournament_size + i
        t = rng.integers(0, j + 1)
        is_duplicate = False
        for x in range(i):
            if tournament[x] == t:
                is_duplicate = True
                break
        tournament[i] = j if is_duplicate else t
    return tournament
