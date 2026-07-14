import numpy as np
from numpy.typing import NDArray
from numba import njit  # type: ignore
import concurrent.futures
from numpy.random import Generator
from dataclasses import dataclass, asdict, fields, replace
import json
from typing import Callable, Any, ClassVar
import argparse
from PIL import Image
import renderer
from itertools import repeat
import os
from datetime import datetime

""" ToDo's
    - Add 'layer_images' functionatity
    - Add game of life algorithm
    - Add smoothing function, apply to background image
    - Add non-linear video time
    - Add video start, where prompt is typed
    - Add linger to end of video
    - Add option to provide json file for settings
    - Remove magic number for smoothing function and integrate smoothing with argparse
    - Gaussian blur instead of box?
"""


@dataclass
class EdenConfig:
    height: int
    width: int
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
    norm_method: int
    norm_power: float
    sig_steepness: float
    sig_midpoint: float
    color_palette: str
    background_color: str

    COLOR_PALETTES: ClassVar[dict[str, list[list[int]]]] = {
        "grayscale": [[0, 0, 0, 255], [255, 255, 255, 255]],
        "fire": [[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255]],
        "ocean": [[0, 0, 50, 255], [0, 150, 255, 255], [200, 255, 255, 255]],
        "neon": [[20, 0, 40, 255], [255, 0, 255, 255], [0, 255, 255, 255]],
        "circuit": [[10, 30, 15, 255], [0, 160, 70, 255], [215, 175, 55, 255]],
        "tree": [[92, 64, 51, 255], [52, 199, 89, 255]],
    }

    BG_COLORS: ClassVar[dict[str, list[int]]] = {
        "transparent_black": [0, 0, 0, 0],
        "black": [0, 0, 0, 255],
        "white": [255, 255, 255, 255],
        "transparent_white": [255, 255, 255, 0],
    }

    def __post_init__(self):
        if self.minkowski_radius <= 0:
            raise ValueError(
                f"Fatal: minkowski_radius must be > 0. Received {self.minkowski_radius}"
            )
        if self.iterations <= 0:
            raise ValueError(
                f"Fatal: iterations must be > 0. Received {self.iterations}"
            )
        if self.tournament_size < 1:
            raise ValueError(
                f"Tournament size must be > 0. Received {self.tournament_size}"
            )
        if self.color_palette not in self.COLOR_PALETTES:
            valid_keys = list(self.COLOR_PALETTES.keys())
            raise ValueError(
                f"Fatal: Invalid palette '{self.color_palette}'. Valid options: {valid_keys}"
            )
        if self.background_color not in self.BG_COLORS:
            valid_keys = list(self.BG_COLORS.keys())
            raise ValueError(
                f"Fatal: Invalid background '{self.background_color}'. Valid options: {valid_keys}"
            )

    def get_palette_array(self) -> np.ndarray:
        """Returns the Numba-ready NumPy array for this configuration's palette."""
        return np.array(self.COLOR_PALETTES[self.color_palette], dtype=np.uint8)

    def get_background_array(self) -> np.ndarray:
        return np.array(self.BG_COLORS[self.background_color], dtype=np.uint8)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class EdenFactory:
    _RULES: ClassVar[dict[str, Callable[[Generator], Any]]] = {
        "height": lambda rng: int(rng.integers(250, 1001)) * 2,
        "width": lambda rng: int(rng.integers(250, 1001)) * 2,
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
        "norm_method": lambda rng: int(rng.integers(0, 4)),
        "norm_power": lambda rng: float(rng.uniform(0.5, 2.5)),
        "sig_steepness": lambda rng: float(rng.uniform(1.0, 10.0)),
        "sig_midpoint": lambda rng: float(rng.uniform(0.1, 0.9)),
        "color_palette": lambda rng: str(
            rng.choice(list(EdenConfig.COLOR_PALETTES.keys()))
        ),
        "background_color": lambda rng: str(
            rng.choice(list(EdenConfig.BG_COLORS.keys()))
        ),
    }

    @classmethod
    def generate_random(cls, rng: Generator) -> EdenConfig:
        new_values = {key: rule(rng) for key, rule in cls._RULES.items()}
        return EdenConfig(**new_values)

    @classmethod
    def unlock(
        cls, base_config: EdenConfig, rng: Generator, unlocked_keys: list[str]
    ) -> EdenConfig:
        new_values = {key: cls._RULES[key](rng) for key in unlocked_keys}
        return replace(base_config, **new_values)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mimikry Generative Art Engine")

    parser.add_argument(
        "algorithm",
        type=str,
        choices=["eden"],
        help="The specific generative algorithm to run.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to generate per execution.",
    )

    parser.add_argument(
        "-m",
        "--master_seed",
        type=int,
        default=None,
        help="Master seed for batch determinism. If omitted, pulls from OS entropy.",
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=None,
        help="Path to a PNG file to load the base EdenConfig from.",
    )

    parser.add_argument(
        "--show_metadata",
        action="store_true",
        help="Print the EdenConfig metadata of the provided --image and exit.",
    )

    parser.add_argument(
        "-u",
        "--unlock",
        nargs="+",
        type=str,
        default=None,
        help="Parameters to unlock for mutation (e.g., -u bias minkowski_radius).",
    )

    parser.add_argument(
        "-bg",
        "--background_image",
        type=str,
        default=None,
        help="Path to a PNG file to use as the background layer.",
    )

    override_group = parser.add_argument_group("Parameter Overrides (Locks)")
    for f in fields(EdenConfig):
        if f.name == "background_color":
            override_group.add_argument(
                f"--{f.name}",
                type=str,
                choices=list(EdenConfig.BG_COLORS.keys()),
                default=None,
                help=f"Lock the {f.name} parameter.",
            )
        elif f.name == "color_palette":
            override_group.add_argument(
                f"--{f.name}",
                type=str,
                choices=list(EdenConfig.COLOR_PALETTES.keys()),
                default=None,
                help=f"Lock the {f.name} parameter.",
            )
        else:
            override_group.add_argument(
                f"--{f.name}",
                type=str,
                default=None,
                help=f"Lock the {f.name} parameter.",
            )

    video_group = parser.add_argument_group("Video Rendering Options")
    video_group.add_argument(
        "--to_video",
        action="store_true",
        help="Trigger video generation instead of static image output.",
    )
    video_group.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Target video duration in seconds.",
    )
    video_group.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for the output video.",
    )

    return parser.parse_args(args)


def main(
    batch_size: int,
    master_seed: int | None = None,
    image_filepath: str | None = None,
    background_image: str | None = None,
    unlocked_parameters: list[str] | None = None,
    locked_parameters: dict[str, Any] | None = None,
    to_video: bool = False,
    duration: float = 30.0,
    fps: int = 60,
):
    validate_execution(image_filepath, unlocked_parameters, batch_size)
    tasks = range(batch_size)
    locked_params = locked_parameters or {}

    timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
    batch_directory = f"batch_{timestamp}"
    os.makedirs(batch_directory, exist_ok=True)

    master_rng = np.random.default_rng(master_seed)

    if image_filepath:
        base_config = load_image_config(image_filepath)

        if unlocked_parameters:
            configs = [
                EdenFactory.unlock(base_config, master_rng, unlocked_parameters)
                for _ in range(batch_size)
            ]
        else:
            configs = [base_config for _ in range(batch_size)]

    else:
        configs = [EdenFactory.generate_random(master_rng) for _ in range(batch_size)]

    if locked_params:
        configs = [replace(config, **locked_params) for config in configs]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            run_eden,
            tasks,
            configs,
            repeat(to_video),
            repeat(duration),
            repeat(fps),
            repeat(batch_directory),
            repeat(background_image),
        )
        list(results)


def run_eden(
    i: int,
    config: EdenConfig,
    to_video: bool,
    duration: float,
    fps: int,
    batch_directory: str,
    background_image: str | None,
):
    task_rng = np.random.default_rng(config.simulation_seed)
    canvas = np.zeros((config.height, config.width), dtype=np.uint32)
    color_palette = config.get_palette_array()
    background_color = config.get_background_array()
    color_buffer = np.zeros(
        (config.height, config.width, color_palette.shape[1]), dtype=np.uint8
    )

    background_array = None
    if background_image:
        with Image.open(background_image) as img:
            background_array = np.array(img.convert("RGBA"), dtype=np.uint8)

    capture_interval = 0
    process = None

    if to_video:
        total_frames = int(duration * fps)
        capture_interval = max(1, config.iterations // total_frames)
        output_filepath = os.path.join(batch_directory, f"{i:04d}.mp4")
        process = renderer.initialize_video_stream(
            config.width, config.height, fps, output_filepath
        )

    eden_generator = eden(
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

    for status in eden_generator:
        if to_video and status in (0, 1) and process:
            norm_c = renderer.normalize(
                canvas,
                config.norm_method,
                config.norm_power,
                config.sig_steepness,
                config.sig_midpoint,
            )
            renderer.apply_shader(norm_c, color_palette, background_color, color_buffer)
            blurred_buffer = renderer.apply_box_blur(color_buffer, radius=0)
            out_buffer = (
                renderer.layer_images(blurred_buffer, background_array)
                if background_array is not None
                else blurred_buffer
            )
            process.stdin.write(color_buffer.tobytes())  # type: ignore

    if to_video and process:
        process.stdin.close()  # type: ignore
        process.wait()
    else:
        norm_c = renderer.normalize(
            canvas,
            config.norm_method,
            config.norm_power,
            config.sig_steepness,
            config.sig_midpoint,
        )

        renderer.apply_shader(norm_c, color_palette, background_color, color_buffer)
        blurred_buffer = renderer.apply_box_blur(color_buffer, radius=0)
        out_buffer = (
            renderer.layer_images(blurred_buffer, background_array)
            if background_array is not None
            else blurred_buffer
        )
        output_filepath = os.path.join(batch_directory, f"{i:04d}.png")
        renderer.save_static_image(output_filepath, out_buffer, config.to_json())


@njit(cache=True)  # type: ignore
def eden(
    rng: Generator,
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
            tournament = build_tournament(
                rng, num_candidates, min(tournament_size, num_candidates)
            )
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
                    num_candidates = push_candidate(
                        cy, cx, num_candidates, candidates, canvas
                    )

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
                    dist_sq = float(
                        (y - initial_seeds[s, 0]) ** 2 + (x - initial_seeds[s, 1]) ** 2
                    )
                    total_dist_sq += dist_sq
                fitness_map[y, x] = (
                    total_dist_sq / (seed_amount * canvas_max_dist_sq)
                ) ** bias_power

    return fitness_map


def extract_locks(parsed_args: argparse.Namespace) -> dict[str, Any]:
    locks: dict[str, Any] = {}
    for f in fields(EdenConfig):
        raw_value = getattr(parsed_args, f.name, None)
        if raw_value is None:
            continue

        if f.type is int:
            locks[f.name] = int(raw_value)
        elif f.type is float:
            locks[f.name] = float(raw_value)
        elif f.type is str:
            locks[f.name] = str(raw_value)
        elif f.type is bool:
            val_lower = raw_value.lower()
            if val_lower == "true":
                locks[f.name] = True
            elif val_lower == "false":
                locks[f.name] = False
            else:
                raise ValueError(
                    f"Invalid boolean value for --{f.name}: '{raw_value}'. "
                    "Expected 'True' or 'False'."
                )
        else:
            raise NotImplementedError(
                f"CLI parsing for field '{f.name}' of type {f.type} is not yet implemented."
            )

    return locks


def load_image_config(filepath: str) -> EdenConfig:
    with Image.open(filepath) as img:
        metadata = img.info

    if "EdenConfig" not in metadata:
        raise ValueError(f"Missing 'EdenConfig' chunk in PNG metadata for {filepath}")

    config_dict = json.loads(metadata["EdenConfig"])

    return EdenConfig(**config_dict)


def show_metadata(filepath: str) -> None:
    config = load_image_config(filepath)
    print(json.dumps(asdict(config), indent=4))


@njit(cache=True)  # type: ignore
def build_tournament(
    rng: Generator, pool_size: int, tournament_size: int
) -> NDArray[np.int32]:
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


def validate_execution(
    image_filepath: str | None, unlocked_parameters: list[str] | None, batch_size: int
) -> None:
    if image_filepath and not unlocked_parameters and batch_size > 1:
        raise ValueError(
            "Cannot generate a batch > 1 from a parent image without unlocking parameters to prevent redundant processing."
        )


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.algorithm == "eden":
        if parsed_args.show_metadata and parsed_args.image:
            show_metadata(parsed_args.image)
            exit(0)
        locked_parameters = extract_locks(parsed_args)
        main(
            batch_size=parsed_args.batch_size,
            master_seed=parsed_args.master_seed,
            image_filepath=parsed_args.image,
            background_image=parsed_args.background_image,
            unlocked_parameters=parsed_args.unlock,
            locked_parameters=locked_parameters,
            to_video=parsed_args.to_video,
            duration=parsed_args.duration,
            fps=parsed_args.fps,
        )
