import numpy as np
import concurrent.futures
from dataclasses import dataclass, fields, replace
import json
from typing import Any, Callable
import argparse
from PIL import Image
from itertools import repeat
import os
from datetime import datetime

import algorithms.eden as eden
import algorithms.game_of_life as game_of_life

""" ToDo's
    - Height and width parameters should describe final image, scaling_factor as divisor
    - Independent scaling for height and width
    - Why is fps parameter not transfered from --image flag?
    - How to handle different image dimension during layering?
    - Max_neighborhood weirdness causes tournament to spawn seed at 0,0
    - Interaction between fps and duration parameters is unintuitive. Refactor.
    - Add non-linear video time
    - Add video start, where prompt is typed
    - Add linger to end of video
    - Should color palettes be defined in renderer.py?
"""


@dataclass
class AlgorithmEntry:
    config: type[Any]
    factory: type[Any]
    runner: Callable[..., Any]


ALGORITHM_REGISTRY: dict[str, AlgorithmEntry] = {
    "eden": AlgorithmEntry(
        config=eden.EdenConfig,
        factory=eden.EdenFactory,
        runner=eden.run_eden,
    ),
    "game_of_life": AlgorithmEntry(
        config=game_of_life.GameOfLifeConfig,
        factory=game_of_life.GameOfLifeFactory,
        runner=game_of_life.run_game_of_life,
    ),
}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument(
        "algorithm",
        type=str,
        choices=list(ALGORITHM_REGISTRY.keys()),
        nargs="?",
        default=None,
    )
    initial_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file.",
    )
    known_args, _ = initial_parser.parse_known_args(args)

    config_defaults: dict[str, Any] = {}
    if known_args.config:
        with open(known_args.config, "r") as f:
            raw_config = json.load(f)
            if "engine" in raw_config:
                config_defaults.update(raw_config["engine"])
            if "algorithm" in raw_config:
                config_defaults.update(raw_config["algorithm"])

    algo_key = str(known_args.algorithm or config_defaults.get("algorithm"))

    target_config = ALGORITHM_REGISTRY[algo_key].config

    parser = argparse.ArgumentParser(description="Mimikry Generative Art Engine")

    parser.add_argument(
        "algorithm",
        type=str,
        choices=list(ALGORITHM_REGISTRY.keys()),
        nargs="?",
        help="The specific generative algorithm to run.",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file.",
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
        help="Path to a PNG file to load the base configuration from.",
    )

    parser.add_argument(
        "--show_metadata",
        action="store_true",
        help="Print the configuration metadata of the provided --image and exit.",
    )

    parser.add_argument(
        "-u",
        "--unlock",
        nargs="+",
        type=str,
        default=None,
        help="Parameters to unlock for mutation (e.g., -u bias).",
    )

    parser.add_argument(
        "-bg",
        "--background_image",
        type=str,
        default=None,
        help="Path to a PNG file to use as the background layer.",
    )

    render_group = parser.add_argument_group("Rendering Options")
    render_group.add_argument(
        "--blur_radius",
        type=int,
        default=0,
        help="Radius for the Gaussian blur. A value of 0 disables the blur.",
    )
    render_group.add_argument(
        "--blur_sigma",
        type=float,
        default=0.0,
        help="Sigma for the Gaussian blur. A value of 0.0 auto-calculates the sigma based on the radius.",
    )

    render_group.add_argument(
        "--scaling_factor",
        type=int,
        default=1,
        help="Target scaling factor.",
    )

    override_group = parser.add_argument_group(f"{algo_key.capitalize()} Parameter Overrides (Locks)")

    for f in fields(target_config):
        if hasattr(target_config, "BG_COLORS") and f.name == "background_color":
            override_group.add_argument(
                f"--{f.name}",
                type=str,
                choices=list(target_config.BG_COLORS.keys()),
                default=None,
                help=f"Lock the {f.name} parameter.",
            )
        elif hasattr(target_config, "COLOR_PALETTES") and f.name == "color_palette":
            override_group.add_argument(
                f"--{f.name}",
                type=str,
                choices=list(target_config.COLOR_PALETTES.keys()),
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

    parser.set_defaults(**config_defaults)

    return parser.parse_args(args)


def main(
    algo_key: str,
    batch_size: int,
    master_seed: int | None = None,
    image_filepath: str | None = None,
    background_image: str | None = None,
    unlocked_parameters: list[str] | None = None,
    locked_parameters: dict[str, Any] | None = None,
    to_video: bool = False,
    duration: float = 30.0,
    fps: int = 60,
    blur_radius: int = 0,
    blur_sigma: float = 0.0,
    scaling_factor: int = 1,
):

    target_factory = ALGORITHM_REGISTRY[algo_key].factory
    target_runner = ALGORITHM_REGISTRY[algo_key].runner

    validate_execution(image_filepath, unlocked_parameters, batch_size)
    tasks = range(batch_size)
    locked_params = locked_parameters or {}

    timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
    batch_directory = f"batch_{timestamp}"
    os.makedirs(batch_directory, exist_ok=True)

    master_rng = np.random.default_rng(master_seed)

    if image_filepath:
        base_config = load_image_config(image_filepath, algo_key)
        if unlocked_parameters:
            configs = [target_factory.unlock(base_config, master_rng, unlocked_parameters) for _ in range(batch_size)]
        else:
            configs = [base_config for _ in range(batch_size)]
    else:
        configs = [target_factory.generate_random(master_rng) for _ in range(batch_size)]

    if locked_params:
        configs = [replace(config, **locked_params) for config in configs]

    engine_config: dict[str, Any] = {
        "algorithm": algo_key,
        "master_seed": master_seed,
        "batch_size": batch_size,
        "background_image": background_image,
        "to_video": to_video,
        "duration": duration,
        "fps": fps,
        "blur_radius": blur_radius,
        "blur_sigma": blur_sigma,
        "scaling_factor": scaling_factor,
    }

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            target_runner,
            tasks,
            configs,
            repeat(to_video),
            repeat(duration),
            repeat(fps),
            repeat(batch_directory),
            repeat(background_image),
            repeat(blur_radius),
            repeat(blur_sigma),
            repeat(scaling_factor),
            repeat(engine_config),
        )
        list(results)


def extract_locks(parsed_args: argparse.Namespace, algo_key: str) -> dict[str, Any]:
    target_config = ALGORITHM_REGISTRY[algo_key].config
    locks: dict[str, Any] = {}
    for f in fields(target_config):
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
            if isinstance(raw_value, bool):
                locks[f.name] = raw_value
            else:
                val_lower = str(raw_value).lower()
                if val_lower == "true":
                    locks[f.name] = True
                elif val_lower == "false":
                    locks[f.name] = False
                else:
                    raise ValueError(f"Invalid boolean value for --{f.name}: '{raw_value}'. " "Expected 'True' or 'False'.")
        else:
            raise NotImplementedError(f"CLI parsing for field '{f.name}' of type {f.type} is not yet implemented.")

    return locks


def load_image_config(filepath: str, algo_key: str) -> Any:
    target_config = ALGORITHM_REGISTRY[algo_key].config
    meta_key = "MimikryConfig"

    with Image.open(filepath) as img:
        metadata = img.info

    full_config = json.loads(metadata[meta_key])

    algo_config = full_config["algorithm"]
    return target_config(**algo_config)


def show_metadata(filepath: str) -> None:
    meta_key = "MimikryConfig"

    with Image.open(filepath) as img:
        metadata = img.info

    print(json.dumps(json.loads(metadata[meta_key]), indent=4))


def validate_execution(image_filepath: str | None, unlocked_parameters: list[str] | None, batch_size: int) -> None:
    if image_filepath and not unlocked_parameters and batch_size > 1:
        raise ValueError("Cannot generate a batch > 1 from a parent image without unlocking parameters to prevent redundant processing.")


if __name__ == "__main__":
    parsed_args = parse_args()
    algo_key = parsed_args.algorithm

    if parsed_args.show_metadata and parsed_args.image:
        show_metadata(parsed_args.image)
        exit(0)

    locked_parameters = extract_locks(parsed_args, algo_key)
    main(
        algo_key=algo_key,
        batch_size=parsed_args.batch_size,
        master_seed=parsed_args.master_seed,
        image_filepath=parsed_args.image,
        background_image=parsed_args.background_image,
        unlocked_parameters=parsed_args.unlock,
        locked_parameters=locked_parameters,
        to_video=parsed_args.to_video,
        duration=parsed_args.duration,
        fps=parsed_args.fps,
        blur_radius=parsed_args.blur_radius,
        blur_sigma=parsed_args.blur_sigma,
        scaling_factor=parsed_args.scaling_factor,
    )
