import numpy as np
import concurrent.futures
from dataclasses import dataclass, fields, replace, asdict
import json
from typing import Any, Callable
import argparse
from PIL import Image
from itertools import repeat
import os
from datetime import datetime
import sys

from renderer import RendererConfig
import algorithms.eden as eden
import algorithms.game_of_life as game_of_life

""" ToDo's
    - Improve how RendererConfig is constructed in __name__ block
    - Implement reduced argparser from Gemini?"
    - Should RendererConfig really be repeated? Or randomize and 1 per AlgorithmConfig?
    - Bring metadata structure in line with configs used
    - Engine config for top-level arguments?
    - remove initial parser and make algorithm just another argument?
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
    target_args = args if args is not None else sys.argv[1:]
    algo_key = next((arg for arg in target_args if arg in ALGORITHM_REGISTRY), None)

    parser = argparse.ArgumentParser(description="Mimikry Generative Art Engine")
    parser.add_argument("algorithm", type=str, choices=list(ALGORITHM_REGISTRY.keys()), nargs="?")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a JSON configuration file.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Number of images to generate per execution.")
    parser.add_argument("-m", "--master_seed", type=int, default=None, help="Master seed for batch determinism.")
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to a PNG file to load the base configuration from.")
    parser.add_argument("--show_metadata", action="store_true", help="Print the configuration metadata of the provided --image and exit.")
    parser.add_argument("-u", "--unlock", nargs="+", type=str, default=None, help="Parameters to unlock for mutation (e.g., -u bias).")

    def add_dataclass_args(dataclass_type: Any):
        for f in fields(dataclass_type):
            if f.type is bool:
                parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
            else:
                parser.add_argument(f"--{f.name}", type=f.type if f.type in [int, float, str] else str, default=argparse.SUPPRESS)

    add_dataclass_args(RendererConfig)

    if algo_key:
        add_dataclass_args(ALGORITHM_REGISTRY[algo_key].config)

    return parser.parse_args(args)


def main(
    algo_key: str,
    batch_size: int,
    renderer_config: RendererConfig,
    master_seed: int | None = None,
    image_filepath: str | None = None,
    unlocked_parameters: list[str] | None = None,
    locked_parameters: dict[str, Any] | None = None,
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

    engine_config: dict[str, Any] = {"algorithm": algo_key, "master_seed": master_seed, "batch_size": batch_size, **asdict(renderer_config)}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            target_runner,
            tasks,
            configs,
            repeat(renderer_config),
            repeat(batch_directory),
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

    args_dict = vars(parsed_args)
    renderer_kwargs = {f.name: args_dict[f.name] for f in fields(RendererConfig) if f.name in args_dict}
    renderer_config = RendererConfig(**renderer_kwargs)

    main(
        algo_key=algo_key,
        batch_size=parsed_args.batch_size,
        renderer_config=renderer_config,
        master_seed=parsed_args.master_seed,
        image_filepath=parsed_args.image,
        unlocked_parameters=parsed_args.unlock,
        locked_parameters=locked_parameters,
    )
