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
import sys

from renderer import RendererConfig
import algorithms.eden as eden
import algorithms.game_of_life as game_of_life

""" ToDo's
    - Why is background color in algorithm config class?
    - Where should EngineConfig live?
    - reimplement unlocking parameters
    - Should RendererConfig really be repeated? Or randomize and 1 per AlgorithmConfig?
    - Independent scaling for height and width
    - How to handle different image dimension during layering?
    - Max_neighborhood weirdness causes tournament to spawn seed at 0,0
    - Interaction between fps and duration parameters is unintuitive. Refactor.
    - Add non-linear video time
    - Add video start, where prompt is typed
    - Add linger to end of video
    - Should color palettes be defined in renderer.py?
"""


@dataclass(kw_only=True)
class EngineConfig:
    algorithm: str
    config: str | None = None
    batch_size: int = 1
    master_seed: int | None = None
    image: str | None = None
    show_metadata: bool = False
    unlock: list[str] | None = None


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
    parser.add_argument("-j", "--json", type=str, default=None, help="Path to a JSON configuration file.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Number of images to generate per execution.")
    parser.add_argument("-m", "--master_seed", type=int, default=None, help="Master seed for batch determinism.")
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to a PNG file to load the base configuration from.")
    parser.add_argument("--show_metadata", action="store_true", default=None, help="Print the configuration metadata of the provided --image and exit.")
    parser.add_argument("-u", "--unlock", nargs="+", type=str, default=None, help="Parameters to unlock for mutation (e.g., -u bias).")

    def add_dataclass_args(dataclass_type: Any):
        for f in fields(dataclass_type):
            if f.type is bool:
                parser.add_argument(f"--{f.name}", action="store_true", default=None)
            else:
                parser.add_argument(f"--{f.name}", type=f.type if f.type in [int, float, str] else str, default=argparse.SUPPRESS)

    add_dataclass_args(RendererConfig)

    if algo_key:
        add_dataclass_args(ALGORITHM_REGISTRY[algo_key].config)

    return parser.parse_args(args)


def main(engine_config: EngineConfig, renderer_config: RendererConfig, algorithm_configs: list[Any]):
    target_runner = ALGORITHM_REGISTRY[engine_config.algorithm].runner

    tasks = range(engine_config.batch_size)
    timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
    batch_directory = f"batch_{timestamp}"
    os.makedirs(batch_directory, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            target_runner,
            tasks,
            algorithm_configs,
            repeat(renderer_config),
            repeat(batch_directory),
            repeat(engine_config),
        )
        list(results)


def extract_locks(parsed_args: argparse.Namespace, config_class: type) -> dict[str, Any]:
    locks: dict[str, Any] = {}

    for f in fields(config_class):
        raw_value = getattr(parsed_args, f.name, None)
        if raw_value is None:
            continue

        if f.type in (int, float, str):
            locks[f.name] = f.type(raw_value)
        elif f.type is bool:
            locks[f.name] = raw_value
        else:
            raise NotImplementedError(f"CLI parsing for field '{f.name}' of type {f.type} is not yet implemented.")

    return locks


def load_configs(parsed_args: argparse.Namespace, algo_key: str):
    target_config = ALGORITHM_REGISTRY[algo_key].config
    target_factory = ALGORITHM_REGISTRY[algo_key].factory

    engine_config_locks = extract_locks(parsed_args, EngineConfig)
    renderer_config_locks = extract_locks(parsed_args, RendererConfig)
    algorithm_config_locks = extract_locks(parsed_args, target_config)

    image_engine_config, image_renderer_config, image_algorithm_config = {}, {}, {}
    json_engine_config, json_renderer_config, json_algorithm_config = {}, {}, {}
    if parsed_args.image:
        with Image.open(parsed_args.image) as image:
            image_full_config = json.loads(image.info.get("MimikryConfig", "{}"))
            image_engine_config = image_full_config.get("engine", {})
            image_engine_config.update(engine_config_locks)
            image_renderer_config = image_full_config.get("renderer", {})
            image_renderer_config.update(renderer_config_locks)
            image_algorithm_config = image_full_config.get("algorithm", {})
            image_algorithm_config.update(algorithm_config_locks)
            algorithm_configs = [target_config(**image_algorithm_config) for _ in range(image_engine_config["batch_size"])]
            return EngineConfig(**image_engine_config), RendererConfig(**image_renderer_config), algorithm_configs

    elif parsed_args.json:
        with open(parsed_args.json) as file:
            json_full_config = json.load(file)
            json_engine_config = json_full_config.get("engine", {})
            json_engine_config.update(engine_config_locks)
            json_renderer_config = json_full_config.get("renderer", {})
            json_renderer_config.update(renderer_config_locks)
            json_algorithm_config = json_full_config.get("algorithm", {})
            json_algorithm_config.update(algorithm_config_locks)
            algorithm_configs = [target_config(**json_algorithm_config) for _ in range(json_engine_config["batch_size"])]
            return EngineConfig(**json_engine_config), RendererConfig(**json_renderer_config), algorithm_configs

    else:
        rng = np.random.default_rng(parsed_args.master_seed)
        algorithm_configs = [target_factory.generate_random(rng) for _ in range(parsed_args.batch_size)]
        algorithm_configs = [replace(config, **algorithm_config_locks) for config in algorithm_configs]
        return EngineConfig(algorithm=algo_key), RendererConfig(), algorithm_configs


def show_metadata(filepath: str) -> None:
    meta_key = "MimikryConfig"

    with Image.open(filepath) as img:
        metadata = img.info

    print(json.dumps(json.loads(metadata[meta_key]), indent=4))


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.show_metadata and parsed_args.image:
        show_metadata(parsed_args.image)
        exit(0)

    algo_key = parsed_args.algorithm
    engine_config, renderer_config, algorithm_configs = load_configs(parsed_args=parsed_args, algo_key=algo_key)
    main(engine_config=engine_config, renderer_config=renderer_config, algorithm_configs=algorithm_configs)
