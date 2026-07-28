import concurrent.futures
from dataclasses import dataclass, fields
import json
from typing import Any, Callable
import argparse
from PIL import Image
from itertools import repeat
import os
from datetime import datetime
import sys
import subprocess

from algorithms.config_resolver import EngineConfig, RendererConfig, load_configs
import algorithms.eden as eden
import algorithms.game_of_life as game_of_life

""" ToDo's
    - Video metadata is sometimes not saved correctly, reason unknown.
    - Multicore for a single video generation?
    - come up with a flag for "blur stacking" in layer_images
    - reimplement --background_image functionality
    - move show_metadata to config_resolver?
    - decouple background color from apply_shader, layering before background color
    - switch to blur after layering
    - config_resolver.py and base_config should not be in algorithm directory
    - Why is background color in algorithm config class?
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
    parser.add_argument("-v", "--video", type=str, default=None, help="Path to a mp4 file to load the base configuration from.")
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


def show_metadata(filepath: str) -> None:
    meta_key = "MimikryConfig"

    if filepath.lower().endswith((".mp4", ".mkv", ".mov", ".webm")):
        result = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format_tags=description", "-of", "default=nw=1:nk=1", filepath], capture_output=True, text=True)
        print(result)
        print(json.dumps(json.loads(result.stdout.strip()), indent=4))
        return

    with Image.open(filepath) as img:
        metadata = img.info

    print(json.dumps(json.loads(metadata[meta_key]), indent=4))


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.show_metadata:
        if parsed_args.image:
            show_metadata(parsed_args.image)
        elif parsed_args.video:
            show_metadata(parsed_args.video)
    else:
        algo_key = parsed_args.algorithm
        target_config = ALGORITHM_REGISTRY[algo_key].config
        target_factory = ALGORITHM_REGISTRY[algo_key].factory
        engine_config, renderer_config, algorithm_configs = load_configs(parsed_args=parsed_args, target_config=target_config, target_factory=target_factory)
        main(engine_config=engine_config, renderer_config=renderer_config, algorithm_configs=algorithm_configs)
