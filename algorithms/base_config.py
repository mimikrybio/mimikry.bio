import numpy as np
from dataclasses import dataclass, asdict, replace
import json
from typing import Callable, Any, ClassVar, Type, Generator
import os
from PIL import Image
import renderer


@dataclass(kw_only=True)
class AlgorithmBaseConfig:
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
        if self.color_palette not in self.COLOR_PALETTES:
            valid_keys = list(self.COLOR_PALETTES.keys())
            raise ValueError(f"Fatal: Invalid palette '{self.color_palette}'. Valid options: {valid_keys}")
        if self.background_color not in self.BG_COLORS:
            valid_keys = list(self.BG_COLORS.keys())
            raise ValueError(f"Fatal: Invalid background '{self.background_color}'. Valid options: {valid_keys}")

    def get_palette_array(self) -> np.ndarray:
        return np.array(self.COLOR_PALETTES[getattr(self, "color_palette", "grayscale")], dtype=np.uint8)

    def get_background_array(self) -> np.ndarray:
        return np.array(self.BG_COLORS[getattr(self, "background_color", "black")], dtype=np.uint8)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class AlgorithmBaseFactory:
    _CONFIG_CLASS: ClassVar[Type[Any]]
    _RULES: ClassVar[dict[str, Callable[[np.random.Generator], Any]]]

    @classmethod
    def generate_random(cls, rng: np.random.Generator) -> Any:
        new_values = {key: rule(rng) for key, rule in cls._RULES.items()}
        return cls._CONFIG_CLASS(**new_values)

    @classmethod
    def unlock(cls, base_config: Any, rng: np.random.Generator, unlocked_keys: list[str]) -> Any:
        new_values = {key: cls._RULES[key](rng) for key in unlocked_keys}
        return replace(base_config, **new_values)


def run_algorithm(
    i: int,
    config: Any,
    to_video: bool,
    duration: float,
    fps: int,
    batch_directory: str,
    background_image: str | None,
    blur_radius: int,
    blur_sigma: float,
    engine_config: dict[str, Any],
    generator_factory: Callable[[np.random.Generator, np.ndarray, int], Generator[int, None, None]],
):
    task_rng = np.random.default_rng(config.simulation_seed)
    canvas = np.zeros((config.height, config.width), dtype=np.uint32)
    color_palette = config.get_palette_array()
    background_color = config.get_background_array()
    color_buffer = np.zeros((config.height, config.width, color_palette.shape[1]), dtype=np.uint8)

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
        process = renderer.initialize_video_stream(config.width * 1, config.height * 1, fps, output_filepath)

    algorithm_generator = generator_factory(task_rng, canvas, capture_interval)

    for status in algorithm_generator:
        if to_video and status == 0 and process:
            norm_c = renderer.normalize(
                canvas,
                config.norm_method,
                config.norm_power,
                config.sig_steepness,
                config.sig_midpoint,
            )
            renderer.apply_shader(norm_c, color_palette, background_color, color_buffer)
            blurred_buffer = renderer.apply_gaussian_blur(color_buffer, blur_radius, blur_sigma)
            scaled_buffer = renderer.apply_scaling(blurred_buffer, 1)
            out_buffer = renderer.layer_images(scaled_buffer, background_array) if background_array is not None else scaled_buffer
            process.stdin.write(out_buffer.tobytes())  # type: ignore

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
        blurred_buffer = renderer.apply_gaussian_blur(color_buffer, blur_radius, blur_sigma)
        scaled_buffer = renderer.apply_scaling(blurred_buffer, 1)
        out_buffer = renderer.layer_images(scaled_buffer, background_array) if background_array is not None else scaled_buffer

        unified_config = {"engine": engine_config, "algorithm": asdict(config)}
        output_filepath = os.path.join(batch_directory, f"{i:04d}.png")
        renderer.save_static_image(output_filepath, out_buffer, json.dumps(unified_config))
