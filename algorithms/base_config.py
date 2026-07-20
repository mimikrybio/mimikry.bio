import numpy as np
from dataclasses import asdict
import json
from typing import Callable, Any, Generator
import os
from PIL import Image
import renderer
import math

from algorithms.config_resolver import EngineConfig, RendererConfig


def run_algorithm(
    i: int,
    algorithm_config: Any,
    renderer_config: RendererConfig,
    batch_directory: str,
    engine_config: EngineConfig,
    generator_factory: Callable[[np.random.Generator, np.ndarray, int], Generator[int, None, None]],
):
    unified_config = {"engine": asdict(engine_config), "renderer": asdict(renderer_config), "algorithm": asdict(algorithm_config)}

    nearest_common_divisor = get_nearest_common_divisor(algorithm_config.height, algorithm_config.width, renderer_config.scaling_factor)
    scaled_height = int(algorithm_config.height / nearest_common_divisor)
    scaled_width = int(algorithm_config.width / nearest_common_divisor)
    task_rng = np.random.default_rng(algorithm_config.simulation_seed)
    canvas = np.zeros((scaled_height, scaled_width), dtype=np.uint32)
    color_palette = algorithm_config.get_palette_array()
    background_color = algorithm_config.get_background_array()
    color_buffer = np.zeros((scaled_height, scaled_width, color_palette.shape[1]), dtype=np.uint8)

    background_array = None
    if renderer_config.background_image:
        with Image.open(renderer_config.background_image) as img:
            background_array = np.array(img.convert("RGBA"), dtype=np.uint8)

    capture_interval = 0
    process = None

    if renderer_config.to_video:
        total_frames = int(renderer_config.duration * renderer_config.fps)
        capture_interval = max(1, algorithm_config.iterations // total_frames)
        output_filepath = os.path.join(batch_directory, f"{i:04d}.mp4")
        process = renderer.initialize_video_stream(unified_config, output_filepath)

    algorithm_generator = generator_factory(task_rng, canvas, capture_interval)

    for status in algorithm_generator:
        if renderer_config.to_video and status == 0 and process:
            norm_c = renderer.normalize(
                canvas,
                renderer_config.norm_method,
                renderer_config.norm_power,
                renderer_config.sig_steepness,
                renderer_config.sig_midpoint,
            )
            renderer.apply_shader(norm_c, color_palette, background_color, color_buffer)
            blurred_buffer = renderer.apply_gaussian_blur(color_buffer, renderer_config.blur_radius, renderer_config.blur_sigma)
            scaled_buffer = renderer.apply_scaling(blurred_buffer, nearest_common_divisor)
            out_buffer = renderer.layer_images(scaled_buffer, background_array) if background_array is not None else scaled_buffer
            process.stdin.write(out_buffer.tobytes())  # type: ignore

    if renderer_config.to_video and process:
        process.stdin.close()  # type: ignore
        process.wait()
    else:
        norm_c = renderer.normalize(
            canvas,
            renderer_config.norm_method,
            renderer_config.norm_power,
            renderer_config.sig_steepness,
            renderer_config.sig_midpoint,
        )

        renderer.apply_shader(norm_c, color_palette, background_color, color_buffer)
        blurred_buffer = renderer.apply_gaussian_blur(color_buffer, renderer_config.blur_radius, renderer_config.blur_sigma)
        scaled_buffer = renderer.apply_scaling(blurred_buffer, nearest_common_divisor)
        out_buffer = renderer.layer_images(scaled_buffer, background_array) if background_array is not None else scaled_buffer

        output_filepath = os.path.join(batch_directory, f"{i:04d}.png")
        renderer.save_static_image(output_filepath, out_buffer, json.dumps(unified_config))


def get_nearest_common_divisor(height: int, width: int, suggested_scale: int) -> int:
    gcd_val = math.gcd(height, width)

    divisors: list[int] = []
    for i in range(1, int(math.isqrt(gcd_val)) + 1):
        if gcd_val % i == 0:
            divisors.append(i)
            if i != gcd_val // i:
                divisors.append(gcd_val // i)

    nearest_divisor = min(divisors, key=lambda d: abs(d - suggested_scale))

    return nearest_divisor
