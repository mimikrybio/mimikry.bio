import subprocess
import numpy as np
from numpy.typing import NDArray
from numba import njit  # type: ignore
from PIL import Image
from PIL.PngImagePlugin import PngInfo


@njit(cache=True)  # type: ignore
def normalize(
    canvas: NDArray[np.uint32],
    method: int,
    power: float,
    steepness: float,
    midpoint: float,
) -> NDArray[np.float32]:
    max_val = canvas.max()
    if max_val == 0:
        return canvas.astype(np.float32)
    if method == 2:
        log_max = np.log1p(max_val)
        normalized_canvas = np.log1p(canvas) / log_max
    else:
        normalized_canvas = canvas / max_val
        if method == 1:
            normalized_canvas = normalized_canvas**power
        elif method == 3:
            sig = 1.0 / (1.0 + np.exp(-steepness * (normalized_canvas - midpoint)))
            sig_min = sig.min()
            sig_max = sig.max()
            if sig_min == sig_max:
                return normalized_canvas.astype(np.float32)
            normalized_canvas = (sig - sig_min) / (sig_max - sig_min)
    return normalized_canvas.astype(np.float32)


@njit(fastmath=True, cache=True)  # type: ignore
def apply_shader(
    norm_c: NDArray[np.float32], palette: NDArray[np.uint8], out: NDArray[np.uint8]
) -> None:
    h, w = norm_c.shape
    colors, chans = palette.shape

    if colors == 0:
        return

    if colors == 1:
        color = palette[0]
        for y in range(h):  # Reverted to standard range
            for x in range(w):
                if norm_c[y, x] > 0.0:
                    for c in range(chans):
                        out[y, x, c] = color[c]
                else:
                    for c in range(chans):
                        out[y, x, c] = 0
        return

    pal = palette.astype(np.float32)
    segs = colors - 1

    for y in range(h):  # Reverted to standard range
        for x in range(w):
            v = norm_c[y, x]

            if v > 0.0:
                scaled = v * segs
                idx = min(int(scaled), segs - 1)
                frac = scaled - idx

                for c in range(chans):
                    out[y, x, c] = int(
                        pal[idx, c] + (pal[idx + 1, c] - pal[idx, c]) * frac
                    )
            # else:
            #     for c in range(chans):
            #         out[y, x, c] = palette[0, c]


def save_static_image(filepath: str, color_buffer: NDArray[np.uint8], config_json: str):
    meta_data = PngInfo()
    meta_data.add_text("EdenConfig", config_json)
    Image.fromarray(color_buffer, mode="RGBA").save(filepath, pnginfo=meta_data)


def initialize_video_stream(
    width: int, height: int, fps: int, output_filepath: str
) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf",
        "18",
        output_filepath,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
