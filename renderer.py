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
    norm_c: NDArray[np.float32],
    palette: NDArray[np.uint8],
    background_color: NDArray[np.uint8],
    out: NDArray[np.uint8],
) -> None:
    h, w = norm_c.shape
    colors, chans = palette.shape

    if colors == 0:
        return

    if colors == 1:
        color = palette[0]
        for y in range(h):
            for x in range(w):
                v = norm_c[y, x]
                if v > 0.0:
                    for c in range(chans):
                        out[y, x, c] = color[c]
                else:
                    for c in range(chans):
                        out[y, x, c] = background_color[c]
        return

    pal = palette.astype(np.float32)
    segs = colors - 1

    for y in range(h):
        for x in range(w):
            v = norm_c[y, x]

            if v > 0.0:
                scaled = v * segs
                idx = min(int(scaled), segs - 1)
                frac = scaled - idx

                for c in range(chans):
                    out[y, x, c] = int(pal[idx, c] + (pal[idx + 1, c] - pal[idx, c]) * frac)
            else:
                for c in range(chans):
                    out[y, x, c] = background_color[c]


@njit(fastmath=True, cache=True)  # type: ignore
def apply_gaussian_blur(img: NDArray[np.uint8], radius: int, sigma: float = 0.0) -> NDArray[np.uint8]:
    h, w, c = img.shape
    out = np.zeros_like(img)

    if radius <= 0:
        return img.copy()

    if sigma <= 0.0:
        sigma = max(radius / 2.0, 1.0)

    k_size = 2 * radius + 1
    kernel = np.zeros(k_size, dtype=np.float32)
    k_sum = 0.0

    for i in range(-radius, radius + 1):
        val = np.exp(-(i**2) / (2.0 * sigma**2))
        kernel[i + radius] = val
        k_sum += val

    for i in range(k_size):
        kernel[i] /= k_sum

    temp = np.zeros((h, w, c), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                val = 0.0
                for dx in range(-radius, radius + 1):
                    nx = x + dx
                    if nx < 0:
                        nx = 0
                    elif nx >= w:
                        nx = w - 1

                    val += img[y, nx, ch] * kernel[dx + radius]
                temp[y, x, ch] = val

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                val = 0.0
                for dy in range(-radius, radius + 1):
                    ny = y + dy
                    if ny < 0:
                        ny = 0
                    elif ny >= h:
                        ny = h - 1

                    val += temp[ny, x, ch] * kernel[dy + radius]
                out[y, x, ch] = np.uint8(val)

    return out


@njit(fastmath=True, cache=True)  # type: ignore
def apply_scaling(img: NDArray[np.uint8], scaling_factor: float) -> NDArray[np.uint8]:
    h, w, c = img.shape
    new_h = int(h * scaling_factor)
    new_w = int(w * scaling_factor)

    out = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for y in range(new_h):
        src_y = int(y / scaling_factor)
        if src_y >= h:
            src_y = h - 1

        for x in range(new_w):
            src_x = int(x / scaling_factor)
            if src_x >= w:
                src_x = w - 1

            for ch in range(c):
                out[y, x, ch] = img[src_y, src_x, ch]

    return out


@njit(fastmath=True, cache=True)  # type: ignore
def layer_images(fg: NDArray[np.uint8], bg: NDArray[np.uint8]) -> NDArray[np.uint8]:
    h, w, c = fg.shape
    out = np.zeros_like(fg)

    for y in range(h):
        for x in range(w):
            if fg[y, x, 3] != 0:
                if bg[y, x, 3] != 0:
                    for ch in range(c):
                        val = int(fg[y, x, ch]) + int(bg[y, x, ch])
                        if ch == 3:
                            out[y, x, ch] = 255
                        else:
                            out[y, x, ch] = val // 2
                else:
                    out[y, x] = fg[y, x]

    return out


def save_static_image(filepath: str, color_buffer: NDArray[np.uint8], unified_json: str):
    meta_data = PngInfo()
    meta_data.add_text("MimikryConfig", unified_json)
    Image.fromarray(color_buffer, mode="RGBA").save(filepath, pnginfo=meta_data)


def initialize_video_stream(width: int, height: int, fps: int, output_filepath: str) -> subprocess.Popen[bytes]:
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
