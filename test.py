from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from pyray import (
    init_window, set_window_state, set_target_fps, window_should_close,
    get_screen_width, get_screen_height, begin_drawing, clear_background,
    end_drawing, draw_text, close_window,
    BLACK, Color, ConfigFlags
)
from wave_eqn2d import WaveEqn2D

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
TITLE = "Matrix Wave Digits"

GRID_SPACING = 10
FONT_SIZE = 3

CHANGE_EVERY_N_FRAMES = 1
RANDOMIZE_EACH_FRAME_WHEN_IN_BAND = False

BAND_MIN = 0.10
BAND_MAX = 0.90

BASE_GREEN = 25
MAX_INCREMENT = 200

SOURCE_OSC_FREQ = 0.3
SOURCE_AMPLITUDE = 2.0

EXPECTED_MAX_AMPLITUDE = SOURCE_AMPLITUDE
BRIGHTNESS_GAMMA = 0.6

@dataclass
class CellData:
    digit: str
    was_in_band: bool = False

def compute_brightness(abs_amp: float, frame_max: float) -> int:
    norm = abs_amp / EXPECTED_MAX_AMPLITUDE
    norm = max(0.0, min(1.0, norm))
    norm = norm ** BRIGHTNESS_GAMMA
    return BASE_GREEN + int(norm * MAX_INCREMENT)

def main() -> None:
    init_window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    set_window_state(ConfigFlags.FLAG_WINDOW_RESIZABLE)
    set_target_fps(144)

    wave_sim = WaveEqn2D(nx=55, ny=35, c=0.3, h=1, dt=1, use_mur_abc=True)

    digits: Dict[Tuple[int, int], CellData] = {}
    frame_count = 0

    try:
        while not window_should_close():
            width = get_screen_width()
            height = get_screen_height()

            wave_sim.update()

            cx, cy = wave_sim.nx // 2, wave_sim.ny // 2
            wave_sim.u[0, cy, cx] = np.sin(frame_count * SOURCE_OSC_FREQ) * SOURCE_AMPLITUDE

            field = wave_sim.u[0]
            frame_max = EXPECTED_MAX_AMPLITUDE

            begin_drawing()
            clear_background(BLACK)

            for x in range(0, width, GRID_SPACING):
                for y in range(0, height, GRID_SPACING):
                    key = (x, y)
                    if key not in digits:
                        digits[key] = CellData(digit=str(random.randrange(2)))
                    cell = digits[key]

                    wave_x = int(x * wave_sim.nx / width)
                    wave_y = int(y * wave_sim.ny / height)

                    if 0 <= wave_x < wave_sim.nx and 0 <= wave_y < wave_sim.ny:
                        amp = field[wave_y, wave_x]
                        abs_amp = abs(amp)
                        in_band = BAND_MIN <= abs_amp <= BAND_MAX
                        brightness = compute_brightness(abs_amp, frame_max)
                    else:
                        in_band = False
                        brightness = BASE_GREEN

                    should_randomize = False
                    if in_band:
                        if RANDOMIZE_EACH_FRAME_WHEN_IN_BAND:
                            should_randomize = True
                        elif CHANGE_EVERY_N_FRAMES == 1 or frame_count % CHANGE_EVERY_N_FRAMES == 0:
                            should_randomize = True

                    if should_randomize:
                        cell.digit = str(random.randrange(2))

                    color = Color(0, brightness, 0, 255)
                    draw_text(cell.digit, x, y, FONT_SIZE, color)
                    cell.was_in_band = in_band

            frame_count += 1
            end_drawing()
    finally:
        close_window()

if __name__ == "__main__":
    main()