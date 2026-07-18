"""
generate_dummy_video.py
------------------------
Generates a short placeholder outputs/tracked_video.mp4: a green pitch with
22 moving colored circles (11 team 1, 11 team 2) + a small white ball,
so the Live Match page has something real to display.

Run this once, after generate_sample_data.py:
    python generate_dummy_video.py
"""

import cv2
import numpy as np
from pathlib import Path

from config import OUTPUTS_DIR, TRACKED_VIDEO, VIDEO_FPS

WIDTH, HEIGHT = 1280, 720
DURATION_SEC = 12
NUM_FRAMES = VIDEO_FPS * DURATION_SEC

BLUE = (255, 140, 40)   # BGR
RED = (60, 60, 220)     # BGR
WHITE = (255, 255, 255)
PITCH_GREEN = (60, 140, 40)
LINE_WHITE = (230, 230, 230)


def draw_pitch(frame: np.ndarray) -> None:
    frame[:] = PITCH_GREEN
    cv2.rectangle(frame, (40, 40), (WIDTH - 40, HEIGHT - 40), LINE_WHITE, 2)
    cv2.line(frame, (WIDTH // 2, 40), (WIDTH // 2, HEIGHT - 40), LINE_WHITE, 2)
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 80, LINE_WHITE, 2)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(TRACKED_VIDEO), fourcc, VIDEO_FPS, (WIDTH, HEIGHT))

    rng = np.random.default_rng(42)
    team1_players = rng.integers([80, 80], [WIDTH // 2 - 40, HEIGHT - 80], size=(11, 2)).astype(float)
    team2_players = rng.integers([WIDTH // 2 + 40, 80], [WIDTH - 80, HEIGHT - 80], size=(11, 2)).astype(float)
    blue_vel = rng.uniform(-2, 2, size=(11, 2))
    red_vel = rng.uniform(-2, 2, size=(11, 2))
    ball_pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=float)
    ball_vel = rng.uniform(-3, 3, size=2)

    for i in range(NUM_FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_pitch(frame)

        team1_players += blue_vel
        team2_players += red_vel
        ball_pos += ball_vel

        for arr, vel, bounds in [
            (team1_players, blue_vel, (60, WIDTH - 60, 60, HEIGHT - 60)),
            (team2_players, red_vel, (60, WIDTH - 60, 60, HEIGHT - 60)),
        ]:
            for j in range(len(arr)):
                if arr[j, 0] < bounds[0] or arr[j, 0] > bounds[1]:
                    vel[j, 0] *= -1
                if arr[j, 1] < bounds[2] or arr[j, 1] > bounds[3]:
                    vel[j, 1] *= -1

        if ball_pos[0] < 60 or ball_pos[0] > WIDTH - 60:
            ball_vel[0] *= -1
        if ball_pos[1] < 60 or ball_pos[1] > HEIGHT - 60:
            ball_vel[1] *= -1

        for (x, y) in team1_players:
            cv2.circle(frame, (int(x), int(y)), 12, BLUE, -1)
            cv2.circle(frame, (int(x), int(y)), 12, WHITE, 1)
        for (x, y) in team2_players:
            cv2.circle(frame, (int(x), int(y)), 12, RED, -1)
            cv2.circle(frame, (int(x), int(y)), 12, WHITE, 1)
        cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 7, WHITE, -1)

        cv2.putText(frame, f"Frame {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, WHITE, 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"-> {TRACKED_VIDEO} written ({NUM_FRAMES} frames, {DURATION_SEC}s @ {VIDEO_FPS}fps)")


if __name__ == "__main__":
    main()
