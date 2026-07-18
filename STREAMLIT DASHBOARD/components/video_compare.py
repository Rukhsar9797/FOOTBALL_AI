import cv2
import numpy as np

from config import ORIGINAL_VIDEO, TRACKED_VIDEO, COMPARISON_VIDEO


def generate_comparison_video():

    cap1 = cv2.VideoCapture(str(ORIGINAL_VIDEO))
    cap2 = cv2.VideoCapture(str(TRACKED_VIDEO))

    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width = width * 2

    writer = cv2.VideoWriter(
        str(COMPARISON_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, height),
    )

    while True:

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break
        
        frame2 = cv2.resize(frame2, (width, height))

        comparison = np.hstack((frame1, frame2))

        cv2.putText(
            comparison,
            "Original",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2,
        )

        cv2.putText(
            comparison,
            "AI Detection",
            (width+30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2,
        )

        cv2.line(
            comparison,
            (width,0),
            (width,height),
            (255,255,255),
            2,
        )

        writer.write(comparison)

    cap1.release()
    cap2.release()
    writer.release()