import cv2
import time

from utils import load_json, group_by_frame

from distance import DistanceTracker
from speed import SpeedTracker
from possession import PossessionTracker
from heatmap import LiveHeatmap
from activity_score import ActivityTracker
from momentum import MomentumTracker
from analytics_output import save_analytics


VIDEO_PATH = "input/original_video.mp4"
JSON_PATH = "input/detections.json"

OUTPUT_VIDEO = "output/final_tracked_video.mp4"


# Heatmap states
show_ball_heatmap = False
show_red_heatmap = False
show_white_heatmap = False



def blend_heatmap(frame, heatmap, alpha=0.45):

    heatmap = cv2.resize(
        heatmap,
        (frame.shape[1], frame.shape[0])
    )

    if len(heatmap.shape) == 2:
        heatmap = cv2.applyColorMap(
            heatmap.astype("uint8"),
            cv2.COLORMAP_JET
        )

    frame = cv2.addWeighted(
        frame,
        0.55,
        heatmap,
        alpha,
        0
    )

    return frame



def draw_dashboard(frame,
                   possession_tracker,
                   distance_tracker,
                   speed_tracker,
                   activity_tracker,
                   momentum_tracker,
                   fps):

    y = 40

    try:

        possessor = possession_tracker.get_current_possessor()

        possession = possession_tracker.get_team_percentage()

        distance_rank = distance_tracker.get_rankings()

        speed = speed_tracker.get_top_speed_player()

        active = activity_tracker.get_most_active_player()

        momentum = momentum_tracker.get_momentum()


        texts = [

            "AI FOOTBALL ANALYTICS",

            f"FPS : {fps:.1f}",

            f"BALL HOLDER : Player {possessor}",

            f"RED POSSESSION : {possession['Red Team']} %",

            f"WHITE POSSESSION : {possession['White Team']} %",

            "---------------------------",

            "TOP DISTANCE PLAYERS"

        ]


        for i, player in enumerate(distance_rank[:3]):

            texts.append(
                f"{i+1}. Player {player[0]} : {player[1]:.1f} m"
            )


        texts.extend([

            "---------------------------",

            f"HIGHEST SPEED : Player {speed['player_id']}",

            f"MOST ACTIVE : Player {active['player_id']}",

            f"MOMENTUM : {momentum['Dominating']}"

        ])


        for text in texts:

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0,255,100),
                2,
                cv2.LINE_AA
            )

            y += 32


    except Exception as e:

        cv2.putText(
            frame,
            str(e),
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2,
            cv2.LINE_AA
        )




def run_live_analysis():

    global show_ball_heatmap
    global show_red_heatmap
    global show_white_heatmap


    print("Loading detections...")


    detections = load_json(JSON_PATH)

    frames = group_by_frame(detections)


    print("Frames:", len(frames))


    distance_tracker = DistanceTracker()

    speed_tracker = SpeedTracker()

    possession_tracker = PossessionTracker()

    heatmap_tracker = LiveHeatmap()

    activity_tracker = ActivityTracker()

    momentum_tracker = MomentumTracker()



    video = cv2.VideoCapture(VIDEO_PATH)


    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    fps_video = video.get(
        cv2.CAP_PROP_FPS
    )


    writer = cv2.VideoWriter(

        OUTPUT_VIDEO,

        cv2.VideoWriter_fourcc(*"mp4v"),

        fps_video,

        (width,height)

    )



    frame_number = 0

    previous_time = time.time()



    while True:

        ret, frame = video.read()

        if not ret:
            print(f"\nVideo ended at frame {frame_number}")
            break

        current_time = time.time()

        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Objects in current frame
        objects = frames.get(frame_number, [])

        # Update trackers
        distance_tracker.update(objects)
        speed_tracker.update(objects)
        possession_tracker.update(objects)
        heatmap_tracker.update(objects)

        activity_tracker.update(
            distance_tracker,
            speed_tracker,
            possession_tracker
        )

        momentum_tracker.update(
            objects,
            possession_tracker
        )

        # Heatmaps
        if show_ball_heatmap:
            frame = blend_heatmap(
                frame,
                heatmap_tracker.get_ball_heatmap()
            )

        if show_red_heatmap:
            frame = blend_heatmap(
                frame,
                heatmap_tracker.get_red_heatmap()
            )

        if show_white_heatmap:
            frame = blend_heatmap(
                frame,
                heatmap_tracker.get_white_heatmap()
            )

        # Dashboard
        draw_dashboard(
            frame,
            possession_tracker,
            distance_tracker,
            speed_tracker,
            activity_tracker,
            momentum_tracker,
            fps
        )

        writer.write(frame)

        cv2.imshow(
            "AI Football Analytics",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):
            show_ball_heatmap = not show_ball_heatmap

        elif key == ord('r'):
            show_red_heatmap = not show_red_heatmap

        elif key == ord('w'):
            show_white_heatmap = not show_white_heatmap

        elif key == ord('n'):
            show_ball_heatmap = False
            show_red_heatmap = False
            show_white_heatmap = False

        elif key == 27:      # ESC
            break

        frame_number += 1
        

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    save_analytics(
        possession_tracker,
        distance_tracker,
        speed_tracker,
        activity_tracker,
        momentum_tracker
    )

    print("Saved:", OUTPUT_VIDEO)