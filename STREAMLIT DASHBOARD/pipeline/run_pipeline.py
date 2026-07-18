"""
Orchestrates calling Member 1's detection pipeline and Member 2's
analytics pipeline on a freshly uploaded video. Each function tries to
import the real teammate module; if it's not present yet in the project,
it clearly reports that instead of silently doing nothing or faking success.
"""

from pathlib import Path
from typing import Dict, Optional

def run_detection(video_path: Path, output_dir: Path) -> Dict[str, Optional[str]]:
    """
    Attempts to import and call Member 1's process_video(video_path, output_dir).
    Expected module: member1_pipeline.py at project root, with a function:
        def process_video(input_video_path: str, output_dir: str) -> dict
        returning {"tracked_video": ..., "detections_json": ...}
    Returns a dict with keys: success (bool), message (str), outputs (dict or None)
    """
    try:
        from member1_pipeline import process_video
    except ImportError:
        return {
            "success": False,
            "message": "Member 1's detection pipeline (member1_pipeline.py) "
                        "is not yet added to this project.",
            "outputs": None,
        }
    try:
        outputs = process_video(str(video_path), str(output_dir))
        return {"success": True, "message": "Detection completed.", "outputs": outputs}
    except Exception as e:
        return {"success": False, "message": f"Detection pipeline failed: {e}", "outputs": None}

def run_analytics(detections_json_path: Path, output_dir: Path) -> Dict[str, Optional[str]]:
    """
    Same pattern for Member 2. Expected module: member2_pipeline.py at
    project root, with a function:
        def compute_analytics(detections_json_path: str, output_dir: str) -> dict
        returning {"analytics_json": ..., "player_heatmap": ..., "ball_heatmap": ...}
    """
    try:
        from member2_pipeline import compute_analytics
    except ImportError:
        return {
            "success": False,
            "message": "Member 2's analytics pipeline (member2_pipeline.py) "
                        "is not yet added to this project.",
            "outputs": None,
        }
    try:
        outputs = compute_analytics(str(detections_json_path), str(output_dir))
        return {"success": True, "message": "Analytics completed.", "outputs": outputs}
    except Exception as e:
        return {"success": False, "message": f"Analytics pipeline failed: {e}", "outputs": None}