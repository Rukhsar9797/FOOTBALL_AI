"""
Batch runner for SmartMatch AI Tracker.

Runs the tracking pipeline over multiple videos at once (default: 4 in
parallel), each getting its own output folder so results never collide.

Usage examples
--------------
# Process every video in a folder, 4 at a time:
python batch_runner.py --input_dir data/uploads --output_root outputs --max_workers 4

# Process specific files:
python batch_runner.py --input data/match1.mp4 data/match2.mp4 data/match3.mp4 data/match4.mp4

# Mock mode (no real videos needed, generates synthetic test clips):
python batch_runner.py --mock --num_mock 4
"""
import os
import sys
import glob
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from smartmatch_pipeline import run_pipeline

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _worker(job):
    """
    Runs in a separate process. Each job gets its own output_dir and a
    log_prefix so interleaved stdout from parallel workers stays readable.
    """
    input_path = job["input_path"]
    output_dir = job["output_dir"]
    mock_mode = job["mock_mode"]
    name = os.path.splitext(os.path.basename(input_path))[0]
    return run_pipeline(
        input_path,
        mock_mode=mock_mode,
        output_dir=output_dir,
        log_prefix=f"[{name}] ",
    )


def discover_videos(input_dir):
    found = []
    for ext in VIDEO_EXTENSIONS:
        found.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(found)


def build_jobs(video_paths, output_root, mock_mode):
    jobs = []
    for path in video_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        jobs.append({
            "input_path": path,
            "output_dir": os.path.join(output_root, name),
            "mock_mode": mock_mode,
        })
    return jobs


def merge_detections(results, output_root, combined_filename="combined_detections.json"):
    """
    Reads each video's own detections.json (written by run_pipeline) and
    merges them into a single JSON file, tagging every record with which
    video it came from so they stay distinguishable once combined.
    """
    combined_records = []
    for result in results:
        if result.get("status") != "success":
            continue
        json_path = result.get("output_json")
        if not json_path or not os.path.exists(json_path):
            continue
        video_name = os.path.splitext(os.path.basename(result["input"]))[0]
        with open(json_path, "r") as f:
            records = json.load(f)
        for rec in records:
            rec["video"] = video_name
            combined_records.append(rec)

    combined_path = os.path.join(output_root, combined_filename)
    with open(combined_path, "w") as f:
        json.dump(combined_records, f, indent=2)
    return combined_path, len(combined_records)


def run_batch(jobs, max_workers=4):
    """
    Processes jobs in parallel, at most `max_workers` at a time. If there
    are more than max_workers jobs, ProcessPoolExecutor automatically
    queues the rest and starts them as workers free up -- you don't need
    to chunk the list yourself.
    """
    results = []
    total = len(jobs)
    print(f"Starting batch of {total} video(s), up to {max_workers} running at once...\n")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(_worker, job): job for job in jobs}
        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = {"input": job["input_path"], "status": "error", "error": str(exc)}
            results.append(result)
            status = result.get("status")
            print(f"[{completed}/{total}] {job['input_path']} -> {status}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch runner for SmartMatch AI Tracker")
    parser.add_argument("--input", nargs="+", help="One or more explicit video file paths")
    parser.add_argument("--input_dir", type=str, help="Folder to scan for video files")
    parser.add_argument("--output_root", type=str, default="outputs_batch",
                         help="Root folder; each video gets its own subfolder here")
    parser.add_argument("--max_workers", type=int, default=4,
                         help="How many videos to process concurrently (default 4)")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic test videos instead of real input")
    parser.add_argument("--num_mock", type=int, default=4, help="Number of synthetic videos to generate in --mock mode")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    if args.mock:
        video_paths = [
            os.path.join(args.output_root, "_mock_inputs", f"mock_{i+1}.mp4")
            for i in range(args.num_mock)
        ]
    elif args.input:
        video_paths = args.input
    elif args.input_dir:
        video_paths = discover_videos(args.input_dir)
        if not video_paths:
            print(f"No video files found in '{args.input_dir}'")
            sys.exit(1)
    else:
        print("Provide --input <files...>, --input_dir <folder>, or --mock")
        sys.exit(1)

    jobs = build_jobs(video_paths, args.output_root, args.mock)
    results = run_batch(jobs, max_workers=args.max_workers)

    summary_path = os.path.join(args.output_root, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r.get("status") == "success")
    print(f"\nBatch complete: {ok}/{len(results)} succeeded.")
    print(f"Per-video outputs are under: {args.output_root}/<video_name>/")
    print(f"Summary written to: {summary_path}")

    combined_path, num_records = merge_detections(results, args.output_root)
    print(f"Combined JSON ({num_records} records across {ok} video(s)) written to: {combined_path}")


if __name__ == "__main__":
    main() 