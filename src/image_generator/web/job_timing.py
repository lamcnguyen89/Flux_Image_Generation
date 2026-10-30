"""Elapsed job time and sampling estimates from persisted worker progress."""

from datetime import datetime, timezone
import math


def format_duration(seconds):
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def job_timing(job, *, now=None, worker_available=True):
    now = now or datetime.now(timezone.utc)
    elapsed = None
    if job.started_at:
        end = datetime.fromisoformat(job.finished_at) if job.finished_at else now
        elapsed = max(0, (end - datetime.fromisoformat(job.started_at)).total_seconds())
    timing = {
        "elapsed_seconds": elapsed,
        "elapsed_text": format_duration(elapsed) if elapsed is not None else "Not started",
        "remaining_seconds": None,
        "remaining_text": "Estimating…",
    }
    if job.status == "completed":
        timing.update(remaining_seconds=0, remaining_text=format_duration(0))
    elif job.status in {"failed", "interrupted"}:
        timing["remaining_text"] = "Stopped"
    elif job.status == "queued":
        timing["remaining_text"] = "Waiting in queue"
    elif not worker_available:
        timing["remaining_text"] = "Worker unavailable"
    else:
        denoising = next((event for event in reversed(job.events) if event.stage == "denoising"), None)
        variation = next((event for event in reversed(job.events) if event.stage == "video_variation"), None)
        video_count = job.request["settings"].get("num_videos_per_prompt", 1)
        video_index = variation.current if variation else 1
        later_videos = max(0, video_count - video_index)
        if (job.stage == "denoising" and denoising and job.total
                and job.current is not None and job.current > 0):
            # updated_at is the timestamp of the latest observed step. Using
            # the polling time here would change the rate without new progress.
            sampled_seconds = (datetime.fromisoformat(job.updated_at)
                               - datetime.fromisoformat(denoising.created_at)).total_seconds()
            steps_left = job.total - job.current + later_videos * job.total
            if sampled_seconds > 0 and steps_left > 0:
                remaining = math.ceil(sampled_seconds / job.current * steps_left)
                timing.update(remaining_seconds=remaining, remaining_text=format_duration(remaining))
            elif steps_left == 0:
                timing["remaining_text"] = "Finishing…"
        elif (job.stage in {"decoding", "decoding_video", "encoding_video", "saving", "generating"}
              and denoising and denoising.total and denoising.current == denoising.total
              and later_videos == 0):
            timing["remaining_text"] = "Finishing…"
    return timing
