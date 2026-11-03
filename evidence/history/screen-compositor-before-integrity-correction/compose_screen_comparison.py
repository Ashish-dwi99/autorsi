"""Compose two actual matched screen-fixture recordings without speed changes.

Usage: python tools/compose_screen_comparison.py LEFT/result.json RIGHT/result.json OUTDIR
Actor labels come from audited receipts, not a fabricated competitor. This tool
does not record actors, generate gameplay, publish to social media, or infer a
benchmark victory. Exact native videos and source receipts remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_actor(path):
    path = Path(path).resolve()
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("oversized receipt")
    record = json.loads(path.read_bytes())
    if (record.get("context_close_returned") is not True
            or record.get("policy_calls", 0) < 1
            or record.get("video", {}).get("actual_browser_capture") is not True
            or not record.get("frames") or not record.get("actor_label")
            or record.get("split") != "development"):
        raise ValueError("completed actual development recording required")
    video = (path.parent / record["video"]["path"]).resolve()
    if (not video.is_relative_to(path.parent) or video.is_symlink()
            or video.stat().st_size != record["video"]["bytes"]
            or not 0 < video.stat().st_size <= 64 * 1024**2
            or sha(video) != record["video"]["sha256"]):
        raise ValueError("video identity mismatch")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames:format=duration",
        "-of", "json", str(video)], timeout=30))
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    if not 0 < duration <= 120 or int(stream["nb_read_frames"]) < 2:
        raise ValueError("invalid bounded video")
    if {"width": stream["width"], "height": stream["height"]} != record["viewport"]:
        raise ValueError("native recording geometry differs")
    return path, record, video, probe, duration


def compose(left, right, destination):
    a, b = read_actor(left), read_actor(right)
    for key in ("case_id", "fixture_sha256", "goal_sha256", "oracle_sha256", "viewport"):
        if a[1][key] != b[1][key]:
            raise ValueError("comparison cases differ: " + key)
    if shutil.disk_usage(Path(destination).parent).free < 400 * 1024**2:
        raise ValueError("fresh400MiB reserve required")
    out = Path(destination).resolve()
    out.mkdir(exist_ok=False)
    # Visible headings/footnote are rendered as a separate overlay. Screen
    # recordings are only scaled/padded; no action or result is reconstructed.
    from PIL import Image, ImageDraw, ImageFont
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    title_font, detail_font = ImageFont.truetype(font, 36), ImageFont.truetype(font, 23)
    overlay = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, 1920, 119), fill="#142034")
    draw.rectangle((0, 1000, 1920, 1080), fill="#142034")
    for offset, actor in ((0, a), (960, b)):
        label = actor[1]["actor_label"]
        if not 1 <= len(label) <= 45:
            raise ValueError("actor label too long for truthful video caption")
        draw.text((offset + 28, 20), label, font=title_font, fill="white")
        status = "Task completed" if actor[1]["success"] else "Task not completed"
        draw.text((offset + 28, 71), status + " | actual recording, 1x", font=detail_font, fill="#c4d4ee")
    draw.text((28, 1010), "Same synthetic task. Independent recordings aligned at start; final frames held after completion.",
              font=detail_font, fill="white")
    draw.text((28, 1043), "Demonstration only. Different execution environments; no Astra superiority or native OS benchmark claim.",
              font=detail_font, fill="#c4d4ee")
    overlay_path = out / "labels.png"
    overlay.save(overlay_path)
    duration = max(a[4], b[4])
    graph = (f"[0:v]setpts=PTS-STARTPTS,scale=960:880:force_original_aspect_ratio=decrease,"
             f"pad=960:880:(ow-iw)/2:(oh-ih)/2:color=0xeeeeee,setsar=1,fps=25,"
             f"tpad=stop_mode=clone:stop_duration={duration}[l];"
             f"[1:v]setpts=PTS-STARTPTS,scale=960:880:force_original_aspect_ratio=decrease,"
             f"pad=960:880:(ow-iw)/2:(oh-ih)/2:color=0xeeeeee,setsar=1,fps=25,"
             f"tpad=stop_mode=clone:stop_duration={duration}[r];"
             "[l][r]hstack=inputs=2,pad=1920:1080:0:120:color=0x142034[base];"
             "[base][2:v]overlay=0:0:shortest=1[v]")
    result = out / "comparison.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(a[2]), "-i", str(b[2]), "-loop", "1", "-i", str(overlay_path),
        "-filter_complex", graph, "-map", "[v]", "-an", "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(result)], check=True, timeout=300)
    manifest = {"schema": 1, "case_id": a[1]["case_id"], "speed_factor": 1,
        "alignment": "independent native-video starts, not simultaneous execution or frame-exact action timing",
        "no_gameplay_synthesis": True, "published": False,
        "sources": [{"receipt": str(x[0]), "receipt_sha256": sha(x[0]), "video": str(x[2]),
                     "video_sha256": sha(x[2]), "probe": x[3], "actor": x[1]["actor_label"],
                     "actor_identity_declared": x[1].get("actor_identity_declared")} for x in (a, b)],
        "output": {"path": str(result), "sha256": sha(result), "bytes": result.stat().st_size},
        "comparison_claim": "qualitative demonstration; no verified model superiority or cost ratio"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["output"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left"); parser.add_argument("right"); parser.add_argument("destination")
    args = parser.parse_args()
    compose(args.left, args.right, args.destination)
