"""Compose two actual matched screen-fixture recordings without speed changes.

Usage: python tools/compose_screen_comparison.py LEFT/result.json RIGHT/result.json OUTDIR
Actor labels are declared in source receipts; this tool cannot attest identity. It
does not record actors, generate gameplay, publish to social media, or infer a
benchmark victory. Exact native videos and source receipts remain unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

MAX_OUTPUT_BYTES = 128 * 1024**2


def validate_record(record):
    """Reject ambiguous status; do not turn a truthy string into a win caption."""
    if (record.get("context_close_returned") is not True or record.get("policy_pending") is not False
            or record.get("abort_session") is not False or record.get("artifact_error")
            or type(record.get("policy_calls")) is not int or not 1 <= record["policy_calls"] <= 12
            or type(record.get("actions_dispatched")) is not int or not 0 <= record["actions_dispatched"] <= 12
            or type(record.get("success")) is not bool
            or record.get("video", {}).get("actual_browser_capture") is not True
            or not isinstance(record.get("frames"), list) or not 1 <= len(record["frames"]) <= 28
            or record.get("split") != "development"):
        raise ValueError("completed actual development recording without pending operation required")
    label=record.get("actor_label")
    if not isinstance(label,str) or not 1 <= len(label) <= 100 or any(not 32 <= ord(c) <= 126 for c in label):
        raise ValueError("bounded single-line printable actor label required")
    viewport=record.get("viewport",{})
    if (set(viewport)!={"width","height"}
            or any(type(x) is not int or not 1 <= x <= 4096 for x in viewport.values())
            or record["video"].get("record_video_size")!=viewport):
        raise ValueError("exact recorded viewport required")
    if record["success"]:
        final=record.get("final_receipt")
        if (record.get("stop_reason")!="independent_final_state_match" or type(final) is not dict
                or record["actions_dispatched"]<1
                or hashlib.sha256(json.dumps(final,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()!=record.get("oracle_sha256")):
            raise ValueError("successful caption lacks matching final-state receipt")
    elif record.get("stop_reason") not in {"action_limit","policy_done_without_oracle_success","invalid_response_or_ui_error"}:
        raise ValueError("unsupported or uncertain recording stop reason")


def recipe_caption(record):
    identity=record.get("actor_identity_declared") or {}
    if not isinstance(identity,dict):raise ValueError("declared actor identity must be an object")
    values=[]
    for key,label in (("attention_scale","attention"),("visual_gain","vision gain")):
        if key in identity:
            value=identity[key]
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError("invalid declared recipe value")
            values.append(f"{label}={value:g}")
    return " | ".join(values) if values else "Actor identity declared by receipt; external review required"


def media_path(folder, relative):
    relative=Path(relative)
    if relative.is_absolute() or not relative.parts or any(x in (".","..") for x in relative.parts):
        raise ValueError("ordinary relative video path required")
    candidate=folder
    for part in relative.parts:
        candidate=candidate/part
        if candidate.is_symlink():raise ValueError("symlink media path refused")
    if not candidate.is_file() or not candidate.resolve().is_relative_to(folder):
        raise ValueError("video outside receipt directory")
    return candidate


def probe_video(video, viewport, *, expected_duration=None):
    result=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v",
        "-show_entries","stream=width,height,nb_read_frames,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation:format=duration",
        "-of","json",str(video)],capture_output=True,timeout=30,check=True)
    # These are post-return acceptance limits, not subprocess pipe write quotas.
    if result.stderr or len(result.stdout)>64*1024:raise ValueError("media probe reported errors or oversized metadata")
    probe=json.loads(result.stdout)
    if len(probe.get("streams",[]))!=1:raise ValueError("exactly one video stream required")
    stream=probe['streams'][0];duration=float(probe['format']['duration'])
    if not math.isfinite(duration) or not 0 < duration <= 120 or int(stream['nb_read_frames'])<2:
        raise ValueError("invalid bounded video")
    if {"width":stream['width'],"height":stream['height']}!=viewport:
        raise ValueError("native recording geometry differs")
    if stream.get('sample_aspect_ratio') not in (None,'N/A','1:1'):
        raise ValueError("non-square source pixels cannot be silently relabeled")
    rotations=[stream.get('tags',{}).get('rotate',0),*[x.get('rotation',0) for x in stream.get('side_data_list',[])]]
    if any(float(x)!=0 for x in rotations):raise ValueError("rotated media refused; no silent autorotation")
    if expected_duration is not None and abs(duration-expected_duration)>0.08:
        raise ValueError("rendered duration does not preserve original longest clip")
    return probe,duration


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_actor(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError("oversized receipt")
    path=path.resolve()
    record = json.loads(path.read_bytes())
    validate_record(record)
    video = media_path(path.parent,record["video"]["path"])
    if (type(record["video"].get("bytes")) is not int or video.stat().st_size != record["video"]["bytes"]
            or not 0 < video.stat().st_size <= 64 * 1024**2
            or sha(video) != record["video"]["sha256"]):
        raise ValueError("video identity mismatch")
    probe,duration=probe_video(video,record["viewport"])
    return path, record, video, probe, duration


def compose(left, right, destination, *, font_path="/System/Library/Fonts/Supplemental/Arial.ttf"):
    a, b = read_actor(left), read_actor(right)
    for key in ("case_id", "fixture_sha256", "goal_sha256", "oracle_sha256", "viewport"):
        if a[1][key] != b[1][key]:
            raise ValueError("comparison cases differ: " + key)
    if a[2]==b[2] or a[1]['video']['sha256']==b[1]['video']['sha256']:
        raise ValueError("same source recording cannot stand in for two actors")
    if (a[1]['actor_label'],recipe_caption(a[1]))==(b[1]['actor_label'],recipe_caption(b[1])):
        raise ValueError("visible actor/recipe labels must distinguish the actual pair")
    source_pins=[{'receipt':sha(x[0]),'video':sha(x[2])} for x in (a,b)]
    if shutil.disk_usage(Path(destination).parent).free < 400 * 1024**2:
        raise ValueError("fresh400MiB reserve required")
    out = Path(destination).resolve()
    out.mkdir(exist_ok=False)
    # Visible headings/footnote are rendered as a separate overlay. Screen
    # recordings are only scaled/padded; no action or result is reconstructed.
    from PIL import Image, ImageDraw, ImageFont
    font=str(Path(font_path).resolve())
    title_font, detail_font = ImageFont.truetype(font, 32), ImageFont.truetype(font, 20)
    overlay = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, 1920, 119), fill="#142034")
    draw.rectangle((0, 1000, 1920, 1080), fill="#142034")
    for offset, actor in ((0, a), (960, b)):
        label = actor[1]["actor_label"]
        if draw.textlength(label,font=title_font)>904:
            raise ValueError("actor label exceeds its panel; do not truncate identity")
        draw.text((offset + 28, 20), label, font=title_font, fill="white")
        status = "Final recorded outcome: complete" if actor[1]["success"] else "Final recorded outcome: not completed"
        detail=recipe_caption(actor[1])
        if draw.textlength(detail,font=detail_font)>904:raise ValueError("recipe caption exceeds panel")
        draw.text((offset + 28, 61), status + " | 1x", font=detail_font, fill="#c4d4ee")
        draw.text((offset + 28, 88), detail, font=detail_font, fill="#c4d4ee")
    draw.text((28, 1010), "Same synthetic task. Independent clip starts; final frames held after each clip ends.",
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
        "-noautorotate", "-i", str(a[2]), "-noautorotate", "-i", str(b[2]), "-loop", "1", "-i", str(overlay_path),
        "-filter_complex", graph, "-map", "[v]", "-an", "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-fs", str(MAX_OUTPUT_BYTES), str(result)], check=True, timeout=300)
    if not 0<result.stat().st_size<=MAX_OUTPUT_BYTES:raise ValueError("rendered output exceeds128MiB")
    output_probe,_=probe_video(result,{"width":1920,"height":1080},expected_duration=duration)
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-xerror",
        "-err_detect","explode","-i",str(result),"-map","0:v:0","-f","null","-"],check=True,timeout=300)
    for x,pins in zip((a,b),source_pins,strict=True):
        if sha(x[0])!=pins['receipt'] or sha(x[2])!=pins['video']:
            raise ValueError("source receipt/video changed during composition; output not admitted")
    manifest = {"schema": 1, "case_id": a[1]["case_id"], "speed_factor": 1,
        "alignment": "independent native-video starts, not simultaneous execution or frame-exact action timing",
        "no_gameplay_synthesis": True, "published": False,
        "sources": [{"receipt": str(x[0]), "receipt_sha256": sha(x[0]), "video": str(x[2]),
                     "video_sha256": sha(x[2]), "probe": x[3], "actor": x[1]["actor_label"],
                     "actor_identity_declared": x[1].get("actor_identity_declared")} for x in (a, b)],
        "output": {"path": str(result), "sha256": sha(result), "bytes": result.stat().st_size,
                   "probe":output_probe,"full_decode_returned":True},
        "rendering": {"output_fps":25,"timebase_effect":"Frame resampling may duplicate/drop frames; no elapsed-time speedup; final shorter clip frame held", "font_sha256":sha(Path(font))},
        "outcome_validation":"Receipt boolean and successful final-state hash joined; no independent replay or actor/model identity attestation",
        "comparison_claim": "qualitative demonstration; no verified model superiority or cost ratio"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["output"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left"); parser.add_argument("right"); parser.add_argument("destination")
    parser.add_argument("--font",default="/System/Library/Fonts/Supplemental/Arial.ttf")
    args = parser.parse_args()
    compose(args.left, args.right, args.destination,font_path=args.font)
