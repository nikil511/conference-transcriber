#!/usr/bin/env python3
"""
merge_transcript.py
-------------------
Merges fragmented subtitle/speech segments from the same speaker into full sentences
and coherent paragraphs.

Can be used as a standalone script:
    python3 merge_transcript.py input.srt -o output_dir

Or imported into transcriber.py:
    from merge_transcript import (
        merge_speaker_turns,
        generate_merged_srt,
        generate_transcript_text,
        generate_transcript_markdown,
        process_srt_file,
    )
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Optional


def parse_srt(srt_content: str) -> List[Dict]:
    """
    Parses SRT content into a list of segment dictionaries:
    [{'index': 1, 'start': '00:00:01,740', 'end': '00:01:01,840',
      'start_sec': 1.74, 'end_sec': 61.84, 'speaker': 'Speaker 06', 'text': '...'}]
    """
    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"
        r"(?:\[([^\]]+)\]\s*)?"
        r"(.*?)(?=\n\s*\n|\Z)",
        re.DOTALL,
    )

    def _to_sec(ts: str) -> float:
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s

    segments = []
    for match in pattern.finditer(srt_content):
        idx_str, start_ts, end_ts, speaker, text = match.groups()
        clean_text = " ".join(text.strip().split())
        if not clean_text:
            continue
        speaker_name = speaker.strip() if speaker else "UNKNOWN"
        speaker_name = speaker_name.replace("SPEAKER_", "Speaker ")

        segments.append({
            "index": int(idx_str),
            "start": start_ts.replace(".", ","),
            "end": end_ts.replace(".", ","),
            "start_sec": _to_sec(start_ts),
            "end_sec": _to_sec(end_ts),
            "speaker": speaker_name,
            "text": clean_text,
        })

    return segments


def _sec_to_srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def _sec_to_clock(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return "%02d:%02d:%02d" % (h, m, s)
    return "%02d:%02d" % (m, s)


def merge_speaker_turns(
    segments: List[Dict],
    max_silence_gap: Optional[float] = None
) -> List[Dict]:
    """
    Merges consecutive segments spoken by the same speaker into coherent turns.

    Args:
        segments: List of dicts with 'speaker', 'text', 'start_sec' (or 'start'), 'end_sec' (or 'end').
        max_silence_gap: If specified (e.g. 5.0), splits the turn if pause between segments
                         exceeds this duration. If None, merges all consecutive blocks of the same speaker.

    Returns:
        List of merged turn dictionaries.
    """
    if not segments:
        return []

    turns = []
    current = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        spk = seg.get("speaker", "UNKNOWN")
        spk = spk.replace("SPEAKER_", "Speaker ")

        # Support both parsed SRT dicts and transcriber.py internal segment dicts
        start_sec = seg.get("start_sec")
        if start_sec is None:
            start_sec = float(seg.get("start", 0.0))

        end_sec = seg.get("end_sec")
        if end_sec is None:
            end_sec = float(seg.get("end", 0.0))

        if current is None:
            current = {
                "speaker": spk,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": text,
            }
        else:
            same_speaker = (current["speaker"] == spk)
            gap = start_sec - current["end_sec"]
            gap_ok = (max_silence_gap is None) or (gap <= max_silence_gap)

            if same_speaker and gap_ok:
                current["end_sec"] = max(current["end_sec"], end_sec)
                # Concatenate with proper spacing
                current["text"] += " " + text
            else:
                turns.append(current)
                current = {
                    "speaker": spk,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "text": text,
                }

    if current is not None:
        turns.append(current)

    # Format timestamps
    for i, t in enumerate(turns, 1):
        t["index"] = i
        t["start"] = _sec_to_srt_ts(t["start_sec"])
        t["end"] = _sec_to_srt_ts(t["end_sec"])
        t["start_clock"] = _sec_to_clock(t["start_sec"])
        t["end_clock"] = _sec_to_clock(t["end_sec"])

    return turns


def generate_merged_srt(turns: List[Dict]) -> str:
    """Generates an SRT file with one block per merged speaker turn."""
    blocks = []
    for t in turns:
        blocks.append(
            "%d\n%s --> %s\n[%s] %s\n"
            % (t["index"], t["start"], t["end"], t["speaker"], t["text"])
        )
    return "\n".join(blocks)


def generate_transcript_text(turns: List[Dict], title: str = "Conference Transcript") -> str:
    """Generates a clean text document with timestamps and speaker paragraphs."""
    lines = [
        "=" * 70,
        f"  {title.upper()}",
        "=" * 70,
        "",
    ]
    for t in turns:
        lines.append(f"[{t['start_clock']} - {t['end_clock']}] {t['speaker']}:")
        lines.append(f"{t['text']}")
        lines.append("")
    return "\n".join(lines)


def generate_transcript_markdown(turns: List[Dict], title: str = "Conference Transcript") -> str:
    """Generates a GitHub-flavored Markdown transcript."""
    lines = [
        f"# {title}",
        "",
        "> Meeting transcript with connected speaker turns and full sentences.",
        "",
        "---",
        "",
    ]
    for t in turns:
        lines.append(f"### `[{t['start_clock']}]` {t['speaker']}")
        lines.append(f"{t['text']}")
        lines.append("")
    return "\n".join(lines)


def process_srt_file(
    srt_path: str,
    output_dir: Optional[str] = None,
    max_silence_gap: Optional[float] = None
) -> Dict[str, str]:
    """
    Reads an SRT file, merges consecutive speaker blocks, and writes out:
      1. <name>_merged.srt
      2. <name>_transcript.txt
      3. <name>_transcript.md

    Returns a dict mapping output types to their saved file paths.
    """
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = parse_srt(content)
    if not segments:
        raise ValueError(f"No valid subtitle blocks found in {srt_path}")

    turns = merge_speaker_turns(segments, max_silence_gap=max_silence_gap)

    src_dir, src_name = os.path.split(srt_path)
    base_name = os.path.splitext(src_name)[0]
    # Clean up standard suffix if present
    base_name = re.sub(r"(_transcribed|_merged)$", "", base_name)

    target_dir = output_dir if output_dir and os.path.isdir(output_dir) else src_dir

    merged_srt_path = os.path.join(target_dir, f"{base_name}_merged.srt")
    txt_path = os.path.join(target_dir, f"{base_name}_transcript.txt")
    md_path = os.path.join(target_dir, f"{base_name}_transcript.md")

    merged_srt = generate_merged_srt(turns)
    txt_content = generate_transcript_text(turns, title=base_name)
    md_content = generate_transcript_markdown(turns, title=base_name)

    with open(merged_srt_path, "w", encoding="utf-8") as f:
        f.write(merged_srt)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "merged_srt": merged_srt_path,
        "transcript_txt": txt_path,
        "transcript_md": md_path,
        "turns_count": len(turns),
        "orig_count": len(segments),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge consecutive speaker subtitles in an SRT into full sentences."
    )
    parser.add_argument("srt_file", type=str, help="Path to input .srt file")
    parser.add_argument("-o", "--output-dir", type=str, default="", help="Output directory")
    parser.add_argument(
        "-g", "--gap", type=float, default=None,
        help="Max silence gap in seconds before splitting speaker turn (default: None, merge all)"
    )

    args = parser.parse_args()

    try:
        results = process_srt_file(args.srt_file, output_dir=args.output_dir or None, max_silence_gap=args.gap)
        print("=" * 60)
        print(f"Merged {results['orig_count']} segments into {results['turns_count']} speaker turns.")
        print("Generated files:")
        print(f"  Merged SRT:   {results['merged_srt']}")
        print(f"  Text Transcript: {results['transcript_txt']}")
        print(f"  Markdown Transcript: {results['transcript_md']}")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
