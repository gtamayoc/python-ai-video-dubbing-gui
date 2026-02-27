"""
validate_pipeline.py
────────────────────
Standalone validation helper for the AI dubbing pipeline.

Runs the full pipeline on a short audio file (or a generated sine-wave
test clip) and prints timing information for each stage plus a duration-
match report.

Usage:
    python validate_pipeline.py                           # uses test tone
    python validate_pipeline.py path/to/audio.mp3        # real audio
    python validate_pipeline.py path/to/audio.mp3 simple # force simple mode
"""

from __future__ import annotations

import os
import sys
import time
import logging
import subprocess
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("validate")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_test_audio(path: str, duration_s: float = 15.0) -> None:
    """Generate a silent (or sine-wave) test clip using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_s}",
            "-acodec", "libmp3lame", "-ab", "128k",
            path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Created test audio: %s (%.1fs)", path, duration_s)


def _get_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    input_path    = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline_mode = sys.argv[2] if len(sys.argv) > 2 else "advanced"

    # Fix huggingface symlinks on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        import huggingface_hub.file_download
        huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False
    except ImportError:
        pass

    # Load .env so HF_TOKEN is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    with tempfile.TemporaryDirectory(prefix="dubbing_validate_") as tmp:
        # Prepare input
        if input_path and os.path.exists(input_path):
            src = input_path
        else:
            logger.info("No input supplied — generating a 15-second test tone")
            src = os.path.join(tmp, "test_input.mp3")
            _make_test_audio(src, 15.0)

        src_dur = _get_duration(src)
        out_path = os.path.join(tmp, "output_es.wav")

        logger.info("=" * 60)
        logger.info("Input  : %s (%.2fs)", src, src_dur)
        logger.info("Mode   : %s", pipeline_mode)
        logger.info("Output : %s", out_path)
        logger.info("=" * 60)

        # Run pipeline
        from src.services.dubbing_service import DubbingService

        stage_log: list = []

        def progress(stage: str, msg: str) -> None:
            ts = time.perf_counter()
            entry = f"[{stage:15s}] {msg}"
            stage_log.append((ts, entry))
            print(entry)

        svc = DubbingService()
        t0  = time.perf_counter()
        try:
            svc.run_pipeline(
                src,
                out_path,
                progress_callback=progress,
                pipeline_mode=pipeline_mode,
            )
            total = time.perf_counter() - t0

            # Duration check
            out_dur = _get_duration(out_path)
            delta   = abs(out_dur - src_dur)
            passed  = delta <= 0.5

            print()
            print("=" * 60)
            print(f"  Total elapsed      : {total:.2f}s")
            print(f"  Source duration    : {src_dur:.2f}s")
            print(f"  Output duration    : {out_dur:.2f}s")
            print(f"  Delta              : {delta:.3f}s  {'✅ PASS' if passed else '⚠  FAIL (>0.5s)'}")
            print(f"  Realtime factor    : {src_dur / total:.2f}x")
            print("=" * 60)

        except Exception as e:
            logger.exception("Validation pipeline failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
