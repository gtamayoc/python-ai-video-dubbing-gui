"""
app.py
======
Tkinter GUI for English → Spanish audio translator with voice cloning.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from translator_service import TranslatorService, models


# -- Theme colours ---------------------------------------------------------
BG       = "#1e1e2e"
CARD     = "#282840"
FG       = "#cdd6f4"
DIM      = "#7f849c"
ACCENT   = "#89b4fa"
HOVER    = "#74c7ec"
SUCCESS  = "#a6e3a1"
ERROR    = "#f38ba8"
BORDER   = "#45475a"

# -- Pipeline stages in order ----------------------------------------------
STAGE_LIST = [
    "Extracting",
    "Diarizing",
    "Transcribing",
    "Translating",
    "Cloning Voice",
    "Saving",
]
TOTAL_STAGES = len(STAGE_LIST)


def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class VideoDubberApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title("AI Audio Translator — English → Spanish")
        self.geometry("740x580")
        self.minsize(640, 500)
        self.configure(bg=BG)

        self._input_path: str | None = None
        self._output_path: str | None = None
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._service = TranslatorService()
        self._worker: threading.Thread | None = None
        self._start_time: float = 0
        self._current_stage_idx: int = 0
        self._models_loaded: bool = False

        self._styles()
        self._build()

    # ----- styles --------------------------------------------------------

    def _styles(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("H.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 16, "bold"))
        s.configure("Dim.TLabel", background=BG, foreground=DIM, font=("Segoe UI", 9))
        s.configure("Stage.TLabel", background=BG, foreground=SUCCESS, font=("Segoe UI", 10, "bold"))
        s.configure("Time.TLabel", background=BG, foreground=DIM, font=("Segoe UI", 9))
        s.configure("Pct.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        s.configure("Accent.TButton", background=ACCENT, foreground="#1e1e2e",
                     font=("Segoe UI", 10, "bold"), borderwidth=0, padding=(12, 7))
        s.map("Accent.TButton", background=[("active", HOVER), ("disabled", BORDER)],
              foreground=[("disabled", DIM)])
        s.configure("Sec.TButton", background=CARD, foreground=FG,
                     font=("Segoe UI", 10), borderwidth=1, padding=(12, 7))
        s.map("Sec.TButton", background=[("active", BORDER)])
        s.configure("Bar.Horizontal.TProgressbar", troughcolor=CARD, background=ACCENT,
                     thickness=10, borderwidth=0)

    # ----- layout --------------------------------------------------------

    def _build(self) -> None:
        f = ttk.Frame(self); f.pack(fill="both", expand=True, padx=18, pady=12)

        ttk.Label(f, text="🎙  Audio Translator", style="H.TLabel").pack(anchor="w")
        ttk.Label(f, text="Select an English audio file and translate it to Spanish.",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 10))

        # --- input row
        row1 = ttk.Frame(f); row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="Input:").pack(side="left")
        self._in_var = tk.StringVar(value="No file selected")
        ttk.Entry(row1, textvariable=self._in_var, state="readonly", width=52,
                  font=("Segoe UI", 9)).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row1, text="Select audio file", style="Sec.TButton",
                   command=self._pick).pack(side="right")

        # --- output row
        row2 = ttk.Frame(f); row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Output:").pack(side="left")
        self._out_var = tk.StringVar(value="—")
        ttk.Entry(row2, textvariable=self._out_var, state="readonly", width=52,
                  font=("Segoe UI", 9)).pack(side="left", padx=6, fill="x", expand=True)

        # --- translate button
        bf = ttk.Frame(f); bf.pack(fill="x", pady=(8, 4))
        self._btn = ttk.Button(bf, text="Translate to Spanish", style="Accent.TButton",
                               command=self._start)
        self._btn.pack(side="left")

        # --- progress section
        pf = ttk.Frame(f); pf.pack(fill="x", pady=(10, 0))

        # percentage label + progress bar side by side
        top_row = ttk.Frame(pf); top_row.pack(fill="x")
        self._pct_var = tk.StringVar(value="")
        ttk.Label(top_row, textvariable=self._pct_var, style="Pct.TLabel").pack(side="left")
        self._time_var = tk.StringVar(value="")
        ttk.Label(top_row, textvariable=self._time_var, style="Time.TLabel").pack(side="right")

        self._pvar = tk.DoubleVar(value=0)
        ttk.Progressbar(pf, variable=self._pvar, maximum=100, mode="determinate",
                        style="Bar.Horizontal.TProgressbar").pack(fill="x", pady=(4, 2))

        # stage label
        self._stage_var = tk.StringVar(value="Idle")
        ttk.Label(f, textvariable=self._stage_var, style="Stage.TLabel").pack(anchor="w", pady=(2, 6))

        # --- log
        ttk.Label(f, text="Log", style="Dim.TLabel").pack(anchor="w")
        self._log = scrolledtext.ScrolledText(f, height=10, wrap="word", state="disabled",
            font=("Consolas", 9), bg=CARD, fg=FG, insertbackground=FG,
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT)
        self._log.pack(fill="both", expand=True, pady=(2, 0))
        self._log.tag_configure("info", foreground=FG)
        self._log.tag_configure("stage", foreground=ACCENT)
        self._log.tag_configure("ok", foreground=SUCCESS)
        self._log.tag_configure("err", foreground=ERROR)

    # ----- actions -------------------------------------------------------

    def _pick(self) -> None:
        path = filedialog.askopenfilename(
            title="Select English audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"), ("All", "*.*")],
        )
        if not path:
            return
        self._input_path = os.path.normpath(path)
        self._in_var.set(self._input_path)

        base, ext = os.path.splitext(self._input_path)
        self._output_path = f"{base}_es{ext}"
        self._out_var.set(self._output_path)

        self._write_log(f"Selected: {self._input_path}", "info")
        self._write_log(f"Output:   {self._output_path}", "info")

    def _start(self) -> None:
        if not self._input_path:
            self._write_log("⚠ Select an audio file first.", "err")
            return
        if self._worker and self._worker.is_alive():
            self._write_log("⚠ Translation already running.", "err")
            return

        self._pvar.set(0)
        self._pct_var.set("0%")
        self._time_var.set("")
        self._stage_var.set("Starting…")
        self._current_stage_idx = 0
        self._start_time = time.time()
        self._btn.state(["disabled"])
        self._write_log("━" * 46, "info")

        # Preload models on first run (warm start)
        if not self._models_loaded:
            self._write_log("Loading ML models (first run only)…", "stage")
            self._models_loaded = True

        self._write_log("Pipeline started…", "stage")

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._poll()

    # ----- background worker ---------------------------------------------

    def _run(self) -> None:
        try:
            self._service.run_pipeline(self._input_path, self._output_path,
                                       progress_callback=self._cb)
            self._queue.put(("__done__", ""))
        except Exception as exc:
            self._queue.put(("__error__", str(exc)))

    def _cb(self, stage: str, msg: str) -> None:
        self._queue.put((stage, msg))

    # ----- queue polling (UI thread) -------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                stage, msg = self._queue.get_nowait()

                if stage == "__done__":
                    elapsed = time.time() - self._start_time
                    self._pvar.set(100)
                    self._pct_var.set("100%")
                    self._time_var.set(f"Total: {_fmt_time(elapsed)}")
                    self._stage_var.set("Completed ✔")
                    self._write_log(f"Pipeline finished in {_fmt_time(elapsed)}!", "ok")
                    self._btn.state(["!disabled"])
                    return

                if stage == "__error__":
                    self._stage_var.set("Error")
                    self._pct_var.set("—")
                    self._time_var.set("")
                    self._write_log(f"❌ {msg}", "err")
                    self._btn.state(["!disabled"])
                    return

                # Update stage index
                if stage in STAGE_LIST:
                    self._current_stage_idx = STAGE_LIST.index(stage)

                # Calculate percentage (each stage = equal weight)
                pct = int(((self._current_stage_idx + 1) / TOTAL_STAGES) * 100)
                # Show partial progress within stage (first msg = start, second = done)
                stage_base_pct = int((self._current_stage_idx / TOTAL_STAGES) * 100)
                stage_end_pct = int(((self._current_stage_idx + 1) / TOTAL_STAGES) * 100)
                # Use midpoint for first message, full for second
                if pct > self._pvar.get():
                    self._pvar.set(pct)

                self._pct_var.set(f"{int(self._pvar.get())}%")

                # Time elapsed + estimate
                elapsed = time.time() - self._start_time
                if pct > 0 and pct < 100:
                    estimated_total = elapsed / (pct / 100)
                    remaining = estimated_total - elapsed
                    self._time_var.set(
                        f"⏱ {_fmt_time(elapsed)}  ·  ~{_fmt_time(remaining)} remaining"
                    )
                else:
                    self._time_var.set(f"⏱ {_fmt_time(elapsed)}")

                self._stage_var.set(
                    f"Step {self._current_stage_idx + 1}/{TOTAL_STAGES} — {stage}"
                )
                self._write_log(f"[{stage}] {msg}", "stage")

        except queue.Empty:
            # Update elapsed time even when no new messages
            if self._start_time:
                elapsed = time.time() - self._start_time
                current_pct = self._pvar.get()
                if 0 < current_pct < 100:
                    estimated_total = elapsed / (current_pct / 100)
                    remaining = estimated_total - elapsed
                    self._time_var.set(
                        f"⏱ {_fmt_time(elapsed)}  ·  ~{_fmt_time(remaining)} remaining"
                    )
                else:
                    self._time_var.set(f"⏱ {_fmt_time(elapsed)}")

        self.after(200, self._poll)

    # ----- helpers -------------------------------------------------------

    def _write_log(self, text: str, tag: str = "info") -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")
