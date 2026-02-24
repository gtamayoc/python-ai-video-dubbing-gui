"""
app.py
======
Redesigned CustomTkinter GUI for English → Spanish audio translator.
Features: Modern dark theme, Pipeline stepper, Drag-and-drop area, Tabbed logs.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import math
import tkinter as tk
from tkinter import filedialog, scrolledtext
from datetime import timedelta

import customtkinter as ctk
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip

from PIL import Image, ImageTk

from translator_service import TranslatorService

# -- Theme & colors ---------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_COLOR = "#0B0F19"      # Deep matte dark
CARD_COLOR = "#161B22"    # GitHub-like card dark
ACCENT_COLOR = "#3B82F6"  # Electric blue
SUCCESS_COLOR = "#10B981" # Emerald green
ERROR_COLOR = "#EF4444"   # Red-rose
TEXT_MAIN = "#F8FAFC"
TEXT_DIM = "#94A3B8"
BORDER_COLOR = "#30363D"

# -- Pipeline configuration ------------------------------------------------
STAGE_MAP = {
    "Extracting": 0,
    "Diarizing": 1,
    "Transcribing": 2,
    "Translating": 2,
    "Cloning Voice": 3,
    "Saving": 4,
    "Completed": 4
}

UI_STAGES = [
    {"label": "EXTRACT", "icon": "📤"},
    {"label": "DIARIZE", "icon": "👥"},
    {"label": "TRANSCRIBE", "icon": "📝"},
    {"label": "TTS", "icon": "🗣️"},
    {"label": "EXPORT", "icon": "💾"},
]

def format_size(bytes_size: int) -> str:
    if bytes_size == 0: return "0 B"
    s = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(bytes_size, 1024)))
    p = math.pow(1024, i)
    r = round(bytes_size / p, 2)
    return f"{r} {s[i]}"

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))[2:7]

# ---------------------------------------------------------------------------
# Custom Widgets
# ---------------------------------------------------------------------------

class PipelineStepper(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.step_widgets = []
        self.lines = []
        self.grid_columnconfigure(list(range(len(UI_STAGES)*2 - 1)), weight=1)
        
        for i, stage in enumerate(UI_STAGES):
            container = ctk.CTkFrame(self, fg_color="transparent")
            container.grid(row=0, column=i*2, sticky="nsew")
            circle = ctk.CTkLabel(container, text=stage["icon"], width=36, height=36, corner_radius=18, fg_color=BG_COLOR, text_color=TEXT_DIM, font=("Segoe UI", 14))
            circle.pack(pady=(0, 5))
            label = ctk.CTkLabel(container, text=stage["label"], font=("Segoe UI Bold", 9), text_color=TEXT_DIM)
            label.pack()
            self.step_widgets.append({"circle": circle, "label": label, "icon": stage["icon"]})
            if i < len(UI_STAGES) - 1:
                line = ctk.CTkFrame(self, height=2, fg_color=BORDER_COLOR)
                line.grid(row=0, column=i*2 + 1, sticky="ew", pady=(0, 22), padx=2)
                self.lines.append(line)

    def update_stage(self, current_idx, status="pending"):
        for i, widgets in enumerate(self.step_widgets):
            if i < current_idx:
                widgets["circle"].configure(fg_color=SUCCESS_COLOR, text_color="white", text="✓")
                widgets["label"].configure(text_color=SUCCESS_COLOR)
                if i < len(self.lines): self.lines[i].configure(fg_color=SUCCESS_COLOR)
            elif i == current_idx:
                if status == "running":
                    widgets["circle"].configure(fg_color=ACCENT_COLOR, text_color="white", text=widgets["icon"])
                    widgets["label"].configure(text_color=ACCENT_COLOR)
                    if i < len(self.lines): self.lines[i].configure(fg_color=BORDER_COLOR)
                elif status == "error":
                    widgets["circle"].configure(fg_color=ERROR_COLOR, text_color="white", text="✕")
                    widgets["label"].configure(text_color=ERROR_COLOR)
                elif status == "done":
                    widgets["circle"].configure(fg_color=SUCCESS_COLOR, text_color="white", text="✓")
                    widgets["label"].configure(text_color=SUCCESS_COLOR)
                    if i < len(self.lines): self.lines[i].configure(fg_color=SUCCESS_COLOR)
            else:
                widgets["circle"].configure(fg_color=BG_COLOR, text_color=TEXT_DIM, text=widgets["icon"])
                widgets["label"].configure(text_color=TEXT_DIM)
                if i < len(self.lines): self.lines[i].configure(fg_color=BORDER_COLOR)

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class VideoDubberApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        # Inject tkinterdnd2 DnD support into the CTk root without changing base class.
        # TkinterDnD.Tk uses Tk.tk.call('package', 'require', 'tkdnd') internally;
        # we replicate that here so CTk widgets can register as drop targets.
        if HAS_DND:
            try:
                TkinterDnD._require(self)
            except Exception:
                pass  # DnD unavailable at runtime; fall back silently
        self.title("AI Audio Translator")
        self.geometry("960x720")
        self.minsize(800, 600)
        self.configure(fg_color=BG_COLOR)

        self._input_path: str | None = None
        self._output_path: str | None = None
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._service = TranslatorService()
        self._worker: threading.Thread | None = None
        self._start_time: float = 0
        self._current_stage_idx: int = 0
        self._is_running: bool = False

        self._build_ui()
        if HAS_DND: self._setup_dnd()

    def _build_ui(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=30, pady=20)

        # -- App Bar --
        self.header = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.header.pack(fill="x", pady=(0, 20))
        title_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        title_frame.pack(side="left")
        ctk.CTkLabel(title_frame, text="AI Audio Translator", font=("Segoe UI Bold", 26), text_color=TEXT_MAIN).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="English → Spanish", font=("Segoe UI", 15), text_color=ACCENT_COLOR).pack(anchor="w")

        # -- Body --
        self.body = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.body.pack(fill="both", expand=True)
        self.body.grid_columnconfigure(0, weight=4)
        self.body.grid_columnconfigure(1, weight=5)
        self.body.grid_rowconfigure(0, weight=1)

        # -- Left Column --
        self.left_col = ctk.CTkFrame(self.body, fg_color="transparent")
        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # Source Audio Card
        self.source_card = ctk.CTkFrame(self.left_col, fg_color=CARD_COLOR, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.source_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(self.source_card, text="SOURCE AUDIO", font=("Segoe UI Bold", 11), text_color=TEXT_DIM).pack(anchor="w", padx=20, pady=(20, 10))
        
        self.drop_area = ctk.CTkFrame(self.source_card, fg_color=BG_COLOR, height=160, corner_radius=10, border_width=2, border_color=BORDER_COLOR)
        self.drop_area.pack(fill="x", padx=20, pady=(0, 20))
        self.drop_area.pack_propagate(False)
        self.drop_label = ctk.CTkLabel(self.drop_area, text="📁\nDrop or select audio file\nMP3, WAV, FLAC", font=("Segoe UI", 13), text_color=TEXT_DIM)
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")
        self.drop_area.bind("<Button-1>", lambda e: self._pick_file())

        # Output Settings Card
        self.output_card = ctk.CTkFrame(self.left_col, fg_color=CARD_COLOR, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.output_card.pack(fill="x")
        ctk.CTkLabel(self.output_card, text="OUTPUT SETTINGS", font=("Segoe UI Bold", 11), text_color=TEXT_DIM).pack(anchor="w", padx=20, pady=(20, 10))
        
        self.out_path_var = tk.StringVar(value="/Select destination...")
        path_row = ctk.CTkFrame(self.output_card, fg_color="transparent")
        path_row.pack(fill="x", padx=20, pady=(0, 20))
        self.out_entry = ctk.CTkEntry(path_row, textvariable=self.out_path_var, font=("Segoe UI", 12), height=38, fg_color=BG_COLOR, border_color=BORDER_COLOR)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.browse_btn = ctk.CTkButton(path_row, text="Browse", width=85, height=38, fg_color="#30363D", hover_color=ACCENT_COLOR, command=self._pick_output)
        self.browse_btn.pack(side="right")

        # Threads Setting
        threads_row = ctk.CTkFrame(self.output_card, fg_color="transparent")
        threads_row.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(threads_row, text="Max CPU Threads (TTS)", font=("Segoe UI", 12), text_color=TEXT_DIM).pack(side="left")
        
        self.threads_var = tk.IntVar(value=10)
        self.threads_lbl = ctk.CTkLabel(threads_row, text="10", font=("Segoe UI Bold", 12), text_color=ACCENT_COLOR, width=25)
        self.threads_lbl.pack(side="right", padx=(10, 0))
        
        self.threads_slider = ctk.CTkSlider(
            threads_row, from_=1, to=20, variable=self.threads_var, number_of_steps=19,
            command=lambda v: self.threads_lbl.configure(text=str(int(v)))
        )
        self.threads_slider.pack(side="right", fill="x", expand=True, padx=(15, 0))

        # -- Right Column --
        self.right_col = ctk.CTkFrame(self.body, fg_color="transparent")
        self.right_col.grid(row=0, column=1, sticky="nsew")

        # Pipeline Card
        self.status_card = ctk.CTkFrame(self.right_col, fg_color=CARD_COLOR, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.status_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(self.status_card, text="PIPELINE STATUS", font=("Segoe UI Bold", 11), text_color=TEXT_DIM).pack(anchor="w", padx=20, pady=(20, 20))
        self.stepper = PipelineStepper(self.status_card)
        self.stepper.pack(fill="x", padx=25, pady=(0, 25))
        
        prog_row = ctk.CTkFrame(self.status_card, fg_color="transparent")
        prog_row.pack(fill="x", padx=25, pady=(0, 8))
        self.stage_text_var = tk.StringVar(value="Idle")
        ctk.CTkLabel(prog_row, textvariable=self.stage_text_var, font=("Segoe UI", 12), text_color=TEXT_MAIN).pack(side="left")
        self.pct_text_var = tk.StringVar(value="0%")
        ctk.CTkLabel(prog_row, textvariable=self.pct_text_var, font=("Segoe UI Bold", 15), text_color=ACCENT_COLOR).pack(side="right")
        
        self.progress_bar = ctk.CTkProgressBar(self.status_card, height=10, fg_color=BG_COLOR, progress_color=ACCENT_COLOR)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=25, pady=(0, 25))

        # Logs Tabs
        self.logs_card = ctk.CTkFrame(self.right_col, fg_color=CARD_COLOR, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.logs_card.pack(fill="both", expand=True)
        self.log_tabs = ctk.CTkTabview(self.logs_card, fg_color="transparent")
        self.log_tabs.pack(fill="both", expand=True, padx=5, pady=5)
        self.tab_overview = self.log_tabs.add("OVERVIEW")
        self.tab_detailed = self.log_tabs.add("DETAILED LOG")

        self.ov_frame = ctk.CTkFrame(self.tab_overview, fg_color="transparent")
        self.ov_frame.pack(fill="both", expand=True, padx=25, pady=15)
        self._add_stat("Audio Duration:", "duration")
        self._add_stat("Total Run Time:", "total_time")
        self._add_stat("Estimated Remaining:", "remaining")
        self._add_stat("Realtime Factor:", "rt_factor")
        self._add_stat("Processing Mode:", "engine")
        self.stat_engine.set("GPU Accelerated (V3)")

        self.log_text = scrolledtext.ScrolledText(self.tab_detailed, wrap="word", state="disabled", font=("Consolas", 10), bg=BG_COLOR, fg=TEXT_MAIN, borderwidth=0)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.tag_configure("info", foreground=TEXT_DIM)
        self.log_text.tag_configure("stage", foreground=ACCENT_COLOR)

        self.log_text.tag_configure("ok", foreground=SUCCESS_COLOR)
        self.log_text.tag_configure("err", foreground=ERROR_COLOR)

        # Bottom Bar
        self.action_bar = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.action_bar.pack(fill="x", pady=(20, 0))
        self.time_label_var = tk.StringVar(value="")
        ctk.CTkLabel(self.action_bar, textvariable=self.time_label_var, font=("Segoe UI", 12), text_color=TEXT_DIM).pack(side="left")
        
        self.main_btn = ctk.CTkButton(self.action_bar, text="Translate to Spanish →", font=("Segoe UI Bold", 16), height=52, width=240, fg_color=ACCENT_COLOR, hover_color="#2563EB", command=self._start_pipeline)
        self.main_btn.pack(side="right")
        
        self.remaining_label_var = tk.StringVar(value="")
        ctk.CTkLabel(self.action_bar, textvariable=self.remaining_label_var, font=("Segoe UI Bold", 12), text_color=ACCENT_COLOR).pack(side="right", padx=20)

        self.success_banner = ctk.CTkFrame(self, fg_color=SUCCESS_COLOR, height=55, corner_radius=0)
        ctk.CTkLabel(self.success_banner, text="Translation Ready! 🎉", font=("Segoe UI Bold", 14), text_color="white").pack(side="left", padx=25)
        ctk.CTkButton(self.success_banner, text="Open Folder", width=120, height=32, fg_color="white", text_color=SUCCESS_COLOR, font=("Segoe UI Bold", 12), command=self._open_output_folder).pack(side="right", padx=25)

    def _add_stat(self, label, var_name):
        row = ctk.CTkFrame(self.ov_frame, fg_color="transparent")
        row.pack(fill="x", pady=6)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 13), text_color=TEXT_DIM).pack(side="left")
        var = tk.StringVar(value="--")
        setattr(self, f"stat_{var_name}", var)
        ctk.CTkLabel(row, textvariable=var, font=("Segoe UI Bold", 13), text_color=TEXT_MAIN).pack(side="right")

    def _setup_dnd(self):
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', lambda e: self._load_file(os.path.normpath(e.data.strip('{}'))))

    def _pick_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"), ("All", "*.*")])
        if path: self._load_file(os.path.normpath(path))

    def _load_file(self, path):
        self._input_path = path
        base, ext = os.path.splitext(path)
        self._output_path = f"{base}_es{ext}"
        self.out_path_var.set(self._output_path)
        size = format_size(os.path.getsize(path))
        dur_str = "--:--"
        try:
            with AudioFileClip(path) as clip:
                dur_raw = clip.duration
                dur_str = format_time(dur_raw)
                self.stat_duration.set(f"{dur_str} ({int(dur_raw)}s)")
        except: pass
        self.drop_label.configure(text=f"📄 {os.path.basename(path)}\n{dur_str} · {size}", text_color=TEXT_MAIN)
        self._write_log(f"Selected: {path}", "info")
        self.success_banner.place_forget()

    def _pick_output(self):
        path = filedialog.asksaveasfilename(initialfile=os.path.basename(self._output_path or ""), defaultextension=".wav")
        if path:
            self._output_path = os.path.normpath(path)
            self.out_path_var.set(self._output_path)

    def _start_pipeline(self):
        if not self._input_path: return self._write_log("⚠ Select a file first.", "err")
        self._is_running = True
        self.main_btn.configure(text="Processing...", state="disabled", fg_color="#343B42")
        self.success_banner.place_forget()
        self.progress_bar.set(0)
        self.pct_text_var.set("0%")
        self.remaining_label_var.set("")
        self.stat_remaining.set("Calculating...")
        self._current_stage_idx = 0
        self._start_time = time.time()
        self.stepper.update_stage(0, "running")
        self._worker = threading.Thread(target=self._run_bg, daemon=True)
        self._worker.start()
        self._poll()

    def _run_bg(self):
        try:
            self._service.run_pipeline(
                self._input_path, 
                self._output_path, 
                progress_callback=self._cb,
                max_threads=int(self.threads_var.get())
            )
            self._queue.put(("__done__", ""))
        except Exception as exc:
            self._queue.put(("__error__", str(exc)))

    def _cb(self, stage: str, msg: str): self._queue.put((stage, msg))

    def _poll(self):
        try:
            while True:
                stage, msg = self._queue.get_nowait()
                if stage == "__done__": return self._on_complete()
                if stage == "__error__": return self._on_error(msg)
                
                ui_idx = STAGE_MAP.get(stage, self._current_stage_idx)
                if ui_idx > self._current_stage_idx:
                    self.stepper.update_stage(ui_idx, "running")
                    self._current_stage_idx = ui_idx
                
                progress = min(0.95, (ui_idx * 0.2) + 0.1)
                self.progress_bar.set(progress)
                self.pct_text_var.set(f"{int(progress*100)}%")
                self.stage_text_var.set(f"{stage}: {msg}")
                self._write_log(f"[{stage}] {msg}", "stage")
        except queue.Empty: pass

        if self._is_running:
            elapsed = time.time() - self._start_time
            self.time_label_var.set(f"Elapsed: {format_time(elapsed)}")
            self.stat_total_time.set(f"{int(elapsed)}s")

            # 1. Remaining time calculation
            current_prog = self.progress_bar.get()
            if current_prog > 0.05:
                total_est = elapsed / current_prog
                remaining = max(0, total_est - elapsed)
                rem_str = format_time(remaining)
                self.remaining_label_var.set(f"Remaining: ~{rem_str}")
                self.stat_remaining.set(f"{int(remaining)}s approx.")
            else:
                self.remaining_label_var.set("Estimating...")
                self.stat_remaining.set("Estimating...")

            # 2. Realtime factor calculation
            try:
                dur_text = self.stat_duration.get()
                if "(" in dur_text:
                    dur_raw = float(dur_text.split("(")[1].split("s")[0])
                    if elapsed > 0:
                        rtf = elapsed / dur_raw
                        self.stat_rt_factor.set(f"{rtf:.2f}x")
            except:
                pass

            self.after(200, self._poll)

    def _on_complete(self):
        self._is_running = False
        self.progress_bar.set(1.0)
        self.pct_text_var.set("100%")
        self.stage_text_var.set("Pipeline completed")
        self.remaining_label_var.set("Done")
        self.stat_remaining.set("0s")
        self.stepper.update_stage(4, "done")
        self.main_btn.configure(text="Translate to Spanish →", state="normal", fg_color=ACCENT_COLOR)
        self.success_banner.place(relx=0, rely=1, relwidth=1, y=-55)
        self._write_log("Pipeline finished!", "ok")

    def _on_error(self, msg):
        self._is_running = False
        self.stepper.update_stage(self._current_stage_idx, "error")
        self.main_btn.configure(text="Try Again", state="normal", fg_color=ERROR_COLOR)
        self._write_log(f"❌ {msg}", "err")

    def _open_output_folder(self):
        if self._output_path: os.startfile(os.path.dirname(self._output_path))

    def _write_log(self, text, tag="info"):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f" {text}\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

if __name__ == "__main__":
    VideoDubberApp().mainloop()
