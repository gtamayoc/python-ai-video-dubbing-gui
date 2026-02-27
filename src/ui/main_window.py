"""
src/ui/main_window.py  (v2 – pipeline mode toggle added)
──────────────────────────────────────────────────────────
The GUI is intentionally kept almost identical to v1.  The only new UI
element is a small "Pipeline Mode" segmented button in the sidebar that
lets the user switch between:

  • "Simple (fast)"   → Legacy whisper-small + Argos + edge-tts
  • "Advanced (HQ)"  → whisper-large-v3 + MarianMT + edge-tts

This setting is passed directly to ``DubbingService.run_pipeline()`` and
overrides whatever is set in ``config.yaml`` for that single run.
"""

import os
import threading
import tkinter as tk
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog

from src.services.dubbing_service import DubbingService
from src.adapters.web_adapter import download_web_audio, download_web_video
from src.adapters.media_adapter import replace_audio_in_video

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class MainWindow(ctk.CTk):
    """Main application window for AI Audio Dubbing."""

    def __init__(self) -> None:
        super().__init__()

        self.title("AI Audio Dubbing – Professional Edition")
        self.geometry("960x640")
        self.minsize(800, 560)

        # ── Layout ────────────────────────────────────────────────────────────
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, width=210, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🎙 Dubbing AI",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.logo_label.pack(pady=(24, 4))

        ctk.CTkLabel(
            self.sidebar, text="v2.0 – Advanced Pipeline",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(pady=(0, 20))

        # Input source selector
        ctk.CTkLabel(
            self.sidebar, text="Input Source",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 4))

        self.input_mode = ctk.StringVar(value="Local")
        self.radio_local = ctk.CTkRadioButton(
            self.sidebar, text="Local File",
            variable=self.input_mode, value="Local",
            command=self._toggle_mode,
        )
        self.radio_local.pack(pady=4, padx=20, anchor="w")

        self.radio_web = ctk.CTkRadioButton(
            self.sidebar, text="Web Link",
            variable=self.input_mode, value="Web",
            command=self._toggle_mode,
        )
        self.radio_web.pack(pady=4, padx=20, anchor="w")

        # ── Pipeline mode selector (NEW in v2) ────────────────────────────────
        ctk.CTkLabel(
            self.sidebar, text="Pipeline Mode",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(24, 4))

        self._pipeline_var = ctk.StringVar(value="advanced")

        self._rb_advanced = ctk.CTkRadioButton(
            self.sidebar,
            text="Advanced (HQ)",
            variable=self._pipeline_var,
            value="advanced",
            command=self._on_pipeline_change,
        )
        self._rb_advanced.pack(pady=4, padx=20, anchor="w")

        self._rb_simple = ctk.CTkRadioButton(
            self.sidebar,
            text="Simple (fast)",
            variable=self._pipeline_var,
            value="simple",
            command=self._on_pipeline_change,
        )
        self._rb_simple.pack(pady=4, padx=20, anchor="w")

        self._pipeline_desc = ctk.CTkLabel(
            self.sidebar,
            text=self._pipeline_description("advanced"),
            font=ctk.CTkFont(size=10),
            text_color="gray",
            wraplength=180,
            justify="left",
        )
        self._pipeline_desc.pack(padx=12, pady=(4, 0), anchor="w")

        # ── Content area ──────────────────────────────────────────────────────
        self.content = ctk.CTkFrame(self, corner_radius=15)
        self.content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(2, weight=1)

        # Local file input
        self.local_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.local_frame.pack(fill="x", padx=30, pady=24)

        self.file_path_var = ctk.StringVar(value="No file selected")
        self.select_btn = ctk.CTkButton(
            self.local_frame,
            text="📂  Select Media File",
            command=self._select_file,
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.select_btn.pack(pady=8)
        ctk.CTkLabel(
            self.local_frame,
            textvariable=self.file_path_var,
            font=ctk.CTkFont(size=11, slant="italic"),
        ).pack()

        # Web URL input (hidden by default)
        self.web_frame = ctk.CTkFrame(self.content, fg_color="transparent")

        self.url_entry = ctk.CTkEntry(
            self.web_frame,
            placeholder_text="Paste YouTube / TikTok / ... URL here",
            height=40,
            font=ctk.CTkFont(size=13),
        )
        self.url_entry.pack(fill="x", pady=8)

        self.web_type = ctk.StringVar(value="video")
        self.seg_web = ctk.CTkSegmentedButton(
            self.web_frame,
            values=["audio", "video"],
            variable=self.web_type,
        )
        self.seg_web.pack(pady=4)

        self.dl_btn = ctk.CTkButton(
            self.web_frame,
            text="⬇  Download Media",
            command=self._download_web,
            fg_color="#ea4335",
            hover_color="#c53929",
        )
        self.dl_btn.pack(pady=8)

        # Log area
        self.log_area = ctk.CTkTextbox(
            self.content,
            height=180,
            corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        self.log_area.pack(fill="both", expand=True, padx=30, pady=(0, 8))

        # Bottom controls
        self.output_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.output_frame.pack(fill="x", side="bottom", padx=30, pady=20)

        self.progress_bar = ctk.CTkProgressBar(self.output_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 6))

        self.status_label = ctk.CTkLabel(
            self.output_frame, text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=(0, 8))

        self.process_btn = ctk.CTkButton(
            self.output_frame,
            text="▶  TRANSLATE TO SPANISH",
            command=self._process,
            height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#34a853",
            hover_color="#2d8e47",
        )
        self.process_btn.pack(fill="x")

        # ── Service ───────────────────────────────────────────────────────────
        self._service = DubbingService()
        self._toggle_mode()

    # ── Toggle helpers ────────────────────────────────────────────────────────

    def _toggle_mode(self) -> None:
        if self.input_mode.get() == "Local":
            self.web_frame.pack_forget()
            self.local_frame.pack(fill="x", padx=30, pady=24)
        else:
            self.local_frame.pack_forget()
            self.web_frame.pack(fill="x", padx=30, pady=24)

    def _on_pipeline_change(self) -> None:
        mode = self._pipeline_var.get()
        self._pipeline_desc.configure(text=self._pipeline_description(mode))
        self._log(f"[Config] Pipeline mode set to: {mode.upper()}")

    @staticmethod
    def _pipeline_description(mode: str) -> str:
        if mode == "advanced":
            return (
                "Whisper large-v3  +  MarianMT / NLLB  +  edge-tts\n"
                "Best quality. Slower on CPU."
            )
        return (
            "Whisper small  +  Argos  +  edge-tts\n"
            "Fast. Good for testing."
        )

    # ── File selection ────────────────────────────────────────────────────────

    def _select_file(self) -> None:
        file = filedialog.askopenfilename(
            filetypes=[("Media files", "*.mp3 *.wav *.mp4 *.mkv *.avi *.flac *.m4a")]
        )
        if file:
            self.file_path_var.set(file)
            self._log(f"Selected: {os.path.basename(file)}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.log_area.insert("end", msg + "\n")
        self.log_area.see("end")

    def _update_status(self, stage: str, msg: str) -> None:
        """Progress callback passed to DubbingService."""
        # Update status label and log area from any thread safely
        def _update():
            self.status_label.configure(text=f"{stage}: {msg}")
            self._log(f"[{stage}] {msg}")
            p_map = {
                "Extracting":   0.08,
                "Diarizing":    0.18,
                "Transcribing": 0.40,
                "Translating":  0.65,
                "TTS":          0.85,
                "Saving":       0.95,
                "Completed":    1.00,
                "Validation":   1.00,
            }
            pct = p_map.get(stage)
            if pct is not None:
                self.progress_bar.set(pct)

        self.after(0, _update)

    # ── Web download ──────────────────────────────────────────────────────────

    def _download_web(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            self._log("Error: paste a URL first.")
            return

        def task() -> None:
            try:
                self.dl_btn.configure(state="disabled")
                mode = self.web_type.get()
                ext  = ".mp4" if mode == "video" else ".wav"
                out  = os.path.join(os.getcwd(), "downloaded_media" + ext)

                if mode == "audio":
                    download_web_audio(url, out, self._update_status)
                else:
                    download_web_video(url, out, self._update_status)

                self.file_path_var.set(out)
                self._log(f"Download finished: {os.path.basename(out)}")
            except Exception as e:
                self._log(f"Download error: {e}")
            finally:
                self.dl_btn.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()

    # ── Process ───────────────────────────────────────────────────────────────

    def _process(self) -> None:
        file = self.file_path_var.get()
        if file == "No file selected" or not os.path.exists(file):
            self._log("Error: select a file first.")
            return

        pipeline_mode = self._pipeline_var.get()   # "simple" or "advanced"

        def task() -> None:
            try:
                self.process_btn.configure(state="disabled")
                self.progress_bar.set(0)
                base, ext = os.path.splitext(file)
                out = base + "_es" + ext

                if ext.lower() in (".mp4", ".mkv", ".avi"):
                    # For video: dub audio separately, then merge back
                    audio_out = base + "_temp_es.wav"
                    self._service.run_pipeline(
                        file, audio_out,
                        self._update_status,
                        pipeline_mode=pipeline_mode,
                    )
                    replace_audio_in_video(file, audio_out, out, self._update_status)
                    if os.path.exists(audio_out):
                        os.remove(audio_out)
                else:
                    self._service.run_pipeline(
                        file, out,
                        self._update_status,
                        pipeline_mode=pipeline_mode,
                    )

                self._log(f"✅ Saved to: {out}")
            except Exception as e:
                self._log(f"❌ Pipeline error: {e}")
                self._update_status("Error", str(e))
            finally:
                self.process_btn.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()
