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
    def __init__(self):
        super().__init__()

        self.title("AI Audio Dubbing - Professional Edition")
        self.geometry("900x600")

        # Main Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Dubbing AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(pady=(20, 10))
        
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Input Source", font=ctk.CTkFont(size=14))
        self.mode_label.pack(pady=(20, 5))
        
        self.input_mode = ctk.StringVar(value="Local")
        self.radio_local = ctk.CTkRadioButton(self.sidebar, text="Local File", variable=self.input_mode, value="Local", command=self._toggle_mode)
        self.radio_local.pack(pady=5, padx=20, anchor="w")
        self.radio_web = ctk.CTkRadioButton(self.sidebar, text="Web Link", variable=self.input_mode, value="Web", command=self._toggle_mode)
        self.radio_web.pack(pady=5, padx=20, anchor="w")

        # Content Area
        self.content = ctk.CTkFrame(self, corner_radius=15)
        self.content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)

        # Local File Input Container
        self.local_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.local_frame.pack(fill="x", padx=30, pady=30)
        
        self.file_path_var = ctk.StringVar(value="No file selected")
        self.select_btn = ctk.CTkButton(self.local_frame, text="Select Media File", command=self._select_file, height=40, font=ctk.CTkFont(weight="bold"))
        self.select_btn.pack(pady=10)
        self.path_label = ctk.CTkLabel(self.local_frame, textvariable=self.file_path_var, font=ctk.CTkFont(slant="italic"))
        self.path_label.pack()

        # Web Input Container (Hidden by default)
        self.web_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        
        self.url_entry = ctk.CTkEntry(self.web_frame, placeholder_text="Paste YouTube/TikTok URL here...", height=40, font=ctk.CTkFont(size=14))
        self.url_entry.pack(fill="x", pady=10)
        
        self.web_type = ctk.StringVar(value="audio")
        self.seg_web = ctk.CTkSegmentedButton(self.web_frame, values=["audio", "video"], variable=self.web_type)
        self.seg_web.pack(pady=5)
        
        self.dl_btn = ctk.CTkButton(self.web_frame, text="Download Media", command=self._download_web, fg_color="#ea4335", hover_color="#c53929")
        self.dl_btn.pack(pady=10)

        # Bottom Controls
        self.output_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.output_frame.pack(fill="x", side="bottom", padx=30, pady=30)
        
        self.process_btn = ctk.CTkButton(self.output_frame, text="TRANSLATE TO SPANISH", command=self._process, height=50, font=ctk.CTkFont(size=16, weight="bold"), fg_color="#34a853", hover_color="#2d8e47")
        self.process_btn.pack(fill="x", pady=(0, 20))
        
        self.progress_bar = ctk.CTkProgressBar(self.output_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=5)
        
        self.status_label = ctk.CTkLabel(self.output_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.status_label.pack()

        self.log_area = ctk.CTkTextbox(self.content, height=150, corner_radius=10, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_area.pack(fill="both", expand=True, padx=30, pady=10)

        self._service = DubbingService()
        self._toggle_mode()

    def _toggle_mode(self):
        if self.input_mode.get() == "Local":
            self.web_frame.pack_forget()
            self.local_frame.pack(fill="x", padx=30, pady=30)
        else:
            self.local_frame.pack_forget()
            self.web_frame.pack(fill="x", padx=30, pady=30)

    def _select_file(self):
        file = filedialog.askopenfilename(filetypes=[("Media files", "*.mp3 *.wav *.mp4 *.mkv *.avi")])
        if file:
            self.file_path_var.set(file)
            self._log(f"Selected: {os.path.basename(file)}")

    def _log(self, msg: str):
        self.log_area.insert("end", msg + "\n")
        self.log_area.see("end")

    def _update_status(self, stage: str, msg: str):
        self.status_label.configure(text=f"{stage}: {msg}")
        self._log(f"[{stage}] {msg}")
        # Fake progress update for now
        p_map = {"Extracting": 0.1, "Diarizing": 0.3, "Transcribing": 0.5, "Translating": 0.7, "Cloning": 0.9, "Completed": 1.0}
        if stage in p_map:
            self.progress_bar.set(p_map[stage])

    def _download_web(self):
        url = self.url_entry.get().strip()
        if not url: return

        def task():
            try:
                self.dl_btn.configure(state="disabled")
                ext = ".mp4" if self.web_type.get() == "video" else ".wav"
                out = os.path.join(os.getcwd(), "downloaded_media" + ext)
                
                if self.web_type.get() == "audio":
                    download_web_audio(url, out, self._update_status)
                else:
                    download_web_video(url, out, self._update_status)
                
                self.file_path_var.set(out)
                self._log("Download finished.")
            except Exception as e:
                self._log(f"Error: {e}")
            finally:
                self.dl_btn.configure(state="normal")
        
        threading.Thread(target=task, daemon=True).start()

    def _process(self):
        file = self.file_path_var.get()
        if file == "No file selected" or not os.path.exists(file):
            self._log("Error: Select a file first.")
            return

        def task():
            try:
                self.process_btn.configure(state="disabled")
                base, ext = os.path.splitext(file)
                out = base + "_es" + ext
                
                # If it's a video, we first dub the audio then merge
                if ext.lower() == ".mp4":
                    audio_out = base + "_temp_es.wav"
                    self._service.run_pipeline(file, audio_out, self._update_status)
                    replace_audio_in_video(file, audio_out, out, self._update_status)
                    if os.path.exists(audio_out): os.remove(audio_out)
                else:
                    self._service.run_pipeline(file, out, self._update_status)
                
                self._log(f"Saved to: {out}")
            except Exception as e:
                self._log(f"Error: {e}")
            finally:
                self.process_btn.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()
