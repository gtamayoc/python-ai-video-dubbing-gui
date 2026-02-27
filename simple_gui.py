import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

from translator_service import (
    TranslatorService, models, 
    download_web_audio, download_web_video, replace_audio_in_video
)

class SimpleDubbingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Audio Translator (Simple Mode)")
        self.geometry("600x550")
        self.resizable(False, False)

        self._service = TranslatorService()
        
        self.input_mode = tk.StringVar(value="local")
        self.web_type = tk.StringVar(value="audio")
        self.local_file_path = tk.StringVar(value="")
        self.web_url = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        # Mode Selection
        mode_frame = tk.LabelFrame(self, text="Input Mode", padx=10, pady=10)
        mode_frame.pack(fill="x", padx=15, pady=10)

        tk.Radiobutton(mode_frame, text="Local file", variable=self.input_mode, value="local", command=self._toggle_mode).pack(side="left", padx=10)
        tk.Radiobutton(mode_frame, text="Web link", variable=self.input_mode, value="web", command=self._toggle_mode).pack(side="left", padx=10)

        # Local Input Container
        self.local_frame = tk.Frame(self)
        self.local_frame.pack(fill="x", padx=15, pady=5)
        
        tk.Button(self.local_frame, text="Select file", command=self._select_file).pack(side="left", padx=(0, 10))
        tk.Label(self.local_frame, textvariable=self.local_file_path, fg="blue").pack(side="left", fill="x", expand=True)

        # Web Input Container
        self.web_frame = tk.Frame(self)
        
        url_subframe = tk.Frame(self.web_frame)
        url_subframe.pack(fill="x", pady=5)
        tk.Label(url_subframe, text="Paste URL:").pack(side="left")
        tk.Entry(url_subframe, textvariable=self.web_url).pack(side="left", fill="x", expand=True, padx=5)
        
        type_subframe = tk.Frame(self.web_frame)
        type_subframe.pack(fill="x", pady=5)
        tk.Label(type_subframe, text="Download as:").pack(side="left")
        tk.Radiobutton(type_subframe, text="Audio Only", variable=self.web_type, value="audio").pack(side="left", padx=5)
        tk.Radiobutton(type_subframe, text="Video (Dubbed)", variable=self.web_type, value="video").pack(side="left", padx=5)

        tk.Button(self.web_frame, text="Download & extract media", command=self._download_web).pack(anchor="w", pady=5)

        # Output Container
        out_frame = tk.LabelFrame(self, text="Output Path", padx=10, pady=10)
        out_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Entry(out_frame, textvariable=self.output_path, state="readonly").pack(side="left", fill="x", expand=True, padx=5)

        # Process Button
        self.process_btn = tk.Button(self, text="Process to Spanish", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=self._process)
        self.process_btn.pack(pady=15, ipadx=10, ipady=5)

        # Log
        log_frame = tk.LabelFrame(self, text="Progress Log", padx=10, pady=10)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", state="disabled", height=10)
        self.log_text.pack(fill="both", expand=True)

        # Initial toggle
        self._toggle_mode()

    def _toggle_mode(self):
        if self.input_mode.get() == "local":
            self.web_frame.pack_forget()
            self.local_frame.pack(fill="x", padx=15, pady=5, after=self.children['!labelframe'])
        else:
            self.local_frame.pack_forget()
            self.web_frame.pack(fill="x", padx=15, pady=5, after=self.children['!labelframe'])

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Media files", "*.wav *.mp3 *.flac *.mp4 *.mkv"), ("All", "*.*")])
        if path:
            path = os.path.normpath(path)
            self.local_file_path.set(path)
            base, ext = os.path.splitext(path)
            self.output_path.set(f"{base}_es{ext}")
            self._log(f"Selected local file: {path}")

    def _download_web(self):
        url = self.web_url.get().strip()
        if not url:
            self._log("Error: Please paste a URL first.")
            return

        dl_type = self.web_type.get()
        self.process_btn.configure(state="disabled")
        
        def worker():
            self._log("Downloading media...")
            try:
                base_name = "downloaded_media"
                
                if dl_type == "audio":
                    out_path = os.path.join(os.getcwd(), f"{base_name}.wav")
                    download_web_audio(url, out_path, lambda s, m: self._log(f"[{s}] {m}"))
                    self.local_file_path.set(out_path)
                    self.output_path.set(os.path.join(os.getcwd(), f"{base_name}_es.wav"))
                else: # video
                    out_path = os.path.join(os.getcwd(), f"{base_name}.mp4")
                    download_web_video(url, out_path, lambda s, m: self._log(f"[{s}] {m}"))
                    self.local_file_path.set(out_path)
                    self.output_path.set(os.path.join(os.getcwd(), f"{base_name}_es.mp4"))
                    
                self._log("Download complete!")
                self.process_btn.configure(state="normal")
            except Exception as e:
                self._log(f"Error downloading: {e}")
                self.process_btn.configure(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    def _process(self):
        input_path = self.local_file_path.get()
        output_path = self.output_path.get()
        
        if not input_path or not os.path.exists(input_path):
            self._log("Error: Valid input file required.")
            return

        self.process_btn.configure(state="disabled")
        
        def worker():
            try:
                self._log("Preloading models...")
                models.preload_all()
                self._log("Models ready.")

                # If it's a video file, we need a temp audio output, then we merge.
                is_video = output_path.lower().endswith(".mp4")
                
                if is_video:
                    temp_audio_out = input_path + "_temp_es.wav"
                    self._log("Starting audio extraction and dubbing pipeline...")
                    self._service.run_pipeline(
                        input_path, 
                        temp_audio_out, 
                        progress_callback=lambda s, m: self._log(f"[{s}] {m}")
                    )
                    replace_audio_in_video(input_path, temp_audio_out, output_path, lambda s, m: self._log(f"[{s}] {m}"))
                    if os.path.exists(temp_audio_out):
                        os.remove(temp_audio_out)
                else:
                    self._log("Starting audio dubbing pipeline...")
                    self._service.run_pipeline(
                        input_path, 
                        output_path, 
                        progress_callback=lambda s, m: self._log(f"[{s}] {m}")
                    )

                self._log("Completed!")
                messagebox.showinfo("Success", "Processing Complete!")
            except Exception as e:
                self._log(f"Pipeline Error: {e}")
            finally:
                self.process_btn.configure(state="normal")

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    app = SimpleDubbingApp()
    app.mainloop()
