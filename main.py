"""
main.py — Entry point for the Audio Translator application.
"""
import tkinter as tk

def launch_advanced():
    root.destroy()
    from app import VideoDubberApp
    VideoDubberApp().mainloop()

def launch_simple():
    root.destroy()
    from simple_gui import SimpleDubbingApp
    SimpleDubbingApp().mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Select Application Mode")
    root.geometry("350x200")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    label = tk.Label(root, text="Elige el modo de la aplicacion:", font=("Segoe UI", 12))
    label.pack(pady=20)
    
    btn_advanced = tk.Button(root, text="Modo Avanzado (Original)", font=("Segoe UI", 11), width=25, command=launch_advanced)
    btn_advanced.pack(pady=5)
    
    btn_simple = tk.Button(root, text="Modo Simple (Nuevo)", font=("Segoe UI", 11), width=25, command=launch_simple)
    btn_simple.pack(pady=5)
    
    root.mainloop()
