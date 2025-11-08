# ZaiMenu.pyw
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# --- Shortcut paths (edit labels or add more later) ---
SHORTCUTS = [
    ("Open: gui_app", r"D:\sample SLICES  code 0.1\Menu Zai Apps Folder\gui_app - Shortcut (2).lnk"),
    ("Open: ZaiColorTagger_tk", r"D:\sample SLICES  code 0.1\Menu Zai Apps Folder\ZaiColorTagger_tk - Shortcut (2).lnk"),
]

def launch_shortcut(path_str: str):
    p = Path(path_str)
    if not p.exists():
        messagebox.showerror("Not found", f"Shortcut not found:\n{p}")
        return
    try:
        # On Windows, this will follow .lnk and launch the target
        os.startfile(str(p))
    except OSError as e:
        messagebox.showerror("Launch failed", f"Couldn't open:\n{p}\n\n{e}")

def build_ui(root: tk.Tk):
    root.title("ZAI — App Menu")
    root.geometry("420x220")
    root.minsize(360, 180)

    # A little padding frame
    wrap = ttk.Frame(root, padding=20)
    wrap.pack(fill="both", expand=True)

    title = ttk.Label(wrap, text="ZAI — App Menu", font=("Segoe UI", 16, "bold"))
    title.pack(pady=(0, 10))

    desc = ttk.Label(
        wrap,
        text="Click an app to open it.",
        font=("Segoe UI", 10)
    )
    desc.pack(pady=(0, 12))

    # Buttons
    for label, path in SHORTCUTS:
        btn = ttk.Button(wrap, text=label, command=lambda p=path: launch_shortcut(p))
        btn.pack(fill="x", pady=6)

    # Footer
    foot = ttk.Label(wrap, text="You can close this window after launching.", foreground="#666")
    foot.pack(pady=(12, 0))

if __name__ == "__main__":
    root = tk.Tk()
    # Use native ttk theme if available
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "xpnative" in style.theme_names():
            style.theme_use("xpnative")
    except Exception:
        pass
    build_ui(root)
    root.mainloop()
