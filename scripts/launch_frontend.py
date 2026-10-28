"""Run the existing local recognition service from a Windows GUI."""
from __future__ import annotations

import ctypes
import hashlib
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
WINDOW_TITLE = "护目镜识别 · 本机启动器"
LOG_PATH = ROOT / ".cache" / "frontend-launcher.log"
STARTUP_TIMEOUT = 60
BG = "#fafafa"
SURFACE = "#ffffff"
FG = "#111111"
MUTED = "#6b6b6b"
ACCENT = "#2f6feb"
DANGER = "#dc2626"


def service_command() -> list[str]:
    """Use only the project's existing interpreter and completed model."""
    interpreter = ROOT / ".venv" / "Scripts" / "python.exe"
    weights = ROOT / "results/classification/goggles_v2_control/best.pt"
    for path, description in ((interpreter, "项目 Python 环境"), (weights, "当前基线模型")):
        if not path.is_file():
            raise FileNotFoundError(f"找不到{description}：\n{path}")
    command = [
        str(interpreter), "-X", "utf8", "-u", "-m", "src.serve_demo",
        "--weights", str(weights), "--device", "0", "--port", "8765",
    ]
    candidate = ROOT / "results/classification/goggles_v4_tail4/best.pt"
    if candidate.is_file():
        command.extend(["--candidate", str(candidate)])
    return command


def acquire_instance():
    """Keep repeated shortcut clicks from starting additional services."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    identity = hashlib.sha256(str(ROOT).casefold().encode("utf-8")).hexdigest()[:20]
    handle = kernel.CreateMutexW(None, False, f"Local\\GogglesFrontend-{identity}")
    error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    if error == 183:
        user = ctypes.WinDLL("user32", use_last_error=True)
        user.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user.FindWindowW.restype = ctypes.c_void_p
        user.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        window = user.FindWindowW(None, WINDOW_TITLE)
        if window:
            user.ShowWindow(window, 9)
            user.SetForegroundWindow(window)
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)
        return None
    return kernel, handle


class FrontendLauncher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.recent_output = deque(maxlen=6)
        self.generation = 0
        self.state = "stopped"
        self.url = None
        self.started_at = 0.0
        self.closing = False
        self.stop_reason = None
        self.status = tk.StringVar(value="正在准备本机服务…")
        self.detail = tk.StringVar(value="就绪后将自动打开识别页面。")
        self.build_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.poll)
        self.root.after(150, self.start)

    def build_window(self):
        self.root.title(WINDOW_TITLE)
        self.root.geometry("576x500")
        self.root.minsize(536, 500)
        self.root.configure(background=BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure(
            "TLabel", background=BG, foreground=FG,
            font=("Microsoft YaHei UI", 11),
        )
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("TButton", font=("Microsoft YaHei UI", 11), padding=(16, 10))
        style.configure("Primary.TButton", background=ACCENT, foreground=SURFACE)
        style.map(
            "Primary.TButton", background=[("active", ACCENT)],
            foreground=[("disabled", MUTED)],
        )

        frame = ttk.Frame(self.root, padding=32)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(
            frame, text="护目镜佩戴识别", style="Title.TLabel"
        ).grid(sticky="w")
        ttk.Label(frame, text="本机运行 · 图片留在当前电脑", style="Muted.TLabel").grid(
            sticky="w", pady=(8, 24)
        )
        self.status_label = ttk.Label(
            frame, textvariable=self.status, style="Status.TLabel"
        )
        self.status_label.grid(sticky="w")
        ttk.Label(
            frame, textvariable=self.detail, style="Muted.TLabel", wraplength=480
        ).grid(sticky="w", pady=(8, 24))
        buttons = ttk.Frame(frame)
        buttons.grid(sticky="w")
        self.open_button = ttk.Button(
            buttons, text="打开识别页面", command=self.open_page,
            style="Primary.TButton", state="disabled",
        )
        self.open_button.pack(side="left", padx=(0, 12))
        self.service_button = ttk.Button(
            buttons, text="启动服务", command=self.toggle_service
        )
        self.service_button.pack(side="left")
        ttk.Label(
            frame, text="保留此窗口即可持续识别；关闭窗口会停止本次服务。",
            style="Muted.TLabel", wraplength=480,
        ).grid(sticky="w", pady=(24, 8))
        ttk.Button(frame, text="查看运行日志", command=self.open_log).grid(sticky="w")

    def set_state(self, state: str, status: str, detail: str):
        self.state = state
        self.status.set(status)
        self.detail.set(detail)
        self.status_label.configure(
            foreground=DANGER if state == "failed" else FG
        )
        self.open_button.configure(
            state="normal" if state == "running" else "disabled"
        )
        self.service_button.configure(
            text="停止服务" if state in {"starting", "running", "stopping"} else "重新启动",
            state="disabled" if state == "stopping" else "normal",
        )

    def start(self):
        if self.process is not None or self.closing:
            return
        self.generation += 1
        generation = self.generation
        self.url = None
        self.stop_reason = None
        self.recent_output.clear()
        try:
            command = service_command()
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as log:
                timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                log.write(f"\n=== {timestamp} ===\n")
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, ValueError) as error:
            self.set_state("failed", "服务未能启动", str(error))
            return
        self.started_at = time.monotonic()
        self.set_state(
            "starting", "正在启动本机服务…", "正在读取现有模型记录，请稍候。"
        )
        threading.Thread(
            target=self.read_output, args=(self.process, generation), daemon=True
        ).start()

    def read_output(self, process, generation: int):
        try:
            with LOG_PATH.open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.events.put((generation, "line", line.rstrip()))
        except OSError as error:
            self.events.put((generation, "log_error", str(error)))
            # Keep draining the pipe even if the log cannot be written.
            for line in process.stdout:
                self.events.put((generation, "line", line.rstrip()))
        finally:
            process.stdout.close()
            self.events.put((generation, "exit", process.wait()))

    def poll(self):
        while True:
            try:
                generation, kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if generation != self.generation:
                continue
            if kind == "line":
                self.recent_output.append(value)
                prefix = "Local goggles demo: "
                if value.startswith(prefix) and self.state == "starting":
                    self.url = value.removeprefix(prefix)
                    self.set_state(
                        "running", "本机服务已就绪",
                        "页面已在浏览器打开，可上传单人头像进行识别。",
                    )
                    self.open_page()
            elif kind == "log_error":
                self.detail.set("运行日志暂时无法写入。可继续使用页面；重启前请检查磁盘空间。")
            elif kind == "exit":
                self.process = None
                self.url = None
                if self.state == "stopping" and not self.stop_reason:
                    self.set_state(
                        "stopped", "服务已停止", "点击“重新启动”即可继续使用。"
                    )
                else:
                    detail = self.stop_reason or next(
                        (line for line in reversed(self.recent_output) if line),
                        "请查看运行日志了解原因。",
                    )
                    if "10048" in "\n".join(self.recent_output):
                        detail = "本机端口8765已被占用。请先关闭之前的服务，再点“重新启动”。"
                    self.set_state("failed", "服务未能继续运行", detail)
        if self.closing and self.process is None:
            self.root.destroy()
            return
        elapsed = time.monotonic() - self.started_at
        if self.state == "starting" and elapsed > STARTUP_TIMEOUT:
            self.stop_reason = "启动超过60秒，已停止本次服务。请查看日志后重试。"
            self.stop()
        self.root.after(100, self.poll)

    def toggle_service(self):
        if self.state in {"starting", "running"}:
            self.stop()
        elif self.state != "stopping":
            self.start()

    def stop(self):
        if self.process is None or self.state == "stopping":
            return
        self.set_state(
            "stopping", "正在停止本次服务…",
            "停止后，浏览器中的页面将无法继续识别。",
        )
        threading.Thread(
            target=self.terminate, args=(self.process,), daemon=True
        ).start()

    @staticmethod
    def terminate(process):
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        except OSError:
            # The output reader observes an independently exited child as well.
            pass

    def open_page(self):
        if not self.url or self.state != "running":
            return
        try:
            opened = webbrowser.open(self.url)
        except webbrowser.Error:
            opened = False
        if not opened:
            self.detail.set(f"浏览器未能自动打开。请复制此地址：{self.url}")

    def open_log(self):
        if not LOG_PATH.is_file():
            messagebox.showinfo("运行日志", "本次还没有生成运行日志。", parent=self.root)
            return
        try:
            os.startfile(LOG_PATH)
        except OSError as error:
            messagebox.showerror("无法打开日志", str(error), parent=self.root)

    def close(self):
        self.closing = True
        if self.process is None:
            self.root.destroy()
        else:
            self.stop()


def main():
    if sys.platform != "win32":
        raise SystemExit("图形启动器仅支持 Windows；其他系统请使用现有本机服务入口。")
    instance = acquire_instance()
    if instance is None:
        return
    kernel, handle = instance
    launcher = None
    try:
        root = tk.Tk()
        launcher = FrontendLauncher(root)
        root.mainloop()
    finally:
        if launcher is not None and launcher.process is not None:
            launcher.terminate(launcher.process)
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    main()
