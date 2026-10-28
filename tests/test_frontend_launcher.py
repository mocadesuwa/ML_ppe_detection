"""Exercise real child-process cleanup and UTF-8 logs without opening a GUI."""
import queue
import ctypes
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if sys.platform == "win32":
    from scripts.launch_frontend import FrontendLauncher, acquire_instance


@unittest.skipUnless(sys.platform == "win32", "The launcher uses Windows process flags")
class FrontendProcessTests(unittest.TestCase):
    def test_repeated_entry_uses_one_native_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("scripts.launch_frontend.ROOT", Path(directory)),
                patch("scripts.launch_frontend.WINDOW_TITLE", directory),
            ):
                kernel, handle = acquire_instance()
                try:
                    self.assertIsNone(acquire_instance())
                finally:
                    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                    kernel.CloseHandle(handle)
                kernel, handle = acquire_instance()
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(handle)

    def child(self, source):
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-u", "-c", source],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.addCleanup(process.stdout.close)
        self.addCleanup(FrontendLauncher.terminate, process)
        return process

    def test_stop_terminates_the_owned_live_child(self):
        process = self.child("import time; print('ready', flush=True); time.sleep(60)")
        self.assertEqual(process.stdout.readline().strip(), "ready")
        self.assertIsNone(process.poll())
        FrontendLauncher.terminate(process)
        self.assertIsNotNone(process.poll())

    def test_output_reader_preserves_unicode_and_reports_child_exit(self):
        process = self.child("print('启动就绪'); print('服务已退出')")
        launcher = FrontendLauncher.__new__(FrontendLauncher)
        launcher.events = queue.Queue()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "frontend.log"
            with patch("scripts.launch_frontend.LOG_PATH", log_path):
                launcher.read_output(process, generation=7)
            self.assertEqual(log_path.read_text(encoding="utf-8"), "启动就绪\n服务已退出\n")
        events = []
        while not launcher.events.empty():
            events.append(launcher.events.get_nowait())
        self.assertEqual(events, [
            (7, "line", "启动就绪"),
            (7, "line", "服务已退出"),
            (7, "exit", 0),
        ])


if __name__ == "__main__":
    unittest.main()
