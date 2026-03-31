from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path


def _enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDPIAware"):
            user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_windows_desktop_rect() -> tuple[int, int, int, int]:
    if os.name != "nt":
        return (0, 0, 1920, 1080)
    try:
        import ctypes

        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        return (0, 0, max(1, width), max(1, height))
    except Exception:
        return (0, 0, 1920, 1080)


_enable_windows_dpi_awareness()

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_API = "PyQt6"
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

    QT_API = "PyQt5"


VOICE_PREPARE_DELAY_S = 4.0
VOICE_LISTEN_TIMEOUT_S = 5
VOICE_PHRASE_LIMIT_S = 12


class TextPadWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IrisKeys Text Pad")
        self.setObjectName("textPadRoot")
        flags = Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
        if QT_API == "PyQt5":
            flags = Qt.Tool | Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.resize(760, 440)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("IrisKeys Text Pad")
        title.setObjectName("padTitle")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold if QT_API == "PyQt6" else QFont.Bold))

        hint = QLabel("Use the on-screen keyboard here or let voice input collect text into this window.")
        hint.setObjectName("padHint")
        hint.setWordWrap(True)

        self.editor = QTextEdit()
        self.editor.setObjectName("padEditor")
        self.editor.setPlaceholderText("Voice and keyboard text will appear here...")

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.copy_btn = QPushButton("Copy")
        self.clear_btn = QPushButton("Clear")
        self.focus_btn = QPushButton("Focus")
        self.copy_btn.clicked.connect(self.copy_all)
        self.clear_btn.clicked.connect(self.editor.clear)
        self.focus_btn.clicked.connect(self.focus_editor)
        buttons.addWidget(self.copy_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addWidget(self.focus_btn)
        buttons.addStretch(1)

        root.addWidget(title)
        root.addWidget(hint)
        root.addWidget(self.editor, 1)
        root.addLayout(buttons)

        self.setStyleSheet(
            """
            QWidget#textPadRoot {
                background: #111b26;
                color: #edf4fb;
                font-family: Segoe UI;
                font-size: 15px;
            }
            QLabel {
                background: transparent;
            }
            QLabel#padTitle {
                color: #ffffff;
            }
            QLabel#padHint {
                color: #c1d3e2;
            }
            QTextEdit#padEditor {
                background: #0d1620;
                border: 1px solid #294355;
                border-radius: 16px;
                padding: 12px;
                color: #ffffff;
                font-size: 18px;
            }
            QPushButton {
                min-height: 52px;
                min-width: 116px;
                border: none;
                border-radius: 14px;
                color: #ffffff;
                background: #1b4868;
                font-size: 16px;
                font-weight: 600;
                padding: 10px 16px;
            }
            QPushButton:hover {
                background: #27628a;
            }
            """
        )

    def append_text(self, text: str) -> None:
        existing = self.editor.toPlainText().strip()
        self.editor.setPlainText((existing + ("\n" if existing else "") + text).strip())
        self.focus_editor()

    def focus_editor(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.editor.setFocus()
        cursor = self.editor.textCursor()
        move_end = cursor.MoveOperation.End if QT_API == "PyQt6" else cursor.End
        cursor.movePosition(move_end)
        self.editor.setTextCursor(cursor)

    def copy_all(self) -> None:
        QApplication.clipboard().setText(self.editor.toPlainText())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IrisKeys floating accessibility toolbar")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--dock", choices=("top", "left", "right"), default="top")
    return parser.parse_args()


class FloatingToolbar(QWidget):
    voice_finished = pyqtSignal(int, str)

    def __init__(self, state_file: Path, dock: str = "top") -> None:
        super().__init__()
        self.state_file = Path(state_file)
        self.dock = dock
        self.tracking_paused = False
        self.next_click_button = "left"
        self.switch_mode = "none"
        self.voice_state = "idle"
        self.voice_status_text = "Ready"
        self.text_pad: TextPadWindow | None = None
        self.voice_target_text_pad = False
        self.voice_prepare_deadline = 0.0
        self.voice_previous_paused_state = False
        self.voice_session_id = 0
        self.voice_prepare_timer = QTimer(self)
        self.voice_prepare_timer.setInterval(200)
        self.voice_prepare_timer.timeout.connect(self._tick_voice_prepare)
        self.voice_finished.connect(self._finish_voice_type)
        self._build_ui()
        self._load_state()
        self._sync_state()
        self._dock_to_edge()

    def _build_ui(self) -> None:
        flags = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        if QT_API == "PyQt5":
            flags = Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setObjectName("toolbarRoot")

        root = QHBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(16)

        title = QLabel("IrisKeys")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if QT_API == "PyQt6" else Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, self._font_bold_weight()))

        subtitle = QLabel("OS toolbar")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter if QT_API == "PyQt6" else Qt.AlignCenter)

        title_wrap = QWidget()
        title_layout = QVBoxLayout(title_wrap)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.addStretch(1)

        self.keyboard_btn = self._make_button("Keyboard", self.open_keyboard)
        self.voice_btn = self._make_button("Voice Type", self.voice_type)
        self.text_pad_btn = self._make_button("Text Pad", self.open_text_pad)
        self.demo_btn = self._make_button("Demo Mode", self.switch_to_demo_mode)
        self.right_click_btn = self._make_button("Right Click", self.toggle_right_click)
        self.pause_btn = self._make_button("Pause", self.toggle_pause)

        self.state_label = QLabel()
        self.state_label.setObjectName("stateLabel")
        self.state_label.setWordWrap(True)
        self.state_label.setMinimumWidth(150)

        root.addWidget(title_wrap)
        root.addWidget(self.keyboard_btn)
        root.addWidget(self.voice_btn)
        root.addWidget(self.text_pad_btn)
        root.addWidget(self.demo_btn)
        root.addWidget(self.right_click_btn)
        root.addWidget(self.pause_btn)
        root.addWidget(self.state_label)

        self.setStyleSheet(
            """
            QWidget#toolbarRoot {
                background: #101923;
                border: 1px solid #314659;
                border-radius: 24px;
            }
            QLabel {
                background: transparent;
            }
            QLabel#titleLabel {
                color: #ffffff;
                padding: 2px 10px 0 4px;
            }
            QLabel#subtitleLabel {
                color: #8fb8d8;
                font-size: 13px;
                padding: 0 10px 0 4px;
            }
            QLabel#stateLabel {
                color: #c8dceb;
                background: #13212e;
                border: 1px solid #223646;
                border-radius: 16px;
                padding: 14px;
                font-size: 16px;
            }
            QPushButton {
                min-width: 148px;
                min-height: 148px;
                border: none;
                border-radius: 24px;
                color: #ffffff;
                background: #183249;
                font-size: 19px;
                font-weight: 600;
                padding: 14px;
            }
            QPushButton:hover {
                background: #245071;
            }
            QPushButton:pressed {
                background: #143044;
            }
            QPushButton[state="armed"] {
                background: #2f7d3e;
            }
            QPushButton[state="arming"] {
                background: #5a4d94;
            }
            QPushButton[state="paused"] {
                background: #856126;
            }
            """
        )

    @staticmethod
    def _font_bold_weight() -> int:
        if QT_API == "PyQt6":
            return int(QFont.Weight.Bold)
        return int(QFont.Bold)

    def _make_button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _dock_to_edge(self) -> None:
        vx, vy, width_px, height_px = _get_windows_desktop_rect()
        top_margin = vy + max(18, int(height_px * 0.025))
        side_margin = max(18, int(width_px * 0.018))
        if self.dock == "top":
            available_width = max(980, width_px - side_margin * 2)
            width = min(max(1380, int(width_px * 0.84)), available_width)
            height = 194
            x = vx + max(side_margin, int((width_px - width) / 2))
            y = top_margin
        else:
            width = 164
            height = min(680, max(520, int(height_px * 0.72)))
            x = vx + side_margin if self.dock == "left" else vx + width_px - width - side_margin
            y = vy + max(20, int((height_px - height) * 0.16))
        self.setGeometry(int(x), int(y), int(width), int(height))

    def _load_state(self) -> None:
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        self.tracking_paused = bool(payload.get("paused", False))
        self.next_click_button = "right" if payload.get("next_click_button") == "right" else "left"
        self.switch_mode = "demo" if payload.get("switch_mode") == "demo" else "none"

    def _sync_state(self) -> None:
        payload = {
            "paused": bool(self.tracking_paused),
            "next_click_button": "right" if self.next_click_button == "right" else "left",
            "switch_mode": self.switch_mode if self.switch_mode in ("demo", "none") else "none",
        }
        tmp_path = str(self.state_file) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp_path, self.state_file)
        except Exception:
            pass
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        pause_text = "Paused" if self.tracking_paused else "Tracking Live"
        click_text = "Next click: Right" if self.next_click_button == "right" else "Next click: Left"
        mode_text = "Mode switch: Demo pending" if self.switch_mode == "demo" else "Mode switch: None"
        self.state_label.setText(f"{pause_text}\n{click_text}\nVoice: {self.voice_status_text}\n{mode_text}")

        self.right_click_btn.setProperty("state", "armed" if self.next_click_button == "right" else "")
        self.pause_btn.setProperty("state", "paused" if self.tracking_paused else "")
        self.pause_btn.setText("Resume" if self.tracking_paused else "Pause")
        self.demo_btn.setProperty("state", "arming" if self.switch_mode == "demo" else "")
        self.demo_btn.setText("Opening Demo..." if self.switch_mode == "demo" else "Demo Mode")
        self.voice_btn.setProperty("state", "arming" if self.voice_state == "prepare" else "")
        self.text_pad_btn.setProperty("state", "armed" if self.text_pad is not None and self.text_pad.isVisible() else "")
        self.demo_btn.setEnabled(self.switch_mode != "demo")
        self.voice_btn.setEnabled(self.voice_state != "listening")
        if self.voice_state == "idle":
            self.voice_btn.setText("Voice Type")
        elif self.voice_state == "prepare":
            remaining = max(1, int(self.voice_prepare_deadline - time.monotonic() + 0.999))
            self.voice_btn.setText(f"Voice in {remaining}...\nTap to cancel")
        else:
            self.voice_btn.setText("Listening...")
        self._polish_button(self.right_click_btn)
        self._polish_button(self.pause_btn)
        self._polish_button(self.demo_btn)
        self._polish_button(self.voice_btn)
        self._polish_button(self.text_pad_btn)

    @staticmethod
    def _polish_button(button: QPushButton) -> None:
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

    def open_keyboard(self) -> None:
        try:
            self.open_text_pad()
            os.system("osk")
        except Exception as exc:
            self._show_error(f"Failed to open On-Screen Keyboard.\n\n{exc}")

    def open_text_pad(self) -> None:
        if self.text_pad is None:
            self.text_pad = TextPadWindow()
            vx, vy, width_px, _ = _get_windows_desktop_rect()
            pad_width = self.text_pad.width()
            x = vx + max(24, int((width_px - pad_width) / 2))
            y = vy + 230
            self.text_pad.move(int(x), int(y))
        self.text_pad.focus_editor()
        self._refresh_labels()

    def voice_type(self) -> None:
        if self.voice_state == "prepare":
            self._cancel_voice_prepare()
            return
        if self.voice_state != "idle":
            return

        self.voice_target_text_pad = bool(self.text_pad is not None and self.text_pad.isVisible())
        if self.voice_target_text_pad:
            self.open_text_pad()

        self.voice_state = "prepare"
        self.voice_status_text = f"Ready in {int(VOICE_PREPARE_DELAY_S)}s"
        self.voice_prepare_deadline = time.monotonic() + VOICE_PREPARE_DELAY_S
        self.voice_prepare_timer.start()
        self._refresh_labels()

    def _tick_voice_prepare(self) -> None:
        if self.voice_state != "prepare":
            self.voice_prepare_timer.stop()
            return
        remaining = self.voice_prepare_deadline - time.monotonic()
        if remaining <= 0.0:
            self.voice_prepare_timer.stop()
            self._start_voice_listening()
            return
        self.voice_status_text = f"Ready in {max(1, int(remaining + 0.999))}s"
        self._refresh_labels()

    def _cancel_voice_prepare(self) -> None:
        self.voice_prepare_timer.stop()
        self.voice_state = "idle"
        self.voice_status_text = "Ready"
        self.voice_target_text_pad = False
        self.voice_prepare_deadline = 0.0
        self._refresh_labels()

    def _start_voice_listening(self) -> None:
        if self.voice_state != "prepare":
            return
        self.voice_state = "listening"
        self.voice_prepare_deadline = 0.0
        self.voice_previous_paused_state = bool(self.tracking_paused)
        self.tracking_paused = True
        self.voice_status_text = "Listening"
        self.voice_session_id += 1
        current_session = self.voice_session_id
        self._sync_state()

        def worker() -> None:
            error_message = None
            recognized_text = ""
            try:
                import pyautogui
                import speech_recognition as sr
                import win32clipboard
                import win32con

                recognizer = sr.Recognizer()
                recognizer.pause_threshold = 0.95
                recognizer.non_speaking_duration = 0.6
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    audio = recognizer.listen(
                        source,
                        timeout=VOICE_LISTEN_TIMEOUT_S,
                        phrase_time_limit=VOICE_PHRASE_LIMIT_S,
                    )
                text = ""
                last_error: Exception | None = None
                for language in ("kk-KZ", "ru-RU", "en-US"):
                    try:
                        text = recognizer.recognize_google(audio, language=language).strip()
                        if text:
                            break
                    except Exception as exc:
                        last_error = exc
                if not text:
                    raise RuntimeError(str(last_error) if last_error is not None else "Speech was not recognized.")
                recognized_text = text
                if self.voice_target_text_pad:
                    pass
                else:
                    win32clipboard.OpenClipboard()
                    try:
                        win32clipboard.EmptyClipboard()
                        win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
                    finally:
                        win32clipboard.CloseClipboard()
                    pyautogui.hotkey("ctrl", "v")
            except Exception as exc:  # pragma: no cover - depends on runtime devices
                error_message = str(exc)
            self.voice_finished.emit(current_session, recognized_text if error_message is None else f"ERROR::{error_message}")

        threading.Thread(target=worker, daemon=True).start()

    def _finish_voice_type(self, session_id: int, payload: str) -> None:
        if session_id != self.voice_session_id or self.voice_state != "listening":
            return
        self.tracking_paused = bool(self.voice_previous_paused_state)
        self.voice_state = "idle"
        self.voice_status_text = "Ready"
        self._sync_state()
        if payload.startswith("ERROR::"):
            self.voice_target_text_pad = False
            self._show_error(
                "Voice typing failed.\n\nMake sure microphone access, speech_recognition, and pyautogui are available.\n\n"
                + payload.removeprefix("ERROR::")
            )
            return
        if payload and self.voice_target_text_pad:
            self.open_text_pad()
            if self.text_pad is not None:
                self.text_pad.append_text(payload)
        self.voice_target_text_pad = False

    def toggle_right_click(self) -> None:
        self.next_click_button = "left" if self.next_click_button == "right" else "right"
        self._sync_state()

    def switch_to_demo_mode(self) -> None:
        if self.switch_mode == "demo":
            return
        self.switch_mode = "demo"
        self.voice_prepare_timer.stop()
        self._sync_state()

    def toggle_pause(self) -> None:
        self.tracking_paused = not self.tracking_paused
        self._sync_state()

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "IrisKeys Toolbar", message)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("IrisKeysToolbar")
    toolbar = FloatingToolbar(Path(args.state_file), dock=args.dock)
    toolbar.show()
    if QT_API == "PyQt6":
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
