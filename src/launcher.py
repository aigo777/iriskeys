from __future__ import annotations

import os
import subprocess
import sys
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


_enable_windows_dpi_awareness()

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"
except ImportError:
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import (
            QApplication,
            QButtonGroup,
            QCheckBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QVBoxLayout,
            QWidget,
        )
        QT_API = "PyQt5"
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("PyQt6 or PyQt5 is required to run launcher.py") from exc


APP_TITLE = "IrisKeys OS"
ROOT_DIR = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT_DIR / "src" / "main.py"


def _is_usable_python(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _preferred_python() -> str:
    current = Path(sys.executable)
    if _is_usable_python(current):
        return str(current)

    candidates = [
        ROOT_DIR / "venv" / "Scripts" / "pythonw.exe",
        ROOT_DIR / "venv" / "Scripts" / "python.exe",
        ROOT_DIR / ".venv" / "Scripts" / "pythonw.exe",
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if _is_usable_python(candidate):
            return str(candidate)
    return sys.executable


class LaunchCard(QFrame):
    def __init__(self, accent: str, title: str, subtitle: str, facts: list[str], button_text: str, callback) -> None:
        super().__init__()
        self.setObjectName("launchCard")
        self.setProperty("accent", accent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 26, 26, 26)
        layout.setSpacing(18)

        accent_bar = QFrame()
        accent_bar.setObjectName("accentBar")
        accent_bar.setFixedHeight(6)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("cardSubtitle")
        subtitle_label.setWordWrap(True)

        facts_box = QVBoxLayout()
        facts_box.setSpacing(6)
        for fact in facts:
            fact_label = QLabel(fact)
            fact_label.setObjectName("factLabel")
            fact_label.setWordWrap(True)
            facts_box.addWidget(fact_label)

        button = QPushButton(button_text)
        button.setObjectName("primaryButton")
        button.clicked.connect(callback)

        layout.addWidget(accent_bar)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addLayout(facts_box)
        layout.addStretch(1)
        layout.addWidget(button)


class IrisKeysLauncher(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1340, 860)
        self.setObjectName("launcherRoot")
        self._build_ui()
        self._refresh_launch_hint()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 28, 34, 28)
        root.setSpacing(22)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(34, 32, 34, 32)
        hero_layout.setSpacing(14)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        for text in ("Finals Build", "Hands-Free Control", "Safe F12 Exit"):
            badge = QLabel(text)
            badge.setObjectName("pill")
            badge_row.addWidget(badge)
        badge_row.addStretch(1)

        title = QLabel(APP_TITLE)
        title.setObjectName("heroTitle")
        title.setFont(QFont("Segoe UI", 34, self._font_bold_weight()))

        subtitle = QLabel(
            "Calibrate once, then move directly into Demo or live OS control. "
            "The launch flow stays simple, safe, and presentation-ready."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        hero_layout.addLayout(badge_row)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(20)

        settings_card = QFrame()
        settings_card.setObjectName("settingsCard")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(28, 26, 28, 26)
        settings_layout.setSpacing(15)

        settings_title = QLabel("Session Settings")
        settings_title.setObjectName("sectionTitle")

        self.assist_checkbox = QCheckBox("Smart Magnetism Assist")
        self.assist_checkbox.setChecked(True)
        self.assist_checkbox.toggled.connect(self._refresh_launch_hint)

        self.dwell_checkbox = QCheckBox("Dwell Click")
        self.dwell_checkbox.setChecked(False)
        self.dwell_checkbox.toggled.connect(self._refresh_launch_hint)

        ml_title = QLabel("ML Stability Mode")
        ml_title.setObjectName("subSectionTitle")

        self.ml_group = QButtonGroup(self)
        self.ml_auto_radio = QRadioButton("Auto (recommended)")
        self.ml_auto_radio.setChecked(True)
        self.ml_off_radio = QRadioButton("Safe fallback only")
        self.ml_on_radio = QRadioButton("Force ML")
        for button in (self.ml_auto_radio, self.ml_off_radio, self.ml_on_radio):
            self.ml_group.addButton(button)
            button.toggled.connect(self._refresh_launch_hint)

        self.launch_hint = QLabel()
        self.launch_hint.setObjectName("launchHint")
        self.launch_hint.setWordWrap(True)

        settings_layout.addWidget(settings_title)
        settings_layout.addWidget(self.assist_checkbox)
        settings_layout.addWidget(self.dwell_checkbox)
        settings_layout.addSpacing(8)
        settings_layout.addWidget(ml_title)
        settings_layout.addWidget(self.ml_auto_radio)
        settings_layout.addWidget(self.ml_off_radio)
        settings_layout.addWidget(self.ml_on_radio)
        settings_layout.addSpacing(8)
        settings_layout.addWidget(self.launch_hint)
        settings_layout.addStretch(1)

        safety_card = QFrame()
        safety_card.setObjectName("safetyCard")
        safety_layout = QVBoxLayout(safety_card)
        safety_layout.setContentsMargins(28, 26, 28, 26)
        safety_layout.setSpacing(14)

        safety_title = QLabel("OS Mode Safety")
        safety_title.setObjectName("sectionTitle")

        safety_text = QLabel(
            "OS Mode hides the OpenCV sandbox and drives the real Windows cursor. "
            "Press F12 or ESC at any time to instantly stop control."
        )
        safety_text.setObjectName("warningText")
        safety_text.setWordWrap(True)

        safety_steps = QLabel(
            "Finals flow:\n"
            "1. Choose Demo Mode or OS Mode\n"
            "2. Calibration starts automatically\n"
            "3. Tracking continues immediately in the selected mode"
        )
        safety_steps.setObjectName("guideText")

        safety_layout.addWidget(safety_title)
        safety_layout.addWidget(safety_text)
        safety_layout.addWidget(safety_steps)
        safety_layout.addStretch(1)

        settings_row.addWidget(settings_card, 3)
        settings_row.addWidget(safety_card, 2)

        cards_grid = QGridLayout()
        cards_grid.setHorizontalSpacing(20)
        cards_grid.setVerticalSpacing(20)

        demo_card = LaunchCard(
            "cool",
            "Launch Demo Mode",
            "Runs the full calibration first, then keeps everything inside the presentation sandbox.",
            [
                "Best for rehearsal, tuning, and stage presentation.",
                "No second click needed after calibration.",
            ],
            "Start Demo",
            self.launch_demo,
        )
        os_card = LaunchCard(
            "green",
            "Launch OS Mode",
            "Runs calibration first, then transitions directly into real Windows cursor control.",
            [
                "Avoids baseline drift from an extra launcher click.",
                "Failsafe: F12 or ESC instantly exits control.",
            ],
            "Start OS Mode",
            self.launch_os_mode,
        )

        cards_grid.addWidget(demo_card, 0, 0)
        cards_grid.addWidget(os_card, 0, 1)

        footer = QLabel(
            "Backend: IrisKeys tracking, calibration, smart magnetism, and dwell selection"
        )
        footer.setObjectName("footerText")

        root.addWidget(hero)
        root.addLayout(settings_row)
        root.addLayout(cards_grid, 1)
        root.addWidget(footer)

        self.setStyleSheet(
            """
            QWidget#launcherRoot {
                background: #091018;
                color: #edf4fb;
                font-family: Segoe UI;
                font-size: 17px;
            }
            QLabel {
                background: transparent;
            }
            QCheckBox, QRadioButton {
                background: transparent;
            }
            QFrame#heroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #122335, stop:0.5 #1a3044, stop:1 #0d1823);
                border: 1px solid #2d4960;
                border-radius: 26px;
            }
            QFrame#settingsCard, QFrame#safetyCard, QFrame#launchCard {
                background: #111b26;
                border: 1px solid #26394a;
                border-radius: 22px;
            }
            QFrame#accentBar {
                border: none;
                border-radius: 3px;
                background: #3f82ff;
            }
            QFrame#launchCard[accent="warm"] QFrame#accentBar {
                background: #ff9a52;
            }
            QFrame#launchCard[accent="cool"] QFrame#accentBar {
                background: #49b0ff;
            }
            QFrame#launchCard[accent="green"] QFrame#accentBar {
                background: #52d28c;
            }
            QLabel#heroTitle {
                color: #ffffff;
            }
            QLabel#heroSubtitle, QLabel#cardSubtitle, QLabel#footerText {
                color: #bfd0df;
                font-size: 18px;
            }
            QLabel#sectionTitle, QLabel#cardTitle {
                color: #ffffff;
                font-size: 22px;
                font-weight: 600;
            }
            QLabel#subSectionTitle {
                color: #e6f0f8;
                font-size: 18px;
                font-weight: 600;
                padding-top: 6px;
            }
            QLabel#factLabel {
                color: #c4d3e1;
                padding-left: 2px;
                font-size: 16px;
            }
            QLabel#launchHint {
                color: #9fd0ff;
                background: transparent;
                border: none;
                padding: 4px 2px;
                font-size: 17px;
            }
            QLabel#warningText {
                color: #ffd79a;
                background: transparent;
                border: none;
                padding: 4px 2px;
                font-size: 17px;
            }
            QLabel#guideText {
                color: #d8e4ef;
                background: transparent;
                border: none;
                padding: 4px 2px;
                font-size: 17px;
            }
            QLabel#pill {
                color: #dceeff;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(180, 220, 255, 0.18);
                border-radius: 12px;
                padding: 6px 12px;
                font-size: 15px;
            }
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1f74db, stop:1 #39a0ff);
                border: none;
                border-radius: 18px;
                padding: 18px 22px;
                color: white;
                font-weight: 600;
                font-size: 18px;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2a83eb, stop:1 #49b0ff);
            }
            QPushButton#primaryButton:pressed {
                background: #1e6ece;
            }
            QCheckBox {
                spacing: 10px;
                color: #edf4fb;
                font-size: 17px;
            }
            QRadioButton {
                spacing: 10px;
                color: #edf4fb;
                font-size: 17px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #47637d;
                border-radius: 5px;
                background: #0e151d;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2b8cff;
                border-radius: 5px;
                background: #2b8cff;
            }
            QRadioButton::indicator:unchecked {
                border: 1px solid #47637d;
                border-radius: 9px;
                background: #0e151d;
            }
            QRadioButton::indicator:checked {
                border: 1px solid #2b8cff;
                border-radius: 9px;
                background: #2b8cff;
            }
            """
        )

    @staticmethod
    def _font_bold_weight() -> int:
        if QT_API == "PyQt6":
            return int(QFont.Weight.Bold)
        return int(QFont.Bold)

    def _refresh_launch_hint(self) -> None:
        assist_text = "ON" if self.assist_checkbox.isChecked() else "OFF"
        dwell_text = "ON" if self.dwell_checkbox.isChecked() else "OFF"
        if self.ml_off_radio.isChecked():
            ml_text = "Safe fallback only"
        elif self.ml_on_radio.isChecked():
            ml_text = "Force ML"
        else:
            ml_text = "Auto safety mode"
        self.launch_hint.setText(
            f"Current launch profile: Smart Magnetism {assist_text} | Dwell Click {dwell_text} | ML {ml_text}"
        )

    def _selected_ml_mode(self) -> str:
        if self.ml_off_radio.isChecked():
            return "off"
        if self.ml_on_radio.isChecked():
            return "on"
        return "auto"

    def _build_backend_args(self, mode: str, auto_calibrate: bool = False) -> list[str]:
        args = [_preferred_python(), str(MAIN_SCRIPT), "--mode", mode]
        args.extend(["--assist", "on" if self.assist_checkbox.isChecked() else "off"])
        args.extend(["--click", "dwell" if self.dwell_checkbox.isChecked() else "off"])
        args.extend(["--os-click", "on" if self.dwell_checkbox.isChecked() else "off"])
        args.extend(["--ml", self._selected_ml_mode()])
        if auto_calibrate:
            args.extend(["--auto-calibrate", "on"])
        return args

    def _launch(self, args: list[str], label: str) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        src_path = str(ROOT_DIR / "src")
        env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
        try:
            subprocess.Popen(args, cwd=str(ROOT_DIR), env=env)
        except Exception as exc:
            QMessageBox.critical(self, APP_TITLE, f"Failed to launch {label}.\n\n{exc}")

    def launch_demo(self) -> None:
        self._launch(
            self._build_backend_args("demo", auto_calibrate=True) + ["--post-calibration-mode", "demo"],
            "demo mode",
        )

    def launch_os_mode(self) -> None:
        answer = QMessageBox.question(
            self,
            APP_TITLE,
            "OS Mode will start calibration first and then automatically switch into real Windows control.\n\nUse F12 or ESC as a kill-switch.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            if QT_API == "PyQt6"
            else QMessageBox.Yes | QMessageBox.No,
        )
        yes_value = QMessageBox.StandardButton.Yes if QT_API == "PyQt6" else QMessageBox.Yes
        if answer != yes_value:
            return
        self._launch(
            self._build_backend_args("demo", auto_calibrate=True) + ["--post-calibration-mode", "os"],
            "OS mode",
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    window = IrisKeysLauncher()
    window.show()
    if QT_API == "PyQt6":
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
