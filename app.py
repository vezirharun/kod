#!/usr/bin/env python3
"""Vezir Pattern Search — Tekstil Desen Görsel Arama Motoru."""

from __future__ import annotations

import faulthandler
import sys
import traceback
from pathlib import Path

# Proje kökünü Python path'e ekle
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# OpenCV harden BEFORE any transitive cv2 import (V3/torch heap corruption).
from core.cv2_runtime import harden_cv2_runtime

harden_cv2_runtime()

from core.logger import setup_logger
from ui.main_window import run_app

logger = setup_logger("vezir")


def _install_exception_hooks() -> None:
    def log_exception(exc_type, exc_value, exc_tb) -> None:
        if exc_type is KeyboardInterrupt:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Yakalanmamış hata:\n%s", msg)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = log_exception


def main() -> None:
    _install_exception_hooks()
    # Native (libvips) crash'lerde stderr'e C stack; Python Traceback ≠ process kill.
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    logger.info("Vezir Pattern Search başlatılıyor…")
    try:
        run_app()
    except Exception:
        logger.exception("Kritik başlatma hatası")
        raise


if __name__ == "__main__":
    main()
