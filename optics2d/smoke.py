"""Exercise the packaged app with --smoke-test <report.json>."""
import json
from pathlib import Path

from .i18n import translator
from .model import OpticalSystem
from .version import APP_TITLE


def run_smoke_test(app, window, report_path):
    assert window.windowTitle() == APP_TITLE
    assert translator.code == "en"
    window.show()
    app.processEvents()
    window.add_element("horn")
    window.add_frequency()
    assert len(window.selected().frequency_channels) == 2
    snapshot = window.system.to_dict()
    window._change_language("ru")
    app.processEvents()
    assert window.undo_action.text() == "Назад"
    window._change_language("en")
    app.processEvents()
    assert window.system.to_dict() == snapshot
    assert OpticalSystem.from_dict(snapshot).to_dict() == snapshot
    window.undo()
    assert len(window.selected().frequency_channels) == 1
    Path(report_path).write_text(json.dumps({
        "application": APP_TITLE,
        "status": "passed",
        "checks": ["startup", "bundled locales", "add horn", "add frequency",
                   "language switching", "JSON round trip", "undo"],
    }, indent=2), encoding="utf-8")
    window.close()
