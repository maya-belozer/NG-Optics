"""UTF-8 JSON translations with English fallback and validated placeholders."""
from collections.abc import Mapping
import json
from pathlib import Path
import re
from string import Formatter

LOCALE_DIR = Path(__file__).resolve().parent.parent / "locales"


def _fields(text):
    fields = []
    for _, field, spec, conversion in Formatter().parse(text):
        if field is not None:
            if not field.isdecimal() or "{" in spec or "}" in spec:
                raise ValueError("Only numbered placeholders with fixed format specifications are allowed")
            fields.append((field, spec, conversion))
    return sorted(fields)


class Translator:
    def __init__(self):
        self.catalogs = {}
        self.code = "en"
        self.load(LOCALE_DIR / "en.json")
        self.load(LOCALE_DIR / "ru.json")

    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        code, name, messages = (data.get(k) for k in ("code", "name", "messages"))
        if not isinstance(code, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,31}", code):
            raise ValueError("Invalid language code")
        if not isinstance(name, str) or not name.strip() or not isinstance(messages, dict):
            raise ValueError("Expected a language name and messages object")
        for key, value in messages.items():
            if not isinstance(value, str) or _fields(key) != _fields(value):
                raise ValueError(f"Invalid translation or placeholders: {key}")
        if self.catalogs and code == "en":
            raise ValueError("The built-in English fallback cannot be replaced")
        unknown = messages.keys() - self.catalogs.get("en", {}).get("messages", messages).keys()
        if unknown:
            raise ValueError(f"Unknown message key: {next(iter(unknown))}")
        self.catalogs[code] = {"name": name.strip(), "messages": messages}
        return code

    def set_language(self, code):
        if code not in self.catalogs:
            raise ValueError(f"Unknown language: {code}")
        self.code = code

    def translate(self, key, *args):
        value = self.catalogs[self.code]["messages"].get(key, key)
        return value.format(*args) if args else value


translator = Translator()
tr = translator.translate


class LocalizedLabels(Mapping):
    """Resolve labels at access time, including after switching languages."""
    def __init__(self, labels):
        self.labels = labels

    def __getitem__(self, key):
        return tr(self.labels[key])

    def __iter__(self):
        return iter(self.labels)

    def __len__(self):
        return len(self.labels)
