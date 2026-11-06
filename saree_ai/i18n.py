"""Tiny local message catalog (gettext-style, no network).

Source strings are English templates written with T("... {n} ...", n=3). A T value behaves like the
rendered English string, but keeps its template + params so it can be re-rendered in another
language: render(msg, "te"). Catalogs live in config/i18n/<lang>.yaml, keyed by the English template.
Parameter values that are themselves words (e.g. side="left") are translated through the
catalog's `words` table.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LANGS = {"en": "English", "te": "తెలుగు"}
DEFAULT_LANG = "en"


class T(str):
    def __new__(cls, template: str, **params):
        s = super().__new__(cls, template.format(**params))
        s.template, s.params = template, params
        return s


@lru_cache(maxsize=None)
def catalog(lang: str) -> dict:
    p = ROOT / "config" / "i18n" / f"{lang}.yaml"
    if lang == "en" or not p.exists():
        return {"messages": {}, "words": {}}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {"messages": d.get("messages", {}), "words": d.get("words", {})}


def render(msg, lang: str = DEFAULT_LANG) -> str:
    """Render a T (or a plain str, which is looked up as a template without params)."""
    if msg is None:
        return ""
    if lang == "en":
        return str(msg)
    c = catalog(lang)
    template = getattr(msg, "template", str(msg))
    params = getattr(msg, "params", {})
    out = c["messages"].get(template)
    if out is None:
        out = template    # untranslated: fall back to the English template rather than show nothing
    params = {k: (render(v, lang) if isinstance(v, T) else c["words"].get(v, v) if isinstance(v, str) else v)
              for k, v in params.items()}
    return out.format(**params)


def missing(lang: str, templates: list[str]) -> list[str]:
    c = catalog(lang)["messages"]
    return [t for t in templates if t not in c]
