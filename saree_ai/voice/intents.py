"""Local command / instruction-response engine (English + Telugu). No LLM, no network.

Matching = normalised exact phrase match, then fuzzy match (difflib) against the phrase table of the
session language PLUS English (people often mix in English words like "next"). The phrase lists
double as the Vosk grammars, which keeps recognition constrained to commands.

Telugu phrases use only words present in vocabulary of vosk-model-small-te-0.42 (checked
2026-11-06; e.g. the loan word "ప్లీట్స్" is NOT in it, so it is not used). They should be reviewed by
native speakers during the pilot (DATA_COLLECTION_PLAN.md) and extended with what people actually say.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from ..i18n import T, render

INTENTS: dict[str, dict[str, list[str]]] = {
    "en": {
        "next": ["next", "next step", "go next", "continue", "done", "i am done", "i'm done", "move on"],
        "previous": ["previous", "previous step", "go back", "back", "last step"],
        "repeat": ["repeat", "repeat instruction", "say again", "say that again", "again", "what do i do"],
        "explain_pleats": ["explain pleats", "explain pleat formation", "how do i make pleats", "how to make pleats", "pleats help"],
        # "pallu" is not in the small English Vosk vocabulary; it drops the word, so residual phrases are listed too.
        "show_pallu": ["show pallu", "show pallu placement", "where does the pallu go", "pallu help", "explain pallu",
                       "show placement", "show shoulder drape", "where does it go on the shoulder"],
        "explain_wrap": ["explain wrapping", "how do i wrap", "wrap help"],
        "restart": ["start again", "restart", "start over", "begin again"],
        "check": ["am i doing this correctly", "am i doing it right", "is this correct", "check", "check me", "how does it look"],
        "confirm": ["confirm", "looks good", "it is correct", "yes it's done", "mark done"],
        "help": ["help", "what can i say", "commands"],
    },
    "te": {
        "next": ["తరువాత", "తర్వాత", "తరువాతి దశ", "తర్వాతి దశ", "తదుపరి", "తదుపరి దశ", "ముందుకు", "నెక్స్ట్", "అయిపోయింది"],
        "previous": ["వెనక్కి", "వెనుకకు", "మునుపటి దశ", "బ్యాక్"],
        "repeat": ["మళ్లీ చెప్పు", "మళ్లీ చెప్పండి", "రిపీట్"],
        "explain_pleats": ["కుచ్చిళ్లు ఎలా చేయాలి", "కుచ్చిళ్లు ఎలా", "మడతలు ఎలా చేయాలి", "కుచ్చిళ్లు వివరించు"],
        "show_pallu": ["పైట ఎలా వేయాలి", "పైట ఎక్కడ", "పైట చూపించు", "కొంగు ఎక్కడ", "కొంగు ఎలా వేయాలి", "పల్లు చూపించు"],
        "explain_wrap": ["ఎలా చుట్టాలి", "చుట్టడం ఎలా"],
        "restart": ["మళ్లీ మొదలు పెట్టు", "మళ్లీ మొదలు పెట్టండి", "మొదటి నుంచి", "మొదటి నుంచి మొదలు పెట్టు"],
        "check": ["నేను సరిగ్గా చేస్తున్నానా", "సరిగ్గా ఉందా", "సరిగా ఉందా", "చెక్"],
        "confirm": ["బాగుంది", "సరిగ్గా ఉంది"],
        "help": ["సహాయం", "ఏమి చెప్పాలి"],
    },
}

_TELUGU = "ఀ-౿"


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower().strip()
    t = t.replace("ళ్ళ", "ళ్ల")                      # two common spellings of the same conjunct
    t = re.sub(rf"[^a-z'{_TELUGU} ]+", " ", t)
    words = t.split()
    # collapse immediate repeats: "next next" (people repeat when unsure they were heard) -> "next"
    return " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])


def _table(lang: str) -> dict[str, str]:
    out = {}
    for lg in ([lang, "en"] if lang != "en" else ["en"]):
        for intent, phrases in INTENTS.get(lg, {}).items():
            for p in phrases:
                out.setdefault(normalise(p), intent)
    return out


def grammar_phrases(lang: str = "en") -> list[str]:
    """Phrase list for a constrained recogniser. '[unk]' lets Vosk reject out-of-grammar speech."""
    return sorted({normalise(p) for ps in INTENTS[lang].values() for p in ps}) + ["[unk]"]


def parse(text: str, lang: str = "en", min_score: float = 0.75) -> dict:
    t = normalise(text)
    table = _table(lang)
    if not t:
        return {"intent": None, "score": 0.0, "text": text}
    if t in table:
        return {"intent": table[t], "score": 1.0, "text": text}
    best, best_s = None, 0.0
    for p, i in table.items():
        s = SequenceMatcher(None, t, p).ratio()
        if s > best_s:
            best, best_s = i, s
    if best_s >= min_score:
        return {"intent": best, "score": round(best_s, 3), "text": text}
    return {"intent": None, "score": round(best_s, 3), "text": text}


HELP = T("You can say: next step, previous step, repeat instruction, explain pleat formation, "
         "show pallu placement, start again, am I doing this correctly, or confirm.")
HELP_TEXT = str(HELP)


def respond(intent: str | None, engine, last_guidance: dict | None) -> dict:
    """Map an intent to an engine action + spoken reply in the session language."""
    r, t = engine.r, engine.t

    def step_line(prefix=None):
        st = engine.step
        line = r(T("Step {n}. {title}.", n=engine.step_index + 1, title=engine.say("title", st))) + " " + engine.say("speech", st)
        return (r(prefix) + " " + line) if prefix else line

    if intent is None:
        return {"action": "none", "speech": r(T("Sorry, I didn't catch that.")) + " " + r(HELP)}
    if intent == "next":
        engine.next(); return {"action": "step", "speech": step_line()}
    if intent == "previous":
        engine.previous(); return {"action": "step", "speech": step_line()}
    if intent == "repeat":
        return {"action": "repeat", "speech": engine.say("speech")}
    if intent == "restart":
        engine.restart(); return {"action": "step", "speech": step_line(T("Starting again."))}
    if intent == "explain_pleats":
        return {"action": "explain", "speech": t.explanation("pleats", engine.lang), "show_guide": "pleat_zone"}
    if intent == "show_pallu":
        return {"action": "explain", "speech": t.explanation("pallu", engine.lang), "show_guide": "pallu_path"}
    if intent == "explain_wrap":
        return {"action": "explain", "speech": t.explanation("wrap", engine.lang), "show_guide": "wrap_arrow"}
    if intent == "confirm":
        engine.confirm_manual(None); return {"action": "confirm", "speech": r(T("Okay, marked as done. Say next when you're ready."))}
    if intent == "help":
        return {"action": "help", "speech": r(HELP)}
    if intent == "check":
        return {"action": "check", "speech": check_answer(last_guidance, engine.lang)}
    return {"action": "none", "speech": r(HELP)}


def check_answer(g: dict | None, lang: str = "en") -> str:
    """Honest answer to 'am I doing this correctly?' built only from the latest guidance
    (whose messages are already rendered in the session language)."""
    r = lambda m: render(m, lang)  # noqa: E731
    if not g:
        return r(T("I can't check this step from the camera here. When it looks right to you, say confirm or next."))
    st = g["status"]
    if st == "need_position":
        return r(T("I can't see you well enough to check.")) + " " + g["headline"]
    if st == "correction":
        return g["headline"]
    if st == "step_complete":
        return r(T("Yes, this step looks right to me."))
    passed = [c for c in g["checks"] if c["status"] == "pass" and c["id"] != "full_body_visible"]
    unsure = [c for c in g["checks"] if c["status"] in ("uncertain", "unverified")]
    if unsure and not passed:
        return r(T("I'm not confident enough to judge this step from the camera.")) + " " + unsure[0]["message"]
    if unsure:
        return r(T("What I can see looks fine, but I couldn't check everything.")) + " " + unsure[0]["message"]
    return r(T("So far so good. Keep going."))
