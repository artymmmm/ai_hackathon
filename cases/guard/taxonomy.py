"""Фиксированная таксономия признаков атаки для кейса 2 (инъекция на входе).

Девять тегов, зафиксированных в PLAN.md, и детерминированные regex-правила,
которые их проставляют. Правила осознанно консервативны (высокий precision,
не 100% recall) — они формируют признаки для baseline.py и объясняемые теги
для отчёта, а не финальный вердикт сами по себе.
"""

import re

# Порядок фиксирован и совпадает с PLAN.md — не менять произвольно.
TAGS: list[str] = [
    "role_override",
    "system_prompt_extraction",
    "instruction_ignore",
    "obfuscation_base64",
    "encoding_smuggling",
    "roleplay_jailbreak",
    "translation_wrapper",
    "code_block_smuggling",
    "direct_harmful_request",
]

_FLAGS = re.IGNORECASE | re.MULTILINE

_ROLE_OVERRIDE = [
    r"\byou are now\b",
    r"\bact as (a|an|the)\b",
    r"\bpretend (to be|you are)\b",
    r"\bfrom now on,?\s+you\b",
    r"\byou are no longer\b",
    r"\byour new (role|persona)\b",
    r"^\s*(system|assistant)\s*:",
]

_SYSTEM_PROMPT_EXTRACTION = [
    r"\bsystem prompt\b",
    r"\breveal\b.{0,30}\b(instructions|prompt|directives|password)\b",
    r"\binternal directives\b",
    r"\bmaster password\b",
    r"\binitial prompt\b",
    r"\bprint\b.{0,20}\b(your instructions|the system prompt)\b",
    r"\bwhat (are|is) your (instructions|system prompt)\b",
    r"\bshow me your (prompt|instructions)\b",
    r"\bexpose\b.{0,25}\b(internal|hidden|secret)\b",
]

_INSTRUCTION_IGNORE = [
    r"\bignore (the |all |any )?(previous|prior|above)\b",
    r"\bdisregard\b",
    r"\bforget (the above|what you know|everything)\b",
    r"\boverride (your |the )?instructions\b",
    r"\bstart from scratch\b",
    r"\bfrom the outset\b",
    r"\breboot\b",
    r"\brelaunch from zero\b",
    r"\bresume from the start\b",
    r"\bbegin anew\b",
    r"\bfrom square one\b",
    r"\bwipe the slate clean\b",
    r"\bclean slate\b",
]

# Длинный непрерывный base64-подобный блок (не короткие случайные совпадения).
_OBFUSCATION_BASE64 = [
    r"(?:[A-Za-z0-9+/]{4}){8,}={0,2}",
]

_ENCODING_SMUGGLING = [
    r"(?:\\u[0-9a-fA-F]{4}){3,}",
    r"(?:%[0-9a-fA-F]{2}){3,}",
    r"(?:\\x[0-9a-fA-F]{2}){3,}",
    r"[​‌‍﻿]",
    r"\bxn--",
]

_ROLEPLAY_JAILBREAK = [
    r"\bDAN\b",
    r"\bjailbreak\b",
    r"\bdo anything now\b",
    r"\bhypothetical scenario\b",
    r"\bfictional story\b",
    r"\brole[- ]?play\b",
    r"\bno restrictions\b",
    r"\bwithout any (restrictions|limitations|filters)\b",
    r"\bas an ai with no\b",
]

_TRANSLATION_WRAPPER = [
    r"\btranslate (the following|this)\b",
    r"\brespond (only )?in (french|spanish|german|chinese|russian|japanese|italian)\b",
    r"\bin (french|spanish|german|chinese|russian|japanese|italian),?\s+(then|and)\b",
    r"\btraduis\b",
    r"\bübersetze\b",
]

_CODE_BLOCK_SMUGGLING = [
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bfunction\s*\w*\s*\(",
    r"console\.log\(",
    r"System\.out\.println",
    r"\bprintln!\(",
]

_HARMFUL_KEYWORDS = (
    r"hack|steal|stealing|exploit|weapon|firearm|bomb|virus|malware|trojan horse|"
    r"kill|murder|poison|drugs online|embezzle|fraud|stalk|harass|counterfeit|"
    r"phishing|launder|traffick|slavery|self-harm|suicide|spy on|drone|"
    r"bypass security|circumvent (government )?(censorship|surveillance)|"
    r"unauthorized access|ransomware|credit card|social security number|"
    r"identity theft|hijack|cyberattack|cyber attack|heist|enter a country illegally|"
    r"classified information|manipulate financial markets|insurance fraud|"
    r"terrorist|government database|break into|human trafficking|"
    r"destroy.{0,20}(computers|data)|lie detector|synthesize a (nerve|toxic)"
)
_IMPERATIVE_START = (
    r"^\s*(write|create|develop|describe|explain|outline|generate|"
    r"give (me )?|provide (me )?|"
    r"tell me how to|design|compose|draft|produce|plan|program|"
    r"teach me|help me|show)\b"
)
_DIRECT_HARMFUL_REQUEST = [
    # императивная формулировка задания + вредоносная лексика в одном тексте
    "__COMPOSITE__",
]

_PATTERN_TABLE: dict[str, list[str]] = {
    "role_override": _ROLE_OVERRIDE,
    "system_prompt_extraction": _SYSTEM_PROMPT_EXTRACTION,
    "instruction_ignore": _INSTRUCTION_IGNORE,
    "obfuscation_base64": _OBFUSCATION_BASE64,
    "encoding_smuggling": _ENCODING_SMUGGLING,
    "roleplay_jailbreak": _ROLEPLAY_JAILBREAK,
    "translation_wrapper": _TRANSLATION_WRAPPER,
    "code_block_smuggling": _CODE_BLOCK_SMUGGLING,
    "direct_harmful_request": _DIRECT_HARMFUL_REQUEST,
}

_COMPILED: dict[str, list[re.Pattern]] = {
    tag: [re.compile(p, _FLAGS) for p in patterns if p != "__COMPOSITE__"]
    for tag, patterns in _PATTERN_TABLE.items()
}
_IMPERATIVE_RE = re.compile(_IMPERATIVE_START, _FLAGS)
_HARMFUL_RE = re.compile(_HARMFUL_KEYWORDS, re.IGNORECASE)


def _base64_hit(text: str) -> bool:
    """Base64-паттерн, отфильтрованный от случайных совпадений на обычных словах."""
    for m in _COMPILED["obfuscation_base64"][0].finditer(text):
        chunk = m.group(0)
        if len(chunk) >= 32 and re.search(r"[0-9]", chunk) and re.search(r"[A-Z]", chunk):
            return True
    return False


def tag_text(text: str) -> list[str]:
    """Вернуть отсортированный список тегов таксономии, найденных в тексте."""
    if not text:
        return []
    hits: set[str] = set()
    for tag, patterns in _COMPILED.items():
        if tag == "obfuscation_base64":
            if _base64_hit(text):
                hits.add(tag)
            continue
        for pat in patterns:
            if pat.search(text):
                hits.add(tag)
                break
    if _IMPERATIVE_RE.search(text) and _HARMFUL_RE.search(text):
        hits.add("direct_harmful_request")
    return sorted(hits)


def tag_batch(texts) -> list[list[str]]:
    return [tag_text(t) for t in texts]
