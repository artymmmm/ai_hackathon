"""Согласованная псевдонимизация: инвентарь сущностей → детерминированный псевдоним → одна
подстановка по всему документу.

Алгоритм (двухпроходность — согласованность гарантирована по построению, не по удаче):

1. **Инвентарь**: спаны документа группируются в сущности. Основной механизм — точное
   совпадение нормализованного текста в пределах одного лейбла (в датасете имя/номер почти
   всегда повторяется буква-в-букву — проверено на примерах). Дополнительно для персональных
   лейблов (`first_name`/`last_name`/`user_name`) сущности с текстом-подстрокой друг друга
   склеиваются в одну (эвристика на случай "Иванов" / "И. Иванов"; на этом датасете почти
   никогда не задействуется, т.к. повторы буквальные — см. `report.md`).
2. **Псевдоним**: `HMAC(соль_документа, нормализованная_сущность)` → детерминированный индекс
   в пул того же типа (имена/города/категории) либо в format-preserving трансформацию для
   ID/номеров. Один и тот же документ + одна и та же сущность → всегда один и тот же псевдоним;
   разные документы → разная соль → разные псевдонимы (не сквозная деанонимизация по псевдониму
   между документами).
3. **Подстановка**: одним проходом по исходным спанам, справа налево (чтобы не сбивать индексы).
4. **Vault**: обратное отображение `alias -> original` возвращается вызывающей стороне — легитимная
   деанонимизация возможна, если у стороны есть vault и право на неё смотреть.

Format-preserving: телефон/ID остаются тем же классом символов (цифра→цифра, буква→буква,
разделители не трогаем), дата сдвигается на одну и ту же случайную (детерминированную) величину
для всех дат внутри документа — сохраняет относительную хронологию, а не только формат.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import date, datetime, timedelta

# Секрет уровня демо-инсталляции: в проде должен приходить из KMS/секрет-стора, не жить в
# коде. Здесь — константа, чтобы прогон был воспроизводимым без внешней инфраструктуры.
_DEMO_SECRET = "pii-case1-demo-secret-v1"


def doc_salt(doc_id: str) -> str:
    return f"{_DEMO_SECRET}:{doc_id}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _hmac_bytes(salt: str, key: str) -> bytes:
    return hmac.new(salt.encode(), key.encode(), hashlib.sha256).digest()


def _hmac_int(salt: str, key: str, modulo: int) -> int:
    return int.from_bytes(_hmac_bytes(salt, key)[:8], "big") % modulo


# ------------------------------------------------------------------- пулы псевдонимов

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Cameron", "Drew", "Skyler",
    "Elena", "Marco", "Sofia", "Ivan", "Nadia", "Omar", "Layla", "Kenji", "Priya", "Diego",
    "Anya", "Lukas", "Mira", "Noah", "Zara", "Felix", "Wren", "Amara", "Theo", "Yuki",
]
LAST_NAMES = [
    "Whitfield", "Novak", "Petrov", "Alvarez", "Nakamura", "O'Brien", "Kessler", "Duarte",
    "Lindgren", "Haddad", "Fontaine", "Kowalski", "Reyes", "Bergstrom", "Okafor", "Suzuki",
    "Marchetti", "Delgado", "Voss", "Andersson", "Castillo", "Mercer", "Tanaka", "Ivanova",
]
COMPANY_NAMES = [
    "Northbridge Holdings", "Cascade Analytics", "Ironwood Partners", "Silverline Logistics",
    "Bluepeak Consulting", "Harborlight Group", "Meridian Ventures", "Oakstone Financial",
    "Redcliff Systems", "Solace Industries",
]
OCCUPATIONS = [
    "operations analyst", "logistics coordinator", "software engineer", "account manager",
    "field technician", "compliance officer", "customer support specialist", "data analyst",
    "project coordinator", "product designer",
]
CITIES = [
    "Millbrook", "Fairhaven", "Cedar Falls", "Northgate", "Rivermont", "Ashford", "Brookhollow",
    "Stonecrest", "Elmridge", "Harborview",
]
STATES = ["Arcadia", "Meridian", "Cascadia", "Highland", "Westmoor", "Lakeshore", "Sunridge"]
COUNTRIES = ["Freedonia", "Astoria", "Valoria", "Norlandia", "Kestria"]
COUNTIES = ["Ashwood County", "Millbrook County", "Fairview County", "Cedar County"]
STREET_NAMES = ["Maple", "Cedar", "Birch", "Willow", "Elm", "Aspen", "Oak", "Pine", "Sycamore"]
STREET_SUFFIX = ["St", "Ave", "Rd", "Ln", "Ct", "Dr", "Way"]

GENDERS = ["male", "female", "non-binary"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
RACE_ETHNICITY = ["Latino", "Slavic", "Han", "Nordic", "Berber", "Bantu", "Han Chinese", "Celtic"]
RELIGIONS = ["Buddhism", "Lutheranism", "Sunni Islam", "Reform Judaism", "Shinto", "Sikhism"]
POLITICAL_VIEWS = ["Centrist", "Green Party", "Social Democrat", "Libertarian", "Independent"]
SEXUALITIES = ["heterosexual", "bisexual", "asexual", "pansexual", "lesbian", "gay"]
EDUCATION_LEVELS = ["bachelor's degree", "master's degree", "associate degree", "high school diploma"]
EMPLOYMENT_STATUSES = ["full-time", "part-time", "self-employed", "on leave", "retired"]
LANGUAGES = ["Portuguese", "Finnish", "Swahili", "Tagalog", "Hindi", "Polish", "Vietnamese"]
AGES = [str(n) for n in range(19, 76)]

POOLS: dict[str, list[str]] = {
    "first_name": FIRST_NAMES,
    "last_name": LAST_NAMES,
    "company_name": COMPANY_NAMES,
    "occupation": OCCUPATIONS,
    "city": CITIES,
    "state": STATES,
    "country": COUNTRIES,
    "county": COUNTIES,
    "gender": GENDERS,
    "blood_type": BLOOD_TYPES,
    "race_ethnicity": RACE_ETHNICITY,
    "religious_belief": RELIGIONS,
    "political_view": POLITICAL_VIEWS,
    "sexuality": SEXUALITIES,
    "education_level": EDUCATION_LEVELS,
    "employment_status": EMPLOYMENT_STATUSES,
    "language": LANGUAGES,
    "age": AGES,
}

DATE_LABELS = {"date", "date_of_birth", "date_time"}
DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%d %b %Y",
]
DATE_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
]


# ------------------------------------------------------------------- format-preserving

def _format_preserving(value: str, salt: str, key: str) -> str:
    """Заменяет цифры на цифры, заглавные на заглавные, строчные на строчные, всё остальное
    (разделители, пробелы, `@`, `.`, `-`) оставляет как есть. Детерминировано по HMAC."""
    digest = _hmac_bytes(salt, key)
    out = []
    di = 0
    for ch in value:
        b = digest[di % len(digest)]
        if ch.isdigit():
            out.append(str(b % 10))
            di += 1
        elif ch.isupper():
            out.append(chr(ord("A") + b % 26))
            di += 1
        elif ch.islower():
            out.append(chr(ord("a") + b % 26))
            di += 1
        else:
            out.append(ch)
    return "".join(out)


def _luhn_fix_last_digit(digits: str) -> str:
    """Подбирает последнюю цифру так, чтобы вся строка проходила проверку Луна — если уж
    подменяем номер карты, пусть выглядит правдоподобно."""
    body = digits[:-1]
    total = 0
    for i, ch in enumerate(reversed(body)):
        n = int(ch)
        if i % 2 == 0:  # эта цифра станет "второй с конца" после добавления check-digit
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return body + str(check)


def _alias_email(value: str, salt: str, key: str) -> str:
    local, _, domain = value.partition("@")
    if not domain:
        return _format_preserving(value, salt, key)
    idx1 = _hmac_int(salt, key + ":fn", len(FIRST_NAMES))
    idx2 = _hmac_int(salt, key + ":ln", len(LAST_NAMES))
    new_local = f"{FIRST_NAMES[idx1].lower()}.{LAST_NAMES[idx2].lower()}"
    # публичные почтовые провайдеры оставляем как есть (сам домен не идентифицирует человека),
    # кастомные корпоративные домены — заменяем на псевдо-компанию, чтобы не палить организацию.
    public_providers = {"gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "icloud.com"}
    if domain.lower() in public_providers:
        new_domain = domain
    else:
        idx3 = _hmac_int(salt, key + ":dom", len(COMPANY_NAMES))
        slug = COMPANY_NAMES[idx3].lower().replace(" ", "")
        new_domain = f"{slug}.example"
    return f"{new_local}@{new_domain}"


def _alias_street_address(salt: str, key: str) -> str:
    num = 100 + _hmac_int(salt, key + ":num", 899)
    name_idx = _hmac_int(salt, key + ":name", len(STREET_NAMES))
    suf_idx = _hmac_int(salt, key + ":suf", len(STREET_SUFFIX))
    return f"{num} {STREET_NAMES[name_idx]} {STREET_SUFFIX[suf_idx]}"


def _alias_credit_card(value: str, salt: str, key: str) -> str:
    digits_only = re.sub(r"[ -]", "", value)
    new_digits = "".join(str(_hmac_bytes(salt, key + str(i))[0] % 10) for i in range(len(digits_only)))
    new_digits = _luhn_fix_last_digit(new_digits)
    # сохраняем исходную группировку пробелами/дефисами
    out, di = [], 0
    for ch in value:
        if ch.isdigit():
            out.append(new_digits[di])
            di += 1
        else:
            out.append(ch)
    return "".join(out)


def _parse_date(value: str) -> tuple[datetime, str] | None:
    for fmt in DATE_TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt), fmt
        except ValueError:
            continue
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt), fmt
        except ValueError:
            continue
    return None


def _alias_date(value: str, shift_days: int, salt: str, key: str) -> str:
    parsed = _parse_date(value)
    if parsed is None:
        # Неизвестный формат даты — не молчим и не оставляем оригинал как есть (это была бы
        # прямая утечка, поймана self-audit'ом на реальных данных для ISO-datetime с 'Z').
        # Форматосохраняющая замена цифр — надёжный fallback: дата перестаёт быть исходной,
        # даже если мы не смогли её распарсить и осмысленно сдвинуть.
        return _format_preserving(value, salt, key)
    dt, fmt = parsed
    try:
        shifted = dt + timedelta(days=shift_days)
    except OverflowError:
        # Дата у самой границы datetime (в test-сплите такая одна — "0007-03-01" при сдвиге
        # -2990 дней уходит за MINYEAR). Сдвигаем в противоположную сторону: документ теряет
        # знак сдвига на этой одной дате, зато результат остаётся правдоподобной датой, а не
        # форматным мусором вида "9476-98-83", который дал бы _format_preserving.
        try:
            shifted = dt - timedelta(days=shift_days)
        except OverflowError:
            return _format_preserving(value, salt, key)
    return shifted.strftime(fmt)


def _doc_date_shift(salt: str) -> int:
    """Один и тот же сдвиг для всех дат документа — сохраняет их относительную хронологию
    (интервал между датой рождения и датой документа не выдаёт реальные даты, но остаётся
    правдоподобным)."""
    raw = _hmac_int(salt, "__date_shift__", 7300)  # ±10 лет
    return raw - 3650


# ------------------------------------------------------------------- инвентарь сущностей

PERSON_LABELS = {"first_name", "last_name", "user_name"}


def build_inventory(spans: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Группирует spans в сущности. Ключ — (label, нормализованный_канонический_текст)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for sp in spans:
        key = (sp["label"], _normalize(sp["text"]))
        groups.setdefault(key, []).append(sp)

    # склейка персональных сущностей: если текст одной группы — подстрока текста другой группы
    # того же лейбла (разные написания одного имени), объединяем под более длинный вариант.
    keys = sorted(groups, key=lambda k: -len(k[1]))
    merged: dict[tuple[str, str], list[dict]] = {}
    canonical_for: dict[tuple[str, str], tuple[str, str]] = {}
    for k in keys:
        label, norm = k
        if label not in PERSON_LABELS:
            merged[k] = groups[k]
            continue
        target = None
        for existing in merged:
            if existing[0] != label:
                continue
            if norm in existing[1] or existing[1] in norm:
                target = existing
                break
        if target is None:
            merged[k] = list(groups[k])
            canonical_for[k] = k
        else:
            merged[target].extend(groups[k])
            canonical_for[k] = target
    return merged


def _make_alias_once(label: str, canonical_text: str, salt: str, key: str) -> str:
    if label in DATE_LABELS:
        return _alias_date(canonical_text, _doc_date_shift(salt), salt, key)
    if label == "email":
        return _alias_email(canonical_text, salt, key)
    if label == "credit_debit_card":
        return _alias_credit_card(canonical_text, salt, key)
    if label == "street_address":
        return _alias_street_address(salt, key)
    if label in POOLS:
        pool = POOLS[label]
        idx = _hmac_int(salt, key, len(pool))
        return pool[idx]
    if label == "user_name":
        idx1 = _hmac_int(salt, key + ":fn", len(FIRST_NAMES))
        idx2 = _hmac_int(salt, key + ":n", 9999)
        return f"{FIRST_NAMES[idx1].lower()}{idx2 % 99}"
    return _format_preserving(canonical_text, salt, key)


def _make_alias(label: str, canonical_text: str, salt: str) -> str:
    key = f"{label}:{canonical_text}"
    alias = _make_alias_once(label, canonical_text, salt, key)
    # Анти-коллизия: HMAC изредка может случайно воспроизвести исходное значение (проверено
    # на данных — например "3:45"/"3:45" совпало по чистой случайности в format-preserving
    # цифровой замене). Формально это утечка (псевдоним неотличим от оригинала), поэтому при
    # совпадении пересчитываем с "подсоленным" ключом, а не оставляем как есть.
    attempt = 0
    while alias == canonical_text and attempt < 5:
        attempt += 1
        alias = _make_alias_once(label, canonical_text, salt, f"{key}:retry{attempt}")
    return alias


def _dedupe_overlaps(spans: list[dict]) -> list[dict]:
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept: list[dict] = []
    last_end = -1
    for sp in spans:
        if sp["start"] < last_end:
            continue
        kept.append(sp)
        last_end = sp["end"]
    return kept


# Метки, для которых короткое/generic значение слишком рискованно искать "везде в тексте" —
# 3-6-значный PIN/CVV или двузначный возраст может случайно совпасть с посторонним числом
# (ценой, номером пункта и т.п.). Для длинных специфичных ID/имён/дат такой риск пренебрежимо мал.
_NO_REOCCURRENCE_SCAN = {"pin", "cvv", "age", "time"}
_MIN_REOCCURRENCE_LEN = 4


def _close_repeat_occurrences(text: str, spans: list[dict], inventory: dict) -> list[dict]:
    """Если сущность уже опознана (есть в инвентаре), но её точный текст встречается в
    документе ещё раз без "подписи поля" рядом (контекстный детектор такое не ловит) —
    подстановка всё равно должна закрыть все вхождения, иначе кусок исходного PII останется
    в анонимизированном тексте. Явно проверено self-audit'ом на реальных данных (see verify.py)."""
    occupied = [(sp["start"], sp["end"]) for sp in spans]
    extra: list[dict] = []
    for cluster_key, group in inventory.items():
        label = cluster_key[0]
        if label in _NO_REOCCURRENCE_SCAN:
            continue
        for value in {sp["text"] for sp in group}:
            if len(value) < _MIN_REOCCURRENCE_LEN:
                continue
            for m in re.finditer(re.escape(value), text):
                s, e = m.span()
                if any(s < oe and os < e for os, oe in occupied):
                    continue
                extra.append({"start": s, "end": e, "label": label, "text": value, "_cluster": cluster_key})
                occupied.append((s, e))
    return extra


def anonymize(text: str, spans: list[dict], salt: str) -> tuple[str, list[dict]]:
    """Двухпроходная анонимизация. Возвращает (анонимизированный_текст, vault).

    `vault` — список записей `{label, alias, original_values}` для легитимной деанонимизации
    (обратное отображение `alias -> original_values`).
    """
    spans = _dedupe_overlaps(spans)
    inventory = build_inventory(spans)

    span_to_cluster: dict[int, tuple[str, str]] = {}
    for cluster_key, group in inventory.items():
        for sp in group:
            span_to_cluster[id(sp)] = cluster_key

    # третий под-проход: закрываем повторные вхождения уже известных сущностей, которые
    # контекстные детекторы не поймали бы сами по себе (нет подписи поля рядом).
    extra_spans = _close_repeat_occurrences(text, spans, inventory)
    for sp in extra_spans:
        span_to_cluster[id(sp)] = sp.pop("_cluster")
    all_spans = spans + extra_spans

    extra_by_cluster: dict[tuple[str, str], int] = {}
    for e in extra_spans:
        k = span_to_cluster[id(e)]
        extra_by_cluster[k] = extra_by_cluster.get(k, 0) + 1

    alias_cache: dict[tuple[str, str], str] = {}
    vault: list[dict] = []
    for cluster_key, group in inventory.items():
        label = group[0]["label"]
        canonical = max((sp["text"] for sp in group), key=len)
        alias = _make_alias(label, canonical, salt)
        alias_cache[cluster_key] = alias
        vault.append(
            {
                "label": label,
                "alias": alias,
                "original_values": sorted({sp["text"] for sp in group}),
                "occurrences": len(group) + extra_by_cluster.get(cluster_key, 0),
            }
        )

    replacements = []
    for sp in all_spans:
        cluster_key = span_to_cluster[id(sp)]
        alias = alias_cache[cluster_key]
        replacements.append((sp["start"], sp["end"], alias))

    replacements.sort(key=lambda r: r[0], reverse=True)
    out = text
    for s, e, alias in replacements:
        out = out[:s] + alias + out[e:]
    return out, vault
