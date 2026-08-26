"""Детерминированные детекторы форматных PII/PHI сущностей.

Возвращают spans в формате `{'start', 'end', 'label', 'text'}` — том же, что и эталонная
разметка датасета (колонка `spans` в `case 1/data/*.parquet`).

Два класса детекторов, см. `detect_pii()`:

- STANDALONE — сущность узнаётся по одной форме, контекст не нужен: email, IP, MAC,
  номер карты (с проверкой Луна), VIN, UUID, ISO-датавремя, координаты.
- CONTEXT — сущность выглядит как произвольная цифро-буквенная строка и **надёжно узнаётся
  только по подписи поля перед ней** ("Account Number:", "SSN:", "PIN is ..."). Без контекста
  отличить `account_number` от `customer_id` от `medical_record_number` по одному значению
  невозможно — в данных это визуально одинаковые цифровые строки (см. `report.md`, раздел
  «разведка»). Правило контекстного детектора: `<фраза-подпись> <разделитель> <значение>`,
  этот порядок подтверждён на выборке документов.

Порядок применения (`detect_pii`) фиксированный и защищает от двойной разметки одного и того
же куска текста: более специфичные/надёжные детекторы идут первым, следующий не может забрать
уже занятый диапазон.
"""

import re

MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"

# ---------------------------------------------------------------- валидаторы

def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _card_validator(raw: str) -> bool:
    # ВАЖНО (проверено на данных): синтетические номера карт в датасете почти всегда НЕ
    # проходят проверку Луна (только ~16% валидны на выборке 500 doc) — это не настоящие
    # карты, а случайные цифры "под формат". Поэтому Луна не используется как фильтр,
    # только длина/группировка. `_luhn_ok` оставлена в модуле для случая реальных данных.
    digits = re.sub(r"[ -]", "", raw)
    return 13 <= len(digits) <= 19


def _ipv6_validator(raw: str) -> bool:
    # Просто ">=2 двоеточия" совпадает и с обычным временем "19:56:27" (баг, найденный
    # self-audit'ом на реальных данных — ipv6 отжирал время как false positive). Настоящий
    # IPv6 либо содержит "::" сжатие, либо 3+ двоеточия, либо хотя бы одну hex-букву — время
    # никогда не даёт ни одного из этих признаков.
    if raw.count(":") < 2:
        return False
    if "::" in raw:
        return True
    if raw.count(":") >= 3:
        return True
    return any(c in "abcdefABCDEF" for c in raw)


def _vin_validator(raw: str) -> bool:
    return any(c.isdigit() for c in raw) and any(c.isalpha() for c in raw)


def _has_digit(raw: str) -> bool:
    return any(c.isdigit() for c in raw)


# ------------------------------------------------------------ standalone-детекторы
# (label, regex, validator_or_None). validator получает совпавшую строку целиком.

STANDALONE = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), None),
    # Последний символ URL не может быть пунктуацией конца предложения — иначе регекс жадно
    # утаскивает точку/запятую сразу после ссылки в тексте (баг, пойманный на реальных данных).
    ("url", re.compile(r"\b(?:https?|ftp)://[^\s\"'<>]*[^\s\"'<>.,;:!?)\]]"), None),
    (
        "date_time",
        re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
        None,
    ),
    # mac_address ПЕРЕД ipv6: формат MAC (ровно 6 групп по 2 hex-символа) — частный случай,
    # который иначе перехватывает более общий ipv6-паттерн (colon-separated hex), если идёт
    # первым — баг, пойманный на реальных данных (0% recall на mac_address).
    ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), None),
    ("ipv6", re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{0,4}\b"), _ipv6_validator),
    (
        "ipv4",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        None,
    ),
    ("credit_debit_card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), _card_validator),
    ("vehicle_identifier", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), _vin_validator),
    ("swift_bic", re.compile(r"\b[A-Z]{6}[A-Z0-9]{2,6}\b"), _has_digit),
    (
        "unique_id",
        re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
        None,
    ),
    ("coordinate", re.compile(r"\b-?\d{1,3}\.\d{3,8},\s*-?\d{1,3}\.\d{3,8}\b"), None),
    (
        "coordinate",
        re.compile(r"Latitude:\s*-?\d{1,3}\.\d+,\s*Longitude:\s*-?\d{1,3}\.\d+", re.IGNORECASE),
        None,
    ),
]


def _phone_validator(raw: str) -> bool:
    # Отсекаем даты (тот же паттерн "цифры-разделитель-цифры" совпадает с ISO/US датами) и
    # чистые цифровые пробеги без разделителей (это, скорее, account/routing/id-номер —
    # такие сущности размечаются только по контексту, см. CONTEXT ниже).
    if DATE_RE.fullmatch(raw):
        return False
    if not re.search(r"[\-. ()]", raw) and not raw.startswith("+"):
        return False
    digits = re.sub(r"\D", "", raw)
    return 7 <= len(digits) <= 15


# телефон без подписи рядом — распознаётся по разделителям (пробел/дефис/скобки) или
# ведущему "+", как настоящий номер, а не сплошной ID. Применяется ПОСЛЕ контекстных
# детекторов (см. `detect_pii`), а не в общем STANDALONE-проходе: иначе он перехватывает
# SSN/fax_number до того, как более специфичный контекстный детектор успевает их распознать
# (баг, пойманный на реальных данных — 0% recall на ssn/fax_number).
PHONE_STANDALONE = ("phone_number", re.compile(r"(?<!\d)\+?\(?\d[\d\-. ()]{6,16}\d(?!\d)"), _phone_validator)


# -------------------------------------------------------- контекстные детекторы

GENERIC_GAP = None  # маркер: использовать свободный разрыв `[^\n]{0,gap}?`
# Разрыв для меток, где значение по смыслу — присвоение ("password: X", "**API Key** is X"):
# требуем явный разделитель (двоеточие/запятая/is/was, с учётом markdown **жирного**), а не
# произвольные до 20 символов текста. Свободный разрыв на "password" ловит случайные упоминания
# слова в обычном предложении ("...the password should be strong...") как будто "should" — это
# и есть пароль. Проверено на данных: без этого сужения self-audit находит буквальные утечки
# из-за ложных срабатываний детектора на первом проходе, а не из-за проблем подстановки.
ASSIGNMENT_GAP = r"[\s*]{0,3}(?:is|was)[\s*]{0,3}|[\s*]{0,3}[:,][\s*]{0,3}"


def _ctx(keywords: str, value: str, gap: int = 20, delim: str | None = GENERIC_GAP) -> re.Pattern:
    # Флаг IGNORECASE намеренно ограничен ключевой фразой через scoped `(?i:...)` (Python
    # 3.11+), а не наложен на весь паттерн — иначе `[A-Z]{2}`-хвост в ID_VALUE перестаёт
    # отличать реальный код (например, суффикс 'FJ') от случайного строчного слова рядом.
    sep = delim if delim is not None else rf"[^\n]{{0,{gap}}}?"
    return re.compile(rf"(?i:{keywords})(?:{sep})(?P<val>{value})")


# Базовый паттерн значения ID: буквенно-цифровой токен, возможно с дефисами, плюс редкий
# «хвост» из короткой цифровой группы или 2 заглавных букв (см. 'AET-7832-1594-67',
# '4789 215 672 FJ' в данных). Пробел допускается только перед таким коротким хвостом —
# это защищает от захвата случайных английских слов дальше по предложению.
# Первый сегмент обязан содержать цифру: без этого нежадный `[^\n]{0,20}?`-разрыв перед
# значением с радостью "находит" минимальное совпадение на слове-связке ("is", "was") между
# подписью поля и настоящим значением — регекс останавливается на первом успешном варианте,
# а не на самом длинном. Обязательная цифра в первом сегменте убирает эту ловушку на уровне
# грамматики паттерна, а не пост-фильтром (пост-фильтр только отбрасывает, но не даёт движку
# попробовать более длинный разрыв).
# До двух ведущих буквенных сегментов-префиксов допускаются перед цифровым ядром (см.
# 'WA-RN-642819', 'FL-0048921' — код штата/категории перед номером). Обязательный дефис сразу
# после букв — то, что не даёт этому совпасть со словом-связкой в свободном тексте.
ID_VALUE = (
    r"(?:[A-Za-z]{1,4}-){0,2}(?:[A-Za-z0-9]*\d[A-Za-z0-9]*)(?:-[A-Za-z0-9]+)*"
    r"(?:\s\d{1,4})?(?:\s[A-Z]{2})?"
)
PHONE_VALUE = r"\+?[\d][\d\s().-]{6,17}\d"

# http_cookie: значение — цепочка "key=value; key=value; ...". Ключевая идея, найденная на
# реальных данных: свободный `[^\n]{1,100}` жадно утекает в текст предложения ПОСЛЕ cookie
# ("...fx7bk2j9m, will be utilized to manage..."), потому что после значения обычно идёт
# запятая+проза, а не перевод строки. Атрибуты cookie не содержат пробелов, кроме одного
# частного случая — `Expires=` с датой в HTTP-формате ("Sun, 15 Oct 2028 14:30:00 GMT"),
# который матчим отдельной явной веткой. Как только следующий attr не начинается с ';' —
# останавливаемся: проза после cookie никогда не начинается с точки-с-запятой.
_COOKIE_DATE_ATTR = r"(?:Expires|expires)=[A-Za-z]{3},\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+GMT"
_COOKIE_GENERIC_ATTR = r"[\w-]+(?:=[\w./%-]+)?"
# Первый сегмент ОБЯЗАН содержать "=значение" — иначе любое случайное слово в разрыве после
# "cookie" (например "cookie information includes...") само по себе матчится как "имя
# атрибута без значения" и ловится как cookie. Флаги без значения (Secure, HttpOnly) допустимы
# только ВНУТРИ цепочки, после уже подтверждённого key=value.
COOKIE_VALUE = rf"(?:{_COOKIE_DATE_ATTR}|[\w-]+=[\w./%-]+)(?:;\s*(?:{_COOKIE_DATE_ATTR}|{_COOKIE_GENERIC_ATTR}))*"

# (label, regex, требуется_цифра_в_значении)
CONTEXT = [
    ("account_number", _ctx(r"account\s*number|accountNumber", ID_VALUE), True),
    ("customer_id", _ctx(r"customer\s*id", ID_VALUE), True),
    ("employee_id", _ctx(r"employee\s*id", ID_VALUE), True),
    ("medical_record_number", _ctx(r"medical\s*record\s*number|\bMRN\b", ID_VALUE), True),
    (
        "health_plan_beneficiary_number",
        _ctx(r"health\s*plan\s*beneficiary\s*number", ID_VALUE),
        True,
    ),
    ("national_id", _ctx(r"national\s*id", ID_VALUE), True),
    ("tax_id", _ctx(r"tax\s*id(?:entification\s*number)?|\bTIN\b", ID_VALUE), True),
    (
        "certificate_license_number",
        _ctx(r"certificate\s*license\s*number|driver'?s?\s*license\s*number|license\s*number", ID_VALUE),
        True,
    ),
    ("ssn", _ctx(r"social\s*security(?:\s*number)?|\bSSN\b", r"\d{3}-?\d{2}-?\d{4}"), True),
    ("bank_routing_number", _ctx(r"(?:bank\s*)?routing\s*number", r"\d{9}"), True),
    ("pin", _ctx(r"\bPIN\b", r"\d{3,6}"), True),
    ("cvv", _ctx(r"\bCVV\b|security\s*code", r"\d{3,4}"), True),
    # api_key: значение само по себе достаточно специфично (12-48 разнорегистровых
    # алфанум-символов подряд) — обычное слово-связка после "API key" под этот паттерн не
    # подходит по длине/составу, поэтому свободный разрыв безопасен (в отличие от password).
    ("api_key", _ctx(r"api\s*key|apiKey", r"[A-Za-z0-9_.]{12,48}", gap=10), False),
    ("password", _ctx(r"password", r"[^\s,;.\n]{4,30}", delim=ASSIGNMENT_GAP), False),
    ("device_identifier", _ctx(r"device\s*identifier|device\s*id", ID_VALUE), True),
    ("biometric_identifier", _ctx(r"biometric\s*identifier", ID_VALUE), True),
    # license_plate: в отличие от большинства ID, номера часто начинаются с буквенной группы
    # без цифры ("PB 09 YP 3852") — общий ID_VALUE (цифра в первом сегменте) их не берёт.
    # Разрыв нарочно узкий (сразу после "license plate"), поэтому жадный value-паттерн не
    # рискует утечь на соседние слова предложения.
    # Только заглавные буквы/цифры в токенах: реальные номера в данных всегда UPPERCASE, а
    # строчный "хвост" вроде "...are speeding" после жадного value больше не подхватывается.
    (
        "license_plate",
        _ctx(
            r"license\s*plate",
            r"[A-Z0-9]{1,5}(?:[\s-][A-Z0-9]{1,5}){0,4}",
            delim=r"[\s:*]{1,5}",
        ),
        True,
    ),
    (
        "postcode",
        _ctx(
            r"postcode|zip\s*code|postal\s*code",
            r"(?:[A-Za-z0-9]{0,3}\d[A-Za-z0-9]{0,3})(?:\s?[A-Za-z0-9]{2,4})?",
        ),
        True,
    ),
    ("fax_number", _ctx(r"fax(?:\s*number)?", PHONE_VALUE), True),
    ("phone_number", _ctx(r"phone(?:\s*number)?|\btel\b|\bmobile\b", PHONE_VALUE), True),
    ("http_cookie", _ctx(r"cookie", COOKIE_VALUE, gap=40), False),
    ("unique_id", _ctx(r"unique\s*(?:bug\s*)?id", ID_VALUE), True),
]

# -------------------------------------------------------------------- даты

DATE_ALT = (
    rf"\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}-\d{{1,2}}-\d{{4}}|"
    rf"{MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+{MONTH}\s+\d{{4}}"
)
DATE_RE = re.compile(rf"\b(?:{DATE_ALT})\b", re.IGNORECASE)
DOB_CONTEXT_RE = re.compile(r"born|birth|dob", re.IGNORECASE)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?\b")


# -------------------------------------------------------------------- движок

def _overlaps(s: int, e: int, occupied: list[tuple[int, int]]) -> bool:
    return any(s < oe and os < e for os, oe in occupied)


def detect_pii(text: str) -> list[dict]:
    """Прогоняет весь набор детекторов и возвращает несортированный-но-затем-сортированный
    список непересекающихся spans `{'start','end','label','text'}`."""
    occupied: list[tuple[int, int]] = []
    results: list[dict] = []

    def claim(s: int, e: int, label: str) -> None:
        results.append({"start": s, "end": e, "label": label, "text": text[s:e]})
        occupied.append((s, e))

    for label, pattern, validator in STANDALONE:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e, occupied):
                continue
            if validator is not None and not validator(m.group(0)):
                continue
            claim(s, e, label)

    for label, pattern, needs_digit in CONTEXT:
        for m in pattern.finditer(text):
            s, e = m.start("val"), m.end("val")
            val = m.group("val")
            if not val or (needs_digit and not _has_digit(val)):
                continue
            if _overlaps(s, e, occupied):
                continue
            claim(s, e, label)

    phone_label, phone_pattern, phone_validator = PHONE_STANDALONE
    for m in phone_pattern.finditer(text):
        s, e = m.start(), m.end()
        if _overlaps(s, e, occupied):
            continue
        if not phone_validator(m.group(0)):
            continue
        claim(s, e, phone_label)

    for m in DATE_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e, occupied):
            continue
        window = text[max(0, s - 30) : s]
        label = "date_of_birth" if DOB_CONTEXT_RE.search(window) else "date"
        claim(s, e, label)

    for m in TIME_RE.finditer(text):
        s, e = m.span()
        if _overlaps(s, e, occupied):
            continue
        claim(s, e, "time")

    results.sort(key=lambda r: r["start"])
    return results
