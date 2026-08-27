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
    # ipv6: без "::"-альтернативы движок не может перепрыгнуть compression (нужен hex-символ
    # между каждой парой двоеточий) и режет сжатый адрес на два отдельных совпадения — баг,
    # пойманный на реальных данных (каждый FN сопровождался ровно двумя FP от этого разреза).
    # Первая альтернатива обязана требовать буквальный "::" на стыке (`{1,7}` группа кончается
    # двоеточием, следом ещё один буквальный ":" от `(?::...)+ `), иначе она перехватывает и
    # обычные "12:34:56"-подобные строки времени без всякого сжатия.
    (
        "ipv6",
        re.compile(
            r"\b(?:[0-9A-Fa-f]{1,4}:){1,7}(?::[0-9A-Fa-f]{1,4})+\b"
            r"|\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{0,4}\b"
        ),
        _ipv6_validator,
    ),
    (
        "ipv4",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        None,
    ),
    ("vehicle_identifier", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), _vin_validator),
    # postcode (UK-формат "TR14 8BE"): единственный формат почтового индекса в данных, который
    # опознаётся без подписи-контекста — форма (буквы+цифры, пробел, цифра+2 буквы) достаточно
    # редкая, чтобы не пересекаться со случайным текстом. Остальные (голые 5-значные US ZIP в
    # прозе адреса без слова "postcode/zip") намеренно не тронуты — неотличимы от произвольных
    # чисел без адресного парсинга, см. `out/pii/detectors_diagnostics.json`.
    ("postcode", re.compile(r"\b[A-Z]{1,2}\d[A-Z0-9]?\s\d[A-Z]{2}\b"), None),
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


# credit_debit_card: вытащен из общего STANDALONE-прохода и применяется ПОСЛЕ CONTEXT (см.
# `detect_pii`) — иначе его generic-паттерн "любые 13-19 цифр" перехватывает account_number,
# device_identifier, national_id и т.п. раньше, чем более специфичный контекстный детектор
# успевает распознать подписанное поле (баг, пойманный на реальных данных — 9 из 15 FP на
# n=200 были ровно этим перехватом). Луна как фильтр НЕ применяется: на выборке n=1000 только
# 11.6% эталонных номеров карт проходят проверку (синтетические данные), фильтр по Луну убил
# бы recall почти вдвое.
# Паттерн переписан так, чтобы каждое повторение обязательно кончалось цифрой — старый вариант
# `(?:\d[ -]?){13,19}` мог захватить висящий разделитель после последней цифры, если дальше по
# тексту не было ещё одной цифры (баг: "4928 7456 2319 5786 for" матчился с хвостовым
# пробелом — off-by-one против эталонной границы).
CREDIT_CARD_STANDALONE = (
    "credit_debit_card",
    re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"),
    _card_validator,
)


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
# password: расширенный разделитель — та же логика (не свободный разрыв), но с тремя формами,
# найденными в данных и не покрытыми ASSIGNMENT_GAP: "password, for example X" (запятая+фраза),
# "| Password | X |" (markdown-таблица, пайп) и голое "the password X" (только пробел, без
# двоеточия/is/was). Последняя форма структурно совпадает с опасным случаем "the password should
# be strong" — защищена не разделителем, а PASSWORD_VALUE (см. ниже): требует цифру/символ в
# значении, которого у обычных английских слов нет.
PASSWORD_GAP = (
    rf"{ASSIGNMENT_GAP}"
    r"|[\s*]{0,3},\s*for\s+example[\s*]{0,3}"
    r"|[\s*]{0,3}\|[\s*]{0,3}"
    r"|[\s*]{1,3}"
)
# Значение обязано содержать хотя бы один не-буквенный символ (цифру/спецсимвол) — это то, что
# отличает реальный пароль ("River99#") от случайного слова-связки ("should") при голом
# пробельном разделителе из PASSWORD_GAP.
PASSWORD_VALUE = r"(?=[^\s,;.\n]*[^A-Za-z\s,;.\n])[^\s,;.\n]{4,30}"


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

# account_number: IBAN-формат ("NL37 RABO 1234 5678 90", "FR76 3000 6000 0112 3456 789") —
# несколько пробел-разделённых групп по 2-4 буквенно-цифровых символа. ID_VALUE рассчитан на
# ОДИН короткий пробельный хвост и обрезает такие значения после первой группы (баг, пойманный
# на реальных данных). Отдельный паттерн только для account_number — не трогаем общий ID_VALUE,
# которым пользуются остальные CONTEXT-типы (national_id, customer_id и т.д.), не разобранные
# в этой задаче.
IBAN_VALUE = r"[A-Z]{2}\d{2}(?:\s[A-Z0-9]{2,4}){2,8}"

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
    # "bank account X" (без слова "number" рядом) — отдельная фраза, встречается в данных
    # наравне с "account number is X"; добавлена как алиас с тем же needs_digit-фильтром.
    (
        "account_number",
        _ctx(r"account\s*number|accountNumber|bank\s*account", rf"(?:{IBAN_VALUE}|{ID_VALUE})"),
        True,
    ),
    # "user id"/"user_id"/"userID" в данных размечены тем же лейблом, что и "customer id"
    # (те же значения, разная подпись поля — таблицы/JSON-ключи используют "user").
    ("customer_id", _ctx(r"customer\s*id|user[\s_]*id", ID_VALUE, gap=40), True),
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
    # Последний символ не может быть точкой — иначе конец предложения ("...key is X.") утекает
    # в значение (баг, пойманный на реальных данных). gap поднят 10->40: реальная фраза "API key
    # used for production environment is X" даёт ~35 символов между ключевым словом и значением.
    ("api_key", _ctx(r"api\s*key|apiKey", r"[A-Za-z0-9_.]{11,47}[A-Za-z0-9_]", gap=40), False),
    ("password", _ctx(r"password", PASSWORD_VALUE, delim=PASSWORD_GAP), False),
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
    # gap поднят 20->40: "please fax the necessary documentation to X" — 32 символа между
    # ключевым словом и значением; при узком gap fax/phone теряли контекстное совпадение, и
    # генерик-детекторы (credit_debit_card/PHONE_STANDALONE) подхватывали значение раньше и
    # без "+"-префикса (баг, пойманный на реальных данных — рост FP у phone_number).
    ("fax_number", _ctx(r"fax(?:\s*number)?", PHONE_VALUE, gap=40), True),
    ("phone_number", _ctx(r"phone(?:\s*number)?|\btel\b|\bmobile\b", PHONE_VALUE, gap=40), True),
    ("http_cookie", _ctx(r"cookie", COOKIE_VALUE, gap=40), False),
    ("unique_id", _ctx(r"unique\s*(?:bug\s*)?id", ID_VALUE), True),
]

# -------------------------------------------------------------------- даты

WEEKDAY = r"(?:Mon|Tue|Tues|Wed|Thu|Thurs|Fri|Sat|Sun)[a-z]*,?\s+"
# \d{1,2}\.\d{1,2}\.\d{4} (точка-разделитель, "15.07.2024") добавлена отдельной веткой: без неё
# такие даты не распознаёт ни DATE_RE, ни (что важнее) исключение внутри `_phone_validator` —
# PHONE_STANDALONE перехватывал их как телефон раньше, чем сюда доходила очередь (баг, пойманный
# на реальных данных). Необязательный `WEEKDAY`-префикс — гэлд размечает "Wed, 15 Oct 2024"
# ОДНИМ спаном вместе с днём недели, а не отдельно.
DATE_ALT = (
    rf"(?:{WEEKDAY})?(?:"
    rf"\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{1,2}}-\d{{1,2}}-\d{{4}}|"
    rf"\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|"
    rf"{MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+{MONTH}\s+\d{{4}})"
)
DATE_RE = re.compile(rf"\b(?:{DATE_ALT})\b", re.IGNORECASE)
DOB_CONTEXT_RE = re.compile(r"born|birth|dob", re.IGNORECASE)
# Два бага, пойманных на реальных данных: (1) `\s*` перед необязательной группой AM/PM жадно
# заглатывает пробел/перевод строки, ДАЖЕ когда AM/PM дальше нет ("18:30 for" -> "18:30 " с
# висящим пробелом) — вынесен ВНУТРЬ необязательной группы, чтобы съедался только вместе с ней.
# (2) хвостовая точка в "p.m."/"a.m." никогда не входила в совпадение: `\b` после необязательной
# `\.?` требует границы слово/не-слово, а точка перед пробелом/переводом строки/пайпом даёт
# не-слово/не-слово — не граница. Заменено на `(?!\d)`, которая делает ровно то же самое (не
# срастись со следующей цифрой), но не блокирует финальную точку.
# Меридием — две отдельные ветки, а не общий "точки не обязательны": "p.m." (точка после
# каждой буквы, последняя опциональна — по ней же и восстанавливаем финальную точку) и "pm" без
# единой точки. Раньше общий `\.?...\.?` мог доесть точку конца предложения после голого "pm"
# ("8:30pm." -> "8:30pm." вместо эталонных "8:30pm") — теперь опциональная точка разрешена
# только когда паттерн уже "в режиме точек" (после первой буквы стоит точка).
_MERIDIEM = r"(?:[APap]\.[Mm]\.?|[APap][Mm])"
# Секунды — опциональная дробная часть (".123"); офсет часового пояса ("+02:00"/"-0500") —
# отдельная опциональная хвостовая группа для ISO-подобных отметок времени с зоной.
TIME_RE = re.compile(
    rf"\b\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?(?:\s*{_MERIDIEM})?(?:[+-]\d{{2}}:?\d{{2}})?(?!\d)"
    rf"|\b\d{{1,2}}\s*{_MERIDIEM}\b"  # голый час без минут: "9 AM"
)
# Разделитель для диапазона времени ("00:00 - 00:10", "9:00 AM to 11:00 AM") — эталон размечает
# диапазон ОДНИМ спаном; используется в `detect_pii` для склейки двух соседних TIME_RE-совпадений.
TIME_RANGE_SEP = re.compile(r"\s*(?:-|to)\s*")


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

    card_label, card_pattern, card_validator = CREDIT_CARD_STANDALONE
    for m in card_pattern.finditer(text):
        s, e = m.start(), m.end()
        if _overlaps(s, e, occupied):
            continue
        if not card_validator(m.group(0)):
            continue
        claim(s, e, card_label)

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
        # диапазон времени ("00:00 - 00:10") эталон размечает одним спаном — если сразу после
        # найденного времени идёт "-"/"to" и ещё одно время, склеиваем в один span.
        sep = TIME_RANGE_SEP.match(text, e)
        if sep:
            m2 = TIME_RE.match(text, sep.end())
            if m2:
                e = m2.end()
        claim(s, e, "time")

    results.sort(key=lambda r: r["start"])
    return results
