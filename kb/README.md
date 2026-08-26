# База знаний для конфигурации A (кейс 3)

Источники доменного знания, не зависящие от размеченного корпуса — то есть переносимые
на любой чужой код. Именно этим конфигурация A отличается от B (retrieval по датасету).

| Файл | Что это | Как получен |
|---|---|---|
| `cwec_v4.20.xml` | Полный каталог MITRE CWE, 969 записей, версия 4.20 от 30.04.2026 | `curl -sSL -o kb/cwec_latest.xml.zip https://cwe.mitre.org/data/xml/cwec_latest.xml.zip && unzip` |
| `cert_c_rules.json` | 109 правил CERT C Secure Coding с привязкой к CWE | извлечено из `Taxonomy_Mapping` каталога CWE |

XML в git не кладётся (17 МБ) — качается заново командой выше. `cert_c_rules.json` небольшой
и лежит в репозитории.

## Что внутри каталога CWE

Namespace `http://cwe.mitre.org/cwe-7`, элементы `Weakness` с атрибутами `ID` и `Name`.
Полезные подэлементы: `Description`, `Extended_Description`, `Common_Consequences`,
`Potential_Mitigations`, `Demonstrative_Examples`, `Observed_Examples`,
`Related_Weaknesses` (связи между смежными CWE — нужны для различения пар 119/787/125/120),
`Applicable_Platforms` (фильтр по языку).

## Статический анализатор

`flawfinder` установлен в `.venv` (`uv pip install flawfinder`). Разбирает код как текст,
не компилируя и не запуская — соответствует ограничению задания. Выдаёт CWE прямо в выводе:

```
.venv/bin/flawfinder --quiet --dataonly --singleline <файл.c>
```

Проверено: на синтетическом `strcpy`/`sprintf` находит CWE-120 и CWE-119.
На реальном фрагменте корпуса нередко не находит ничего — это согласуется с измеренным
recall сигнатурного триажа (10.5%): уязвимости в реальных CVE обычно не сигнатурные.
