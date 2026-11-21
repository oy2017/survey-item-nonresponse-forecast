# Forecasting survey item nonresponse with one language-model call per question

Code and data for the paper *One Call per Question: Language Models Forecast Which Survey Questions Draw "Don't Know"
Answers* (Owen Yang, Jin Huang, Qiaozhu Mei).

**Item nonresponse** is when a person who takes part in a survey gives no substantive answer to one question (an
*item*), for example by saying "don't know". The paper forecasts, for each question of a survey, the share of people
who will say "don't know", "can't choose" or the combined "don't know / no answer", from one call to a language model
that sees only the question, its answer options, the population and the survey mode. It compares the forecast with
the real shares on 908 questions from the World Values Survey, the General Social Survey and Eurobarometer.

## What is here

| Folder | Contents |
|---|---|
| `data/questions.csv` | the 908 questions: set, survey, mode, block, wording, answer options, real "don't know" share, number asked |
| `data/prompts_main.csv` | the exact system and user message sent for each question |
| `data/forecasts_main.csv` | every forecast, one row per question, model, setting and call |
| `data/replies/` | the raw model replies the forecasts were read from |
| `analysis/` | one script per section or table of the paper; each prints the paper's numbers |
| `build/` | scripts that rebuild `data/` from the original sources (survey files and published reports) |
| `results/` | the saved output of every analysis script; `results/README.md` maps each to the paper's sections and tables |
| `check_numbers.py` | reruns every analysis script and reports any line that differs from `results/` |

No script in `analysis/` or `build/` calls a language model. Recomputing the paper needs no API key.

## The question table (`data/questions.csv`)

| Column | Meaning |
|---|---|
| `qid` | question id (`wvs7:Q1`, `gss2022:natchld`, `gss24_int:acpttg`, `gss24_web:aidrive`, `eb:STD105:p21` = survey and annex page) |
| `set` | `wvs7`, `gss2022`, `gss2024_interview`, `gss2024_web`, `eb_face_to_face`, `eb_web`, `eb_telephone` |
| `programme`, `survey`, `mode` | survey programme, survey or wave, and how it was fielded |
| `block` | question block: questions sharing one stem and answer scale; the bootstrap resamples whole blocks |
| `new_question` | Eurobarometer face-to-face only: 1 if the annex does not mark the question as asked in an earlier wave |
| `question_text`, `options_text` | the question and answer options exactly as shown to the model |
| `n_options` | number of substantive answer options |
| `real_dk_pct` | real share answering "don't know" (or "can't choose", or "don't know / no answer"), in percent |
| `n_asked` | number of respondents asked (WVS and GSS; Eurobarometer annexes do not print it per question) |
| `question_chars` | characters in the question line: the paper's "question length" baseline |
| `prompt_chars` | characters in the whole user message |

Forecasts are in `data/forecasts_main.csv` with one row per question, model, setting and call (`draw`); the paper's
forecast is the mean over a model's calls. `data/replies/main_forecast_replies.jsonl` holds the reply each one was read
from (the last bracketed number in the reply).

## Reproducing the paper

```
python3.15 -m venv .venv          # tested with Python 3.15.0
.venv/bin/pip install -r requirements.txt
.venv/bin/python check_numbers.py
```

Scripts that use pilots drawn from real interviews (sections 5 and 7) or rebuild the real shares need the
respondent-level files described below; the others run from `data/` alone.

## Sources and terms of use

| Programme | What we use | Where to get it | In this repo |
|---|---|---|---|
| World Values Survey, wave 7 | respondent-level data file, version 6.0 (`WVS_Cross-National_Wave_7_csv_v6_0.csv`) | worldvaluessurvey.org (free registration) | question-level shares only; the data file may not be redistributed |
| General Social Survey 2022, 2024 | NORC data files (2022 Release 4; 2024 Release 3a), Stata format | gss.norc.org | question-level shares only |
| Eurobarometer, 22 surveys | published data annexes (PDF), October 2025 to October 2026 | europa.eu/eurobarometer; see `build/eurobarometer_sources.csv` | question-level shares and wording, with credit to the European Commission and the European Parliament |

Credits and conditions:

- **World Values Survey.** The data files are free for non-profit use and may not be redistributed; this repository
  holds only shares computed from them. Data source: Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K.,
  Diez-Medrano, J., Lagos, M., Norris, P., Ponarin, E. and Puranen, B. (eds.) (2022). World Values Survey Wave 7
  (2017-2022), cross-national data file, version 6.0. Madrid and Vienna: JD Systems Institute and WVSA Secretariat.
  doi:10.14281/18241.24
- **General Social Survey.** Davern, M., Bautista, R., Freese, J., Herd, P. and Morgan, S. L. *General Social Survey
  1972-2022* (2022 Release 4) and *General Social Survey 1972-2024* (2024 Release 3a). Chicago: NORC.
- **Eurobarometer.** Question wording and "don't know" shares are taken from the published data annexes; each survey's
  title and DOI are in `build/eurobarometer_sources.csv`. Surveys of the European Commission: © European Union,
  2025-2026, reused under the Commission's reuse policy (Decision 2011/833/EU; CC BY 4.0). Surveys of the European
  Parliament (EB049EP, EB050EP, FL015EP): © European Union, 2025-2026 – Source: European Parliament.

## Respondent-level data

Put the downloaded files in `raw/` (git-ignored) under these names: `WVS_Cross-National_Wave_7_csv_v6_0.csv`,
`GSS2022.dta`, `GSS2024.dta`, and the Eurobarometer annexes as `raw/eurobarometer/<code>_data_annex_en.pdf`.
`build/real_shares.py` then recomputes every real share in `data/questions.csv` and reports any difference;
`build/eurobarometer.py` rebuilds every Eurobarometer question, share and prompt from the annex PDFs (pypdf 6.20.1);
`build/ways_of_asking_profiles.py` rebuilds the simulated-respondent profiles into `raw/derived/` (never committed).
`build/hand_check_eurobarometer.csv` records 20 Eurobarometer values checked by hand against the published pages.

## Licence

The code (`analysis/`, `build/`, `check_numbers.py`) is under the MIT licence in `LICENSE`. The survey wording and
shares in `data/` and `results/` come from the sources above and remain under their terms.

## Citation

(to be added with the preprint)
