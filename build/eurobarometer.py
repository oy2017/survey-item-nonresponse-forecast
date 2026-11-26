"""Rebuild the 502 Eurobarometer questions of data/questions.csv (sets eb_face_to_face 374, eb_web 101, eb_telephone 27)
and their exact prompts in data/prompts_main.csv from the 22 published data annexes (PDF), then compare with both tables.
Section 3 and Appendix C of the paper.

Input:  raw/eurobarometer/<code>_data_annex_en.pdf for the 22 survey codes below (download them from
        europa.eu/eurobarometer; build/eurobarometer_sources.csv lists them with the sha256 of the files we used).
Output: a temporary folder (or --out FOLDER) with the text of each annex (<code>_annex.txt), every table read from the
        annexes (tables_read.csv), and the rebuilt rows (questions_eurobarometer.csv, prompts_eurobarometer.csv).
        The script prints how many rows reproduce and every difference as (qid, field, old, new).
Needs pypdf 6.20.1: other versions may lay the page text out differently.
Usage: python build/eurobarometer.py [--out FOLDER]

Steps
1. Text. pypdf's extract_text() for every page. Two web annexes (FL576, FL585) print each country code on its own line;
   those lines are joined to the numbers that follow, and the ligature "ﬅ" is written "ft".
2. Tables. Each annex page holds at most one results table with an EU27 row. A page is kept only when the table can be
   read without guessing: exactly one "don't know" answer, every answer label lines up with one EU27 number, and the
   shares add to 98-102 (single-answer questions). Two layouts:
   - "table" (Standard and Special surveys and some Flash surveys): answer labels above the EU27 row, the question text
     below the country rows. If the table has a "Diff." column (change since the earlier wave), the question was asked in
     an earlier wave ("repeated"); otherwise it is new in this wave.
   - "flash" (some Flash surveys): the question above, all answer labels on one line, "don't know" last. Whether the
     question was asked before is not printed.
   Questions whose code ends in T, R or F after a digit (summary, recode and filter tables) are dropped, then each wording
   is kept once per survey (first page).
3. Prompts. The frozen prompt wording, one for interviews (face-to-face, telephone) and one for web surveys.
4. Cleaning. Rows whose question text has table numbers swept into it (the second page of a table that repeats the EU27
   row) are dropped. Each question is then counted once across surveys: same wording (letters only, lower case) and same
   mode; the first one kept is the earliest in the survey order below.

The questions were collected in three rounds (the round number is the prefix of the question id: eb, eb2, eb3). The rounds
differ in three small ways, kept here so the output is identical: rounds 2 and 3 drop rows with a negative EU27 number and
read "_" in a question code as "." when forming the question block; round 3 is the joined-line text of step 1 and puts
"; " between answer labels.
"""
import argparse, hashlib, re, sys, tempfile, unicodedata
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "eurobarometer"
SOURCES = ROOT / "build" / "eurobarometer_sources.csv"
PYPDF_VERSION = "6.20.1"

# (code, round, mode) in the order the questions were collected; the order decides which copy of a repeated question is kept.
SURVEYS = [(c, 1, "interview") for c in ["STD105", "SP573", "SP572", "SP578", "SP574"]] + \
          [("FL015EP", 1, "web"), ("FL584", 1, "web"), ("EB050EP", 1, "interview"), ("SP579", 1, "interview"),
           ("SP577", 1, "interview"), ("SP569", 1, "interview"), ("FL577", 1, "web"), ("SP571", 1, "interview"),
           ("SP570", 1, "interview"), ("FL567", 1, "phone"),
           ("SP567", 2, "interview"), ("STD104", 2, "interview"), ("EB049EP", 2, "interview"), ("FL564", 2, "web"), ("FL566", 2, "phone"),
           ("FL576", 3, "web"), ("FL585", 3, "web")]
JOINED_LINES = {"FL576", "FL585"}  # annexes that print each country code on its own line
ID_PREFIX = {1: "eb", 2: "eb2", 3: "eb3"}
SET_AND_MODE = {"interview": ("eb_face_to_face", "interviewer-run (face-to-face)"), "web": ("eb_web", "web, don't know shown"),
                "phone": ("eb_telephone", "telephone")}

# ---- step 1: text --------------------------------------------------------------------------------------------------------
def annex_text(code):
    from pypdf import PdfReader
    reader = PdfReader(RAW / f"{code}_data_annex_en.pdf")
    text = "".join(f"=====PAGE {i + 1}=====\n" + (page.extract_text() or "") + "\n" for i, page in enumerate(reader.pages))
    if code in JOINED_LINES:
        text = text.replace("ﬅ", "ft")
        text = re.sub(r"(?m)^(EU27|[A-Z]{2})\n[ \t]+(?=-?\d)", r"\1 ", text)  # "EU27\n  30 31 .." -> "EU27 30 31 .."
    return text

# ---- step 2: tables ------------------------------------------------------------------------------------------------------
DATE_LINE = re.compile(r"^(Diff\.|Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sep|Oct|Nov|Dec|Spring|Autumn|Winter|Summer|EB\s?\d|\d{4}$)", re.I)
COUNTRY_ROW = re.compile(r"^(EU27|EU 27|NON EURO|EURO AREA|[A-Z]{2})\s+(-?\d+\s*)+$")
QUESTION_CODE = re.compile(r"\b(Q[A-Z]{0,2}\d+[a-zA-Z]?(?:[._]\d+[a-z]?)?|SD\d+[a-z]?(?:\.\d+)?|D\d+[a-z]?(?:\.\d+)?)\b")
DONT_KNOW = re.compile(r"don.?t know|^dk\b", re.I)
TOTAL_LABEL = re.compile(r"^(total|net|at least|sum)\b", re.I)
TITLE_LINE = re.compile(r"^(special|standard|flash) eurobarometer|^\d+$|^[A-Z]{2}$|eurobarometer", re.I)

def read_tables(code, text):
    """One row per page whose table passes the checks in step 2."""
    rows = []
    for page in text.split("=====PAGE ")[1:]:
        page_no, body = page.split("=====\n", 1)
        lines = [l.strip() for l in unicodedata.normalize("NFKC", body).replace("’", "'").split("\n") if l.strip()]
        eu = next((i for i, l in enumerate(lines) if re.match(r"^EU ?27\s+-?\d", l)), None)  # the EU27 row
        if eu is None: continue
        numbers = [int(x) for x in re.findall(r"-?\d+", re.sub(r"^EU ?27", "", lines[eu]))]
        if not DONT_KNOW.search(" ".join(lines)): continue
        head = lines[:eu]
        has_diff = any(l.startswith("Diff.") for l in head)
        # a "don't know" label broken over two lines ("Don't" / "know ...") is put back together
        if len(head) > 2 and DONT_KNOW.search(head[-1]) and head[-1][:1].islower() and not has_diff:
            head = head[:-2] + [head[-2] + " " + head[-1]]
        flash = bool(DONT_KNOW.search(head[-1])) and len(head[-1].split()) > 4 and not has_diff
        row = dict(survey=code, page=int(page_no))
        if flash:  # all labels on the last line above the EU27 row, "don't know" last
            labels = head[-1]; dk = re.search(r"(don.?t know.*)$", labels, re.I)
            if not dk or TOTAL_LABEL.search(labels) or not 98 <= sum(numbers) <= 102: continue
            start = next((i for i, l in enumerate(head) if re.match(r"^Q\d", l)), None)  # question text starts at "Q<digit>"
            if start is None: continue
            question = " ".join(head[start:-1]); qcode = QUESTION_CODE.search(question)
            row.update(layout="flash", trend=None, qcode=qcode.group(1) if qcode else None, question=QUESTION_CODE.sub("", question, count=1).strip(),
                       options=labels[:dk.start()].strip(), dk_label=dk.group(1), dk_eu27=numbers[-1], n_opts=len(numbers) - 1)
        else:  # one label per line above the EU27 row, question text below the country rows
            labels = [l for l in head if not DATE_LINE.match(l) and not TITLE_LINE.match(l)]
            values = numbers[::2] if has_diff else numbers  # with a "Diff." column every second number is the change
            if (has_diff and len(numbers) % 2) or len(labels) != len(values): continue
            dk_at = [i for i, l in enumerate(labels) if DONT_KNOW.search(l)]
            answers = [(l, v) for l, v in zip(labels, values) if not TOTAL_LABEL.match(l)]  # drop "Total ..." summary columns
            if len(dk_at) != 1 or not 98 <= sum(v for _, v in answers) <= 102: continue
            below = [l for l in lines[eu:] if not COUNTRY_ROW.match(l) and not TITLE_LINE.match(l) and not re.match(r"^(-?\d+\s*)+$", l)]
            below = " ".join(below); qcode = QUESTION_CODE.search(below)
            question = re.sub(r"\s*\(%\).*$", "", QUESTION_CODE.sub("", below, count=1)).strip()
            question = re.sub(r"^(?:(?:NON EURO|EURO AREA|EURO|TCC|[A-Z]{2})\s+(?:(?:-?\d+|-)\s+)+)+", "", question).strip()  # leftover country rows
            if len(question) < 15 or qcode is None: continue
            row.update(layout="table", trend=has_diff, qcode=qcode.group(1), question=question,
                       options="; ".join(l for l, _ in answers if not DONT_KNOW.search(l)), dk_label=labels[dk_at[0]], dk_eu27=values[dk_at[0]], n_opts=len(answers) - 1)
        rows.append(row)
    return pd.DataFrame(rows)

# ---- step 3: prompts -----------------------------------------------------------------------------------------------------
SYSTEM_INTERVIEW = ("You are a survey methodologist estimating item non-response in live household interviews. "
                    "Reply with ONLY one percentage between 0.0 and 100.0 inside square brackets, such as [0.6] or [9.5].")
SYSTEM_WEB = ("You are a survey methodologist estimating item non-response in self-administered web surveys. "
              "Reply with ONLY one percentage between 0.0 and 100.0 inside square brackets, such as [0.6] or [9.5].")
EU_NOTE = " Results are for the European Union as a whole; '(OUR COUNTRY)' or '(NATIONALITY)' in a question stands for the respondent's own country."

def prompt(mode, question, options):
    if mode in ("interview", "phone"):
        return SYSTEM_INTERVIEW, (
            "Population and mode: a " + ("face-to-face" if mode == "interview" else "telephone") +
            " interview survey of adults in the 27 member states of the European Union." + EU_NOTE +
            " Interviewers read the question and substantive choices aloud; 'Don't know' is NOT read aloud, but recorded if volunteered by the respondent."
            f"\n\nQuestion: {question}\nSubstantive options: {options}\n\n"
            "What percentage (0.0 to 100.0) of respondents asked this question volunteer 'Don't know'? Output ONLY one number in brackets, e.g. [2.4].")
    return SYSTEM_WEB, (
        "Population and mode: a self-administered web survey of adults in the 27 member states of the European Union." + EU_NOTE +
        " Respondents read the question on screen, and 'Don't know' is explicitly displayed in the list of answer options."
        f"\n\nQuestion: {question}\nOn-screen options (in order): {options}; Don't know\n\n"
        "What percentage (0.0 to 100.0) of respondents asked this question select 'Don't know'? Output ONLY one number in brackets, e.g. [2.4].")

# ---- run -----------------------------------------------------------------------------------------------------------------
def build(out):
    import pypdf
    if pypdf.__version__ != PYPDF_VERSION: print(f"warning: pypdf {pypdf.__version__}; the tables were built with {PYPDF_VERSION}")
    sources = pd.read_csv(SOURCES).set_index("code") if SOURCES.exists() else None
    kept = []
    for code, rnd, mode in SURVEYS:
        pdf = RAW / f"{code}_data_annex_en.pdf"
        if not pdf.exists(): sys.exit(f"missing {pdf.relative_to(ROOT)}: download the data annex (see {SOURCES.relative_to(ROOT)})")
        if sources is not None and hashlib.sha256(pdf.read_bytes()).hexdigest() != sources.loc[code, "sha256"]:
            print(f"warning: {code}: this PDF differs from the one used for the paper (sha256)")
        text = annex_text(code); (out / f"{code}_annex.txt").write_text(text)
        t = read_tables(code, text); t.to_csv(out / f"{code}_tables.csv", index=False)
        t = t[~t.qcode.fillna("").str.contains(r"\d[TRF]$")]  # summary, recode and filter tables
        if rnd == 3: t = t.assign(options=t.options.str.replace(r" (?=[A-Z])", "; ", regex=True))
        if rnd >= 2: t = t[t.dk_eu27 >= 0]
        t = t.drop_duplicates("question").copy()  # each wording once per survey
        code_part = t.qcode.str.replace("_", ".") if rnd >= 2 else t.qcode
        t["block"] = code + ":" + code_part.fillna("p" + t.page.astype(str)).str.split(".").str[0]
        # Lettered items with no dot (Q1a, Q1b, ...) that share one opening question form one block, as dotted items do.
        lettered = code_part.fillna("").str.fullmatch(r"[A-Z]*\d+[a-z]"); base = code + ":" + code_part.fillna("").str[:-1]
        opening = t.question.str.split("?").str[0]
        for blabel, idx in t[lettered].groupby(base[lettered]).groups.items():
            if len(idx) >= 2 and opening[idx].nunique() == 1:
                assert blabel not in set(t.block), blabel
                t.loc[idx, "block"] = blabel
        t["qid"] = ID_PREFIX[rnd] + ":" + code + ":p" + t.page.astype(str); t["mode"] = mode
        t["system"], t["user"] = zip(*[prompt(mode, q, o) for q, o in zip(t.question, t.options)]) if len(t) else ([], [])
        kept.append(t)
        print(f"{code:8s} round {rnd} {mode:9s} pages {text.count('=====PAGE '):4d} | tables kept {len(t):3d}")
    A = pd.concat(kept, ignore_index=True); A.to_csv(out / "tables_read.csv", index=False)
    # step 4: cleaning and counting each question once
    shown = A.user.str.extract(r"Question: (.*)\n")[0]
    swept = shown.str.contains(r"^[A-Z_]{2,8} -?\d+ ") | shown.str.contains(r"\d+ -?\d+ -?\d+")
    A = A[~swept].copy()
    key = shown[~swept].str.lower().str.replace(r"[^a-z]", "", regex=True) + "|" + A["mode"]
    U = A[~key.duplicated()].copy()
    print(f"\nrows read {len(swept)}, with table numbers in the question {int(swept.sum())}, clean {len(A)}, each question once {len(U)}")
    Q = pd.DataFrame(dict(qid=U.qid, set=U["mode"].map(lambda m: SET_AND_MODE[m][0]), programme="Eurobarometer", survey=U.survey,
                          mode=U["mode"].map(lambda m: SET_AND_MODE[m][1]), block=U.block,
                          new_question=U.trend.map(lambda x: "" if x is None or pd.isna(x) else int(not x)),
                          question_text=U.user.str.extract(r"\nQuestion: (.*)\n")[0],
                          options_text=U.user.str.extract(r"\n(?:Substantive options|On-screen options \(in order\)): (.*)\n")[0],
                          n_options=U.n_opts.astype(int), real_dk_pct=U.dk_eu27.astype(float), n_asked=""))
    Q["question_chars"] = Q.question_text.str.len(); Q["prompt_chars"] = U.user.str.len()
    Q.to_csv(out / "questions_eurobarometer.csv", index=False); U[["qid", "system", "user"]].to_csv(out / "prompts_eurobarometer.csv", index=False)
    return Q, U[["qid", "system", "user"]]

def same(a, b):
    if pd.isna(a) and pd.isna(b): return True
    try: return float(a) == float(b)
    except (TypeError, ValueError): return str(a) == str(b)

def compare(Q, P):
    oldQ = pd.read_csv(ROOT / "data" / "questions.csv", keep_default_na=False); oldQ = oldQ[oldQ.programme == "Eurobarometer"]
    oldP = pd.read_csv(ROOT / "data" / "prompts_main.csv", keep_default_na=False); oldP = oldP[oldP.qid.isin(oldQ.qid)]
    newQ = pd.read_csv(Q, keep_default_na=False); newP = pd.read_csv(P, keep_default_na=False)
    print(f"\nquestions: data/questions.csv {len(oldQ)} Eurobarometer rows {oldQ.set.value_counts().to_dict()}; rebuilt {len(newQ)} {newQ.set.value_counts().to_dict()}")
    print("same row order:", list(oldQ.qid) == list(newQ.qid))
    diffs = []
    for qid in sorted(set(oldQ.qid) - set(newQ.qid)): diffs.append((qid, "row", "present", "missing"))
    for qid in sorted(set(newQ.qid) - set(oldQ.qid)): diffs.append((qid, "row", "missing", "present"))
    o, n = oldQ.set_index("qid"), newQ.set_index("qid"); both = [q for q in oldQ.qid if q in n.index]
    for qid in both:
        for f in o.columns:
            if not same(o.at[qid, f], n.at[qid, f]): diffs.append((qid, f, o.at[qid, f], n.at[qid, f]))
    po, pn = oldP.set_index("qid"), newP.set_index("qid")
    for qid in [q for q in po.index if q in pn.index]:
        for f in ("system", "user"):
            if po.at[qid, f] != pn.at[qid, f]: diffs.append((qid, "prompt " + f, po.at[qid, f], pn.at[qid, f]))
    bad = {d[0] for d in diffs}
    print(f"rows reproduced exactly (every field of questions.csv and both prompt texts): {sum(q not in bad for q in both)} of {len(oldQ)}")
    print(f"differences: {len(diffs)}")
    for d in diffs: print("  ", d)
    return len(diffs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); args = ap.parse_args()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="eurobarometer_")); out.mkdir(parents=True, exist_ok=True)
    print("writing to", out)
    build(out); sys.exit(1 if compare(out / "questions_eurobarometer.csv", out / "prompts_eurobarometer.csv") else 0)
