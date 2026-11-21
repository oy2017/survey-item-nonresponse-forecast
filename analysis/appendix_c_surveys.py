"""Appendix C, Table C1: the 22 Eurobarometer surveys, with the Spearman correlation per survey.
Rows are the clean table rows of each survey (a question repeated in another survey appears in both surveys' rows).
"Question length" is the number of characters of the question text; "n/a" marks a survey with fewer than 8 questions or a
column whose values do not vary within the survey. Ordered by mode, then by date of first publication."""
import pandas as pd
from common import DATA, spearman
from section5_memorization import eb_clean_rows

if __name__ == "__main__":
    _, C = eb_clean_rows(); S = pd.read_csv(DATA / "eb_surveys.csv").set_index("survey")
    cell = lambda s, c: "n/a" if len(s) < 8 or s[c].nunique() < 2 or s.real_dk_pct.nunique() < 2 else f"{spearman(s[c], s.real_dk_pct):.2f}"
    order = {"face-to-face": 0, "web": 1, "telephone": 2}
    print("| Survey | Mode | Fieldwork | First published | Questions | New questions | Real mean % | Opus 5.5 | Sonnet 5.5 | Question length |")
    total_new = 0
    for sv, s in sorted(C.groupby("survey"), key=lambda t: (order[t[1]["mode"].iloc[0]], S.first_published[t[0]])):
        md = s["mode"].iloc[0]; new = int((s.prior_wave_column == "new").sum()) if md == "face-to-face" else "n/a"; total_new += new if md == "face-to-face" else 0
        print(f"| {sv} | {md} | {S.fieldwork_start[sv]} to {S.fieldwork_end[sv]} | {S.first_published[sv]} | {len(s)} | {new} | {s.real_dk_pct.mean():.1f} | {cell(s, 'opus55')} | {cell(s, 'sonnet55')} | {cell(s, 'q_chars')} |")
    print(f"\nsurveys {C.survey.nunique()}; clean rows {len(C)} {C['mode'].value_counts().to_dict()}; new-question rows (face-to-face) {total_new}")
    print("prior-wave column by mode:", C.groupby("mode").prior_wave_column.value_counts().to_dict())
