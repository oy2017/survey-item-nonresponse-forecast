"""Recompute the real "don't know" share (real_dk_pct) and the number asked (n_asked) for the World Values Survey 7 and
General Social Survey 2022 and 2024 questions in data/questions.csv from the respondent-level files in raw/ (see README),
and check them against the table. Section 3 and Appendix D.1 of the paper.

Rules (Appendix D.1):
- World Values Survey 7: face-to-face interviews (Q_MODE 1 or 2). Asked = any answer code between the lowest and highest
  substantive code, or -1 ("don't know"). Refusals (-2) and not-asked codes are left out of the base.
- General Social Survey 2022: interviewer-run cases (mode 1 or 2). Asked = a substantive answer code (all codes from the lowest to the highest where the
  options are printed as a range, "1=... 7=..."), or ".d" ("don't know").
- General Social Survey 2024: interviewer-run cases (mode 1 or 2) and web cases (mode 4). Asked = any numeric answer or
  ".d". On the web ".d" is the "can't choose" box shown on screen."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis")); from common import questions, raw_file
Q = questions(); out = []
codes = lambda t: [int(c) for c in re.findall(r"(?:^|[;/]|\.\.\.)\s*(-?\d+)\s*=", str(t))]
isnum = lambda x: isinstance(x, (int, float, np.number)) and not pd.isna(x)
# World Values Survey 7
w = Q[Q.set == "wvs7"]; vs = list(w["var"] if "var" in w else w.qid.str.split(":").str[1])
raw = pd.read_csv(raw_file("WVS_Cross-National_Wave_7_csv_v6_0.csv"), usecols=["Q_MODE"] + vs, low_memory=False); raw = raw[raw.Q_MODE.isin([1, 2])]
for (_, q), v in zip(w.iterrows(), vs):
    c = codes(q.options_text); col = raw[v].to_numpy(); asked = ((col >= min(c)) & (col <= max(c))) | (col == -1)
    out.append((q.qid, 100 * (col[asked] == -1).mean(), int(asked.sum())))
# General Social Survey 2022
g = Q[Q.set == "gss2022"]; vs = list(g.qid.str.split(":").str[1])
raw = pd.read_stata(raw_file("GSS2022.dta"), columns=["mode"] + vs, convert_missing=True, convert_categoricals=False); raw = raw[raw["mode"].isin([1, 2])]
for (_, q), v in zip(g.iterrows(), vs):
    c = codes(q.options_text); c = set(range(min(c), max(c) + 1)) if "..." in q.options_text else set(c); s = raw[v]; dk = (s.astype(str) == ".d").to_numpy(); sub = s.apply(lambda x: isnum(x) and int(x) in c).to_numpy()
    out.append((q.qid, 100 * dk.sum() / (dk | sub).sum(), int((dk | sub).sum())))
# General Social Survey 2024
h = Q[Q.set.isin(["gss2024_interview", "gss2024_web"])]; vs = sorted(set(h.qid.str.split(":").str[1]))
raw = pd.read_stata(raw_file("GSS2024.dta"), columns=["mode"] + vs, convert_missing=True, convert_categoricals=False)
for _, q in h.iterrows():
    v = q.qid.split(":")[1]; s = raw.loc[raw["mode"].isin([1, 2]) if q.set == "gss2024_interview" else raw["mode"] == 4, v]
    dk = int((s.astype(str) == ".d").sum()); n = int(s.apply(isnum).sum()) + dk; out.append((q.qid, 100 * dk / n, n))
R = pd.DataFrame(out, columns=["qid", "recomputed_pct", "recomputed_n"]).merge(Q[["qid", "set", "real_dk_pct", "n_asked"]], on="qid")
R["diff"] = (R.recomputed_pct - R.real_dk_pct).abs(); R["n_diff"] = R.recomputed_n - R.n_asked.astype(float)
print(R.groupby("set").agg(questions=("qid", "size"), max_share_diff=("diff", "max"), questions_with_n_diff=("n_diff", lambda d: int((d != 0).sum()))).to_string())
bad = R[(R["diff"] > 1e-4) | (R.n_diff != 0)]; print("mismatches:", len(bad)); print(bad.head(10).to_string())
