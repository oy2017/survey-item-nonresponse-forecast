"""Helpers shared by the scripts that reproduce paper sections 5.1, 5.2 and 5.4 (Appendices D.3, D.4, D.6)."""
import re, numpy as np, pandas as pd
from common import DATA, questions, forecasts, spearman, mae, block_resamples, interval, raw_file
S5 = DATA / "section5"
COUNTRIES = "BE BG CZ DK DE EE IE EL ES FR HR IT CY LV LT LU HU MT NL AT PL PT RO SI SK FI SE".split()

def test_sets():
    """The five interviewer-run sets of Table 3, as data frames with one row per question."""
    q = questions()  # question_chars: length of the question line sent to the model, the paper's "question length"
    return {"World Values Survey 7": q[q.set == "wvs7"], "General Social Survey 2022": q[q.set == "gss2022"],
            "General Social Survey 2024": q[q.set == "gss2024_interview"], "Eurobarometer face-to-face": q[q.set == "eb_face_to_face"],
            "Eurobarometer, new questions": q[(q.set == "eb_face_to_face") & (q.new_question == 1)]}

def one_call(model, setting, qids):
    """The first call (draw 0) of each question."""
    return forecasts(model, setting, 0).reindex(qids)

def paired_diff(a, b, y, blocks, n=2000, seed=0, score=spearman):
    """Score of a minus score of b on the same block resamples: point value and 95% interval."""
    a, b, y = (np.asarray(v, float) for v in (a, b, y))
    d = [score(a[i], y[i]) - score(b[i], y[i]) for i in block_resamples(blocks, n, seed)]
    return score(a, y) - score(b, y), np.nanpercentile(d, [2.5, 97.5])

def inside_blocks(x, y, blocks, min_block=2):
    """Three versions of the correlation inside question blocks (Appendix D.2): (a) block means removed; (b) centred ranks inside
    each block divided by block size; (c) size-weighted mean of the correlations inside blocks of four or more.
    Returns (a, b, c, number of questions, number of blocks, number of blocks used in c, number of those with a positive correlation)."""
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float), "g": np.asarray(blocks)})
    d = d[d.groupby("g").x.transform("size") >= min_block]
    dm = lambda c: d[c] - d.groupby("g")[c].transform("mean")
    rk = lambda c: d.groupby("g")[c].transform(lambda v: (v.rank() - (len(v) + 1) / 2) / len(v))
    cs = [(len(g), spearman(g.x, g.y)) for _, g in d.groupby("g") if len(g) >= 4]; cs = [(k, v) for k, v in cs if not np.isnan(v)]
    c = sum(k * v for k, v in cs) / sum(k for k, _ in cs) if cs else np.nan
    return spearman(dm("x"), dm("y")), spearman(rk("x"), rk("y")), c, len(d), d.g.nunique(), len(cs), sum(v > 0 for _, v in cs)

def fmt_ci(v, ci, signed=False):
    s = "+" if signed else ""
    return f"{v:{s}.3f} [{ci[0]:{s}.2f}, {ci[1]:{s}.2f}]"
