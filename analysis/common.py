"""Shared helpers: load the question table and forecasts, score a forecast, and the paired cluster bootstrap."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
DATA = Path(__file__).resolve().parents[1] / "data"

def questions():
    return pd.read_csv(DATA / "questions.csv")

def forecasts(model, setting, draws="mean", issp="corrected"):
    """One forecast per question: the mean over calls (draws='mean') or a single call number.
    The 47 General Social Survey 2024 interview questions from the ISSP self-completion supplement were first run with a prompt
    saying "don't know" is not read aloud; issp='corrected' (the default) uses their re-run with a prompt saying "Can't choose" is
    offered (data/forecasts_issp_supplement.csv), and issp='as run' keeps the first run (section 6 and the examples check use it)."""
    f = pd.read_csv(DATA / "forecasts_main.csv").query("model == @model and setting == @setting")
    if issp == "corrected":
        s = pd.read_csv(DATA / "forecasts_issp_supplement.csv").query("model == @model and setting == @setting")
        if len(s): f = pd.concat([f[~f.qid.isin(s.qid)], s])
    if draws != "mean": f = f[f.draw == draws]
    return f.groupby("qid").forecast_pct.mean()

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0: return np.nan
    return spearmanr(x[ok], y[ok])[0]

def mae(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return np.nanmean(np.abs(x - y))

def block_resamples(blocks, n=2000, seed=0):
    """Indices for n resamples, each drawing whole question blocks at random with replacement."""
    rng = np.random.default_rng(seed); b = np.asarray(blocks); ub = np.unique(b); ix = {u: np.where(b == u)[0] for u in ub}
    for _ in range(n): yield np.concatenate([ix[u] for u in rng.choice(ub, len(ub))])

def interval(x, y, blocks, score=spearman, n=2000, seed=0):
    vals = [score(np.asarray(x)[i], np.asarray(y)[i]) for i in block_resamples(blocks, n, seed)]
    return np.nanpercentile(vals, [2.5, 97.5])

RAW = Path(__import__("os").environ.get("SURVEY_RAW", Path(__file__).resolve().parents[1] / "raw"))
def raw_file(name):
    """Respondent-level files are not in the repo; see README for where to download them into raw/."""
    p = RAW / name
    if not p.exists(): raise FileNotFoundError(f"{p} not found: download it as described in README (section 'Respondent-level data').")
    return p
