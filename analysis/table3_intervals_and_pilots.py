"""Reproduces Table 3 / Table D2 intervals and pilot column, the notes under Table D2, the forecast-minus-pilot comparisons and the
sampled-pilot sizes in Appendix D.3, and the 5.1 text on one call against three (paper sections 5.1, 5.4 and Appendix D.3).
Real-interview pilots: n whole interviews drawn at random, scored against the people not in the pilot.
Sampled pilots: k answers per question drawn at the real share. Intervals: 2,000 resamples of whole question blocks (seed 0)."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from section5_common import *

T = test_sets()
MODELS = {"Opus 5.5": ("Opus 5.5", "thinking"), "Sonnet 5.5": ("Sonnet 5.5", "no thinking"), "Haiku 5.5": ("Haiku 5.5", "no thinking")}
def gem(name): return ("Gemini Flash", "thinking")

print("=== Table D2: Spearman correlation with 95% interval (blocks resampled 2,000 times, seed 0) ===")
for name, s in T.items():
    y, b = s.real_dk_pct.to_numpy(), s.block.to_numpy(); out = f"{name:30s} {len(s)} / {s.block.nunique()} |"
    for lab, (m, st) in list(MODELS.items()) + [("Gemini Flash", gem(name))]:
        x = forecasts(m, st).reindex(s.qid).to_numpy(); out += f" {lab} {fmt_ci(spearman(x, y), interval(x, y, b))} |"
    out += f" question length {spearman(s.question_chars, y):+.3f} | options {spearman(s.n_options, y):+.3f}"
    print(out)
print("Question length, both definitions (whole prompt = data/questions.csv prompt_chars; question text alone = the 'Question:' line for Eurobarometer, question_text for the others):")
for name, s in T.items():
    qt = s.question_chars
    print(f"  {name:30s} whole prompt {spearman(s.prompt_chars, s.real_dk_pct):+.3f} | question text alone {spearman(qt, s.real_dk_pct):+.3f}")
s = T["General Social Survey 2024"]; y, b = s.real_dk_pct.to_numpy(), s.block.to_numpy(); x0 = forecasts("Opus 5.5", "thinking", 0).reindex(s.qid).to_numpy()
print(f"Opus 5.5 on General Social Survey 2024, first call only: {fmt_ci(spearman(x0, y), interval(x0, y, b))}")

print("\n=== Other Gemini settings (note under Table D2) ===")
for name in ("World Values Survey 7", "General Social Survey 2022", "Eurobarometer face-to-face", "Eurobarometer, new questions", "General Social Survey 2024"):
    nt = "no thinking, earlier run" if name in ("World Values Survey 7", "General Social Survey 2022") else "no thinking"  # Flash without thinking on the two older surveys: the earlier run
    s = T[name]; print(f"  {name:30s} " + " | ".join(f"{m} {st}: {spearman(forecasts(m, st).reindex(s.qid), s.real_dk_pct):.3f}" for m, st in (("Gemini Flash", nt), ("Flash-Lite", "no thinking"))))

print("\n=== One call against the mean of three (Opus 5.5 and Sonnet 5.5) ===")
for name in ("Eurobarometer face-to-face", "Eurobarometer, new questions", "General Social Survey 2024"):
    s = T[name]; y = s.real_dk_pct
    for m, st in (MODELS["Opus 5.5"], MODELS["Sonnet 5.5"]):
        m3 = spearman(forecasts(m, st).reindex(s.qid), y); first = spearman(one_call(m, st, s.qid), y)
        singles = [spearman(forecasts(m, st, d).reindex(s.qid), y) for d in (0, 1, 2)]  # questions with only a stored mean drop out here
        print(f"  {name:30s} {m:10s} mean of 3 {m3:.3f} | first call (stored mean where single calls were not kept) {first:.3f} | each call on the questions with three stored calls {', '.join(f'{v:.3f}' for v in singles)} | largest gap first call vs mean {abs(first - m3):.3f}")

SC = pd.read_csv(S5 / "eb_single_calls.csv")
print("  every single call (all three calls are stored for every Eurobarometer question in data/section5/eb_single_calls.csv):")
for name in ("Eurobarometer face-to-face", "Eurobarometer, new questions", "General Social Survey 2024"):
    s = T[name]; y = s.real_dk_pct
    for m, st in (MODELS["Opus 5.5"], MODELS["Sonnet 5.5"]):
        m3 = spearman(forecasts(m, st).reindex(s.qid), y)
        if name.startswith("Euro"): singles = [spearman(SC[(SC.model == m) & (SC.draw == d)].set_index("qid").forecast_pct.reindex(s.qid), y) for d in (0, 1, 2)]
        else: singles = [spearman(forecasts(m, st, d).reindex(s.qid), y) for d in (0, 1, 2)]
        print(f"  {name:30s} {m:10s} mean of 3 {m3:.3f} | single calls {', '.join(f'{v:.3f}' for v in singles)} | largest gap {max(abs(v - m3) for v in singles):.3f}")

# ---------------- real-interview pilots ----------------
isnum = lambda t: isinstance(t, (int, float, np.number)) and not pd.isna(t)
def wvs_matrix(s):
    vs = [q.split(":", 1)[1] for q in s.qid]; raw = pd.read_csv(raw_file("WVS_Cross-National_Wave_7_csv_v6_0.csv"), usecols=["Q_MODE"] + vs, low_memory=False)
    V = raw[raw.Q_MODE.isin([1, 2])][vs].to_numpy(float); return V == -1, (V > 0) | (V == -1)
def gss_matrix(s, fname, sort=False):
    vs = [q.split(":", 1)[1] for q in s.qid]; raw = pd.read_stata(raw_file(fname), columns=["mode"] + vs, convert_missing=True, convert_categoricals=False)
    raw = raw[raw["mode"].isin([1, 2])].reset_index(drop=True)
    DK = np.column_stack([(raw[v].astype(str) == ".d").to_numpy() for v in vs]); ASK = np.column_stack([raw[v].map(isnum).to_numpy() for v in vs]) | DK; return DK, ASK
MAT = {"World Values Survey 7": wvs_matrix(T["World Values Survey 7"]), "General Social Survey 2022": gss_matrix(T["General Social Survey 2022"], "GSS2022.dta"),
       "General Social Survey 2024": gss_matrix(T["General Social Survey 2024"], "GSS2024.dta")}
print("\n=== Real-interview pilots: n whole interviews, scored against everyone not in the pilot ===")
for name, (DK, ASK) in MAT.items():
    full = 100 * DK.sum(0) / ASK.sum(0); print(f"  {name}: {DK.shape[0]} interviews; each person asked {100 * ASK.mean():.1f}% of the {DK.shape[1]} questions; full-sample share against data/questions.csv, largest difference {np.abs(full - T[name].real_dk_pct.to_numpy()).max():.2f} points")
def pilot_draws(DK, ASK, n, reps, rng):
    r, kk = [], []
    for _ in range(reps):
        pk = rng.choice(DK.shape[0], n, replace=False); k = ASK[pk].sum(0).astype(float); c = DK[pk].sum(0).astype(float)
        if (k < 1).any(): continue
        p = 100 * c / k; y = 100 * (DK.sum(0) - c) / (ASK.sum(0) - k); r.append(0.0 if p.std() == 0 else spearman(p, y)); kk.append(k.mean())
    return np.mean(r), np.mean(kk), len(r)
rng = np.random.default_rng(11)  # one generator across both surveys and all sizes, 1,000 draws each
for name in ("World Values Survey 7", "General Social Survey 2022"):
    DK, ASK = MAT[name]; per = ASK.mean()
    for target in (15, 25, 50, 100, 200):
        n = int(round(target / per)); v, kk, used = pilot_draws(DK, ASK, n, 1000, rng)
        print(f"  {name:28s} {n:4d} interviews (about {kk:.1f} answers per question; {used} draws used): Spearman {v:.3f}")
DK, ASK = MAT["General Social Survey 2024"]
for n in (107,):
    v, kk, used = pilot_draws(DK, ASK, n, 2000, np.random.default_rng(7)); print(f"  General Social Survey 2024     {n:4d} interviews (about {kk:.1f} answers per question; {used} draws used): Spearman {v:.3f}")

# ---------------- sampled pilots ----------------
print("\n=== Sampled pilots: k answers per question drawn at the real share, scored against that share (2,000 draws) ===")
rng = np.random.default_rng(0)
def sampled(y, k, n=2000):
    v = [spearman(x, y) for x in (rng.binomial(k, y / 100) / k for _ in range(n)) if x.std() > 0]; return np.mean(v)
web = questions().query("set == 'eb_web'")
for name, s, ks in (("World Values Survey 7", T["World Values Survey 7"], (50,)), ("General Social Survey 2022", T["General Social Survey 2022"], (50,)),
                    ("General Social Survey 2024", T["General Social Survey 2024"], (50, 55)), ("Eurobarometer face-to-face", T["Eurobarometer face-to-face"], (50,)),
                    ("Eurobarometer, new questions", T["Eurobarometer, new questions"], (50,)), ("Eurobarometer web", web, (50,))):
    print(f"  {name:30s} " + ", ".join(f"k={k} {sampled(s.real_dk_pct.to_numpy(float), k):.3f}" for k in ks))
rng = np.random.default_rng(1)
for name in ("Eurobarometer face-to-face", "Eurobarometer, new questions"):
    p = T[name].real_dk_pct.to_numpy() / 100; out = f"  {name:30s} pilot sizes:"
    for k in (15, 25, 50, 100, 200):
        v = [spearman(x, p) for x in (rng.binomial(k, p) / k for _ in range(2000)) if x.std() > 0]; out += f" k={k} {np.mean(v):.3f}"
    print(out)

# ---------------- forecast minus pilot ----------------
print("\n=== Forecast minus pilot: each of 2,000 replicates redraws the pilot and resamples whole blocks (averages over replicates) ===")
def blocks_ix(groups, rng):
    g = np.asarray(groups); ug = np.unique(g); ix = {u: np.where(g == u)[0] for u in ug}; return np.concatenate([ix[u] for u in rng.choice(ug, len(ug))])
def report(name, F, rows):
    for m in F:
        d = np.array([r[m] for r in rows]); f = np.array([r[m + "_f"] for r in rows]); p = np.array([r["pilot"] for r in rows])
        print(f"  {name:30s} {m:12s} forecast {np.nanmean(f):.3f} | pilot {np.nanmean(p):.3f} | forecast minus pilot {np.nanmean(d):+.3f} [{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}]")
def gap_real(name, DK, ASK, F, groups, n, seed=1, B=2000):
    rng = np.random.default_rng(seed); out = []
    while len(out) < B:
        pk = rng.choice(DK.shape[0], n, replace=False); k = ASK[pk].sum(0).astype(float); c = DK[pk].sum(0).astype(float)
        if (k < 1).any(): continue
        p = 100 * c / k; y = 100 * (DK.sum(0) - c) / (ASK.sum(0) - k); ix = blocks_ix(groups, rng); r = {"pilot": spearman(p[ix], y[ix])}
        for m, x in F.items(): r[m + "_f"] = spearman(x[ix], y[ix]); r[m] = r[m + "_f"] - r["pilot"]
        out.append(r)
    report(f"{name} ({n} interviews)", F, out)
def gap_sampled(name, y, F, groups, k, seed=2, B=2000):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        ix = blocks_ix(groups, rng); p = rng.binomial(k, y / 100) / k * 100; r = {"pilot": spearman(p[ix], y[ix])}
        for m, x in F.items(): r[m + "_f"] = spearman(x[ix], y[ix]); r[m] = r[m + "_f"] - r["pilot"]
        out.append(r)
    report(f"{name} (k={k})", F, out)
for name, n in (("World Values Survey 7", 52), ("General Social Survey 2022", 80), ("General Social Survey 2024", 107)):
    s = T[name]; DK, ASK = MAT[name]
    opus = forecasts("Opus 5.5", "thinking").reindex(s.qid).to_numpy()
    gap_real(name, DK, ASK, {"Opus 5.5": opus, "Sonnet 5.5": forecasts("Sonnet 5.5", "no thinking").reindex(s.qid).to_numpy()}, s.block.to_numpy(), n)
for name in ("Eurobarometer face-to-face", "Eurobarometer, new questions"):
    s = T[name]; F = {m: forecasts(m, st).reindex(s.qid).to_numpy() for m, st in (("Opus 5.5", "thinking"), ("Sonnet 5.5", "no thinking"), ("Gemini Flash", "thinking"))}
    for k in (50, 200): gap_sampled(name, s.real_dk_pct.to_numpy(float), F, s.block.to_numpy(), k)
