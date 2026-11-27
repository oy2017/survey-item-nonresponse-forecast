"""Reproduces paper section 7 with Appendix D.8 (Table D12), section 8 (Table 6) with Appendix C, and the
numbers in the Abstract, Introduction (Table 1) and Conclusion that come from these or from Table 3.
Reads data/questions.csv, data/forecasts_main.csv, data/eb_surveys.csv and, for the real-interview pilots, raw/GSS2024.dta
(via raw_file).
Seeds and draw rules follow the original computations: sampled Eurobarometer pilots seed 3 (pilot plus forecast) and seed 1
(pilot alone); General Social Survey 2024 real-interview pilots seed 7, 2,000 draws, n whole interviews, scored on the people
not in the pilot; block bootstrap seed 0."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from common import questions, forecasts, spearman as sp, mae, raw_file, DATA

Q = questions()
Q["qtext"] = Q.question_text  # the question line sent to the model
Q["qlen_text"] = Q.qtext.str.len()   # question text only
Q["qlen_prompt"] = Q.prompt_chars         # whole prompt (question + options); the column stored in questions.csv
FM = pd.read_csv(DATA / "forecasts_main.csv")

def fc(model, setting, draws="mean"):
    return forecasts(model, setting, draws)
def gem_nothink():
    a = fc("Gemini Flash", "no thinking"); b = fc("Gemini Flash", "no thinking, earlier run"); return pd.concat([a, b])
M = {"opus": lambda d="mean": fc("Opus 5.5", "thinking", d), "sonnet": lambda d="mean": fc("Sonnet 5.5", "no thinking", d),
     "haiku": lambda d="mean": fc("Haiku 5.5", "no thinking", d), "flash_think": lambda d="mean": fc("Gemini Flash", "thinking", d),
     "flash": lambda d="mean": gem_nothink(), "flash_lite": lambda d="mean": fc("Flash-Lite", "no thinking", d)}
S = {s: Q[Q.set == s].set_index("qid") for s in Q.set.unique()}
def col(s, m, d="mean"): return M[m](d).reindex(S[s].index)
y = lambda s: S[s].real_dk_pct
def sp3(x, yy): return f"{sp(x, yy):.3f}"

print("=== A. Level per set (Appendix D.8 'The level is survey-specific'; section 7) ===")
for s in ("eb_face_to_face", "eb_web", "eb_telephone", "wvs7", "gss2022", "gss2024_interview"):
    yy = y(s); out = f"  {s:18s} n {len(yy)} real mean {yy.mean():.2f}% median {yy.median():.2f}% | constant at mean {mae(np.full(len(yy), yy.mean()), yy):.2f}, at median {mae(np.full(len(yy), yy.median()), yy):.2f} |"
    for m in M:
        x = col(s, m)
        if x.notna().all(): out += f" {m} {x.mean():.2f}% err {mae(x, yy):.2f};"
    print(out)
x = col("gss2024_interview", "opus", 0); print(f"  gss2024_interview Opus 5.5 first call: mean {x.mean():.2f}% err {mae(x, y('gss2024_interview')):.2f}")
E = S["eb_face_to_face"]; ye = E.real_dk_pct
print(f"  Opus 5.5 mean / real mean on Eurobarometer face-to-face: {col('eb_face_to_face', 'opus').mean() / ye.mean():.2f}")

print("\n=== B. The 47 ISSP questions: first run, web-prompt substitute, and re-run with the corrected prompt (Appendix D.8, section 3) ===")
I = S["gss2024_interview"]; yi = I.real_dk_pct
web_ids = S["gss2024_web"].index; issp = [w.replace("gss24_web:", "gss24_int:") for w in web_ids]; core = [v for v in I.index if v not in issp]
print(f"  interview questions {len(I)} | ISSP with a web forecast {len(issp)} | core {len(core)} | real mean ISSP {yi[issp].mean():.2f}%, core {yi[core].mean():.2f}%, all {yi.mean():.2f}%")
ASRUN = {"opus": ("Opus 5.5", "thinking"), "sonnet": ("Sonnet 5.5", "no thinking"), "haiku": ("Haiku 5.5", "no thinking"), "flash": ("Gemini Flash", "no thinking"), "flash_think": ("Gemini Flash", "thinking")}
def corrected(m, d="mean"):
    """First run (interview prompt), the web-prompt substitute for the 47, and the re-run with the corrected prompt (now the default)."""
    xi = forecasts(*ASRUN[m], d, issp="as run").reindex(I.index); xw = M[m](d).reindex(web_ids); xw.index = issp
    return pd.concat([xw, xi[core]]).reindex(I.index), xi, xw
for m, d, lab in (("opus", 0, "Opus 5.5 first call"), ("opus", "mean", "Opus 5.5 mean of 3"), ("sonnet", "mean", "Sonnet 5.5 mean of 3"), ("haiku", "mean", "Haiku 5.5"),
                  ("flash", "mean", "Gemini Flash"), ("flash_think", "mean", "Gemini Flash thinking")):
    xc, xi, xw = corrected(m, d)
    print(f"  {lab:22s} 47 ISSP: interview prompt mean {xi[issp].mean():.2f}% err {mae(xi[issp], yi[issp]):.2f} Spearman {sp3(xi[issp], yi[issp])} | web prompt mean {xw.mean():.2f}% err {mae(xw, yi[issp]):.2f} Spearman {sp3(xw, yi[issp])}")
    xr = col("gss2024_interview", m, d)
    print(f"  {'':22s} re-run with the corrected prompt: 47 mean {xr[issp].mean():.2f}% err {mae(xr[issp], yi[issp]):.2f} Spearman {sp3(xr[issp], yi[issp])} | 55 mean {xr.mean():.2f}% err {mae(xr, yi):.2f} Spearman {sp3(xr, yi)}")
    print(f"  {'':22s} 55: as run mean {xi.mean():.2f}% err {mae(xi, yi):.2f} Spearman {sp3(xi, yi)} | corrected mean {xc.mean():.2f}% err {mae(xc, yi):.2f} Spearman {sp3(xc, yi)} | core 8 mean {xi[core].mean():.2f}% err {mae(xi[core], yi[core]):.2f}")

print("\n=== C. Level from other surveys of the programme (Table D12, section 7) ===")
y22 = y("gss2022"); print(f"  real mean 2022 {y22.mean():.2f}% | 2024 {yi.mean():.2f}% | always guessing the 2022 mean on 2024: {mae(np.full(len(yi), y22.mean()), yi):.2f}")
for m, d in (("opus", "mean"), ("sonnet", "mean"), ("flash", "mean")):
    k = y22.mean() / col("gss2022", m).mean()
    for lab, x in (("as run", forecasts(*ASRUN[m], d, issp="as run").reindex(I.index)), ("re-run", col("gss2024_interview", m, d))):
        print(f"  {m:7s} {lab:9s} as is {mae(x, yi):.2f} | factor from 2022 {k:.2f} -> {mae(k * x, yi):.2f} | own 2024 mean {mae(yi.mean() / x.mean() * x, yi):.2f}")
for m in ("opus", "sonnet", "flash_think", "haiku"):
    x = col("eb_face_to_face", m); pr = pd.Series(index=E.index, dtype=float); cst = pr.copy(); own = pr.copy()
    for sv, dd in E.groupby("survey"):
        o = E.survey != sv; pr[dd.index] = x[dd.index] * ye[o].mean() / x[o].mean(); cst[dd.index] = ye[o].mean(); own[dd.index] = x[dd.index] * ye[dd.index].mean() / x[dd.index].mean()
    per = pd.Series({sv: mae(pr[dd.index], ye[dd.index]) < mae(x[dd.index], ye[dd.index]) for sv, dd in E.groupby("survey")})
    print(f"  EB {m:11s} as is {mae(x, ye):.2f} | factor from the other surveys {mae(pr, ye):.2f} | own mean {mae(own, ye):.2f} | constant at other surveys' mean {mae(cst, ye):.2f} | surveys improved {int(per.sum())} of {len(per)}")
# A designer has only the surveys already published (data/eb_surveys.csv). A question repeated from an already published
# survey needs no forecast (its published result is the ceiling of section 5.4), so here each question is scored in the
# first survey that asked it (earliest fieldwork), among all face-to-face rows before repeated questions are removed
# (data/section5/eb_clean_rows.csv, which carries the forecast for every copy). The factor is the real mean over the
# forecast mean of all rows of the surveys first published before this survey's fieldwork began. Added after the scores
# were seen; the 374 questions used everywhere else keep the copy processed first (Appendix D.9).
DATES = pd.read_csv(DATA / "eb_surveys.csv", parse_dates=["fieldwork_start", "first_published"]).set_index("survey")
ALL = pd.read_csv(DATA / "section5/eb_clean_rows.csv"); ALL = ALL[ALL["mode"] == "face-to-face"].reset_index(drop=True)
ALL["qk"] = ALL.question.str.lower().str.replace(r"[^a-z]", "", regex=True)
ALL["order"] = ALL.qid.map(pd.read_csv(DATA / "eb_rows.csv").set_index("row_id").row_order); ALL["fw"] = ALL.survey.map(DATES.fieldwork_start)
assert ALL.groupby("qk").fw.apply(lambda s: (s == s.min()).sum()).eq(1).all()
FIRST = ALL.sort_values(["fw", "order"]).drop_duplicates("qk").reset_index(drop=True)
PRIOR = {sv: [t for t in ALL.survey.unique() if DATES.loc[t, "first_published"] < DATES.loc[sv, "fieldwork_start"]] for sv in ALL.survey.unique()}
hasE = FIRST.survey.map(lambda sv: bool(PRIOR[sv])).to_numpy()
EBM = {"opus": "opus", "sonnet": "sonnet", "flash_think": "gemini_flash_thinking"}
print(f"  -- each question in the first survey that asked it: {len(ALL)} rows, {len(FIRST)} questions, {int(hasE.sum())} first asked in a survey with an earlier one;"
      f" none for {FIRST[~hasE].survey.value_counts().to_dict()} --")
def earlier(c):
    x = FIRST[c].to_numpy(float).copy(); oth = x.copy(); low = []
    for sv in FIRST.survey.unique():
        i = (FIRST.survey == sv).to_numpy(); o = ALL.survey != sv
        oth[i] = x[i] * ALL.real_dk_pct[o].mean() / ALL[c][o].mean()
        if PRIOR[sv]:
            p = ALL.survey.isin(PRIOR[sv]); x[i] = x[i] * ALL.real_dk_pct[p].mean() / ALL[c][p].mean()
            low.append(mae(x[i], FIRST.real_dk_pct[i]) < mae(FIRST[c][i], FIRST.real_dk_pct[i]))
    return x, oth, low
yF = FIRST.real_dk_pct.to_numpy(float)
for m, c in EBM.items():
    ea, oth, low = earlier(c); x0 = FIRST[c].to_numpy(float)
    print(f"  EB {m:11s} {int(hasE.sum())} questions: as is {mae(x0[hasE], yF[hasE]):.2f} | earlier surveys only {mae(ea[hasE], yF[hasE]):.2f} | all other surveys {mae(oth[hasE], yF[hasE]):.2f}"
          f" | earlier-surveys factor lowers the error in {sum(low)} of {len(low)} surveys")
for m in ("sonnet", "opus"):
    kE = ye.mean() / col("eb_face_to_face", m).mean(); kW = y("wvs7").mean() / col("wvs7", m).mean(); kG = y22.mean() / col("gss2022", m).mean(); xe = col("eb_face_to_face", m)
    print(f"  {m:7s} factors EB {kE:.2f}, WVS {kW:.2f}, GSS 2022 {kG:.2f} | EB error as is {mae(xe, ye):.2f}, with WVS factor {mae(kW * xe, ye):.2f}, with GSS 2022 factor {mae(kG * xe, ye):.2f}")
# The earlier-surveys factor against a sampled pilot of the survey itself (D.8, added after the scores were seen): each
# survey's factor from k sampled answers per question on all rows of that survey's questionnaire (drawn as in part D,
# seed 3, 2,000 draws; no "don't know" in the pilot counts as one), scored on the same questions as above.
print("  -- earlier surveys against a sampled pilot of each survey alone, same questions --")
gF, gA, yA = FIRST.survey.to_numpy(), ALL.survey.to_numpy(), ALL.real_dk_pct.to_numpy(float)
for m, c in EBM.items():
    ea, _, _ = earlier(c); xF, xA = FIRST[c].to_numpy(float), ALL[c].to_numpy(float); r3 = np.random.default_rng(3); out = []
    for k in (5, 10, 15):
        errs = []
        for _ in range(2000):
            cnt = r3.binomial(k, yA / 100).astype(float); xc = xF.copy()
            for sv in np.unique(gF):
                i, j = gF == sv, gA == sv; xc[i] = xF[i] * max(cnt[j].sum(), 1.0) / (k * xA[j] / 100).sum()
            errs.append(np.abs(xc[hasE] - yF[hasE]).mean())
        out.append(f"k={k} {np.mean(errs):.2f}")
    print(f"  EB {m:11s} {int(hasE.sum())} questions: earlier surveys {mae(ea[hasE], yF[hasE]):.2f} | pilot of the survey alone " + ", ".join(out))

print("\n=== D. Sampled pilots on the 374 face-to-face Eurobarometer questions (section 7, D.8; Abstract, Table 3) ===")
yv = ye.to_numpy(float); rng = np.random.default_rng(3); mm = 25.0
print(f"  constant at the real mean: {np.abs(yv - yv.mean()).mean():.2f}")
for k in (15, 50):
    acc = {}
    for _ in range(2000):
        c = rng.binomial(k, yv / 100).astype(float); p = 100 * c / k; pm = 100 * c.sum() / (k * len(yv)); cand = {"pilot alone": p, "pilot pulled to own mean": (k * p + mm * pm) / (k + mm)}
        for nm in ("sonnet", "opus", "flash_think"):
            x = col("eb_face_to_face", nm).to_numpy(float); xc = x * max(c.sum(), 1.0) / (k * x / 100).sum(); cand[f"{nm} rescaled"] = xc; cand[f"{nm} blended"] = (k * p + mm * xc) / (k + mm)
        for kx, v in cand.items(): acc.setdefault(kx, []).append((spearmanr(v, yv)[0] if np.std(v) > 0 else 0.0, np.abs(v - yv).mean()))
    print(f"  {k} answers per question: " + " | ".join(f"{kx} {np.mean([a for a, _ in v]):.3f}/{np.mean([b for _, b in v]):.2f}" for kx, v in acc.items()))
rng = np.random.default_rng(1); p = yv / 100
print("  pilot alone, Spearman with the published share: " + " ".join(f"k={k} {np.mean([spearmanr(x, p)[0] for x in (rng.binomial(k, p) / k for _ in range(2000)) if x.std() > 0]):.3f}" for k in (15, 25, 50, 100, 200)))

print("\n=== E. Real-interview pilots, General Social Survey 2024 (D.8, D.9; the pilot-plus-forecast rows are no longer in the paper) ===")
vs = sorted(v.split(":", 1)[1] for v in I.index); ids = ["gss24_int:" + v for v in vs]
raw = pd.read_stata(raw_file("GSS2024.dta"), columns=["id", "mode"] + vs, convert_missing=True, convert_categoricals=False); it = raw[raw["mode"].isin([1, 2])].reset_index(drop=True)
isnum = lambda t: isinstance(t, (int, float, np.number)) and not pd.isna(t)
DK = np.column_stack([(it[v].astype(str) == ".d").to_numpy() for v in vs]); ASK = np.column_stack([it[v].map(isnum).to_numpy() for v in vs]) | DK
full = 100 * DK.sum(0) / ASK.sum(0); print(f"  interviews {len(it)} | full-sample shares equal questions.csv: {np.allclose(full, yi[ids].to_numpy())}")
# A pilot sets a rescaling factor through its overall "don't know" percentage. How much that varies with n real interviews,
# against answers sampled independently at each question's real percentage for the same questions (D.8; seed 0, 4,000 draws).
r0 = np.random.default_rng(0)
for n in (10, 20):
    re_, sa_, kb = [], [], []
    for _ in range(4000):
        pk = r0.choice(len(it), n, replace=False); k = ASK[pk].sum(0); c = DK[pk].sum(0); kb.append(k.mean())
        re_.append(c.sum() / k.sum()); sa_.append(r0.binomial(k, full / 100).sum() / k.sum())
    print(f"  pilot of {n} interviews (about {np.mean(kb):.1f} answers per question): spread (SD) of its overall 'don't know' percentage, real {100 * np.std(re_):.2f} | sampled {100 * np.std(sa_):.2f} points")
# The targets are unweighted proportions (Appendix D.9). The same questions with the survey weight wtssnrps instead
# (added after the scores were seen).
wv = pd.read_stata(raw_file("GSS2024.dta"), columns=["mode", "wtssnrps"], convert_missing=True, convert_categoricals=False)
wv = pd.to_numeric(wv.loc[wv["mode"].isin([1, 2]), "wtssnrps"], errors="coerce").fillna(0).to_numpy()
wt = 100 * (wv[:, None] * DK).sum(0) / (wv[:, None] * ASK).sum(0)
print(f"  weighted targets (wtssnrps): Spearman with the unweighted {sp(wt, full):.3f} | mean {full.mean():.2f}% unweighted, {wt.mean():.2f}% weighted | forecasts as run, unweighted -> weighted: "
      + " | ".join(f"{m} {sp(col('gss2024_interview', m).reindex(ids).to_numpy(float), full):.3f} -> {sp(col('gss2024_interview', m).reindex(ids).to_numpy(float), wt):.3f}" for m in ("opus", "sonnet", "flash_think")))
def spn(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); return spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
for m in ("sonnet", "flash_think"):
    x0 = forecasts(*ASRUN[m], issp="as run").reindex(ids).to_numpy(float); x = col("gss2024_interview", m).reindex(ids).to_numpy(float)
    print(f"  {m}: full-sample Spearman as run {spn(x0, full):.3f} -> re-run {spn(x, full):.3f} | error re-run {np.abs(x - full).mean():.2f} | mean {x0.mean():.2f}% -> {x.mean():.2f}%")
def run(x, n, ms=(25,)):
    acc = {}; rng = np.random.default_rng(7); kbar = []; zero = []
    for _ in range(2000):
        pk = rng.choice(len(it), n, replace=False); k = ASK[pk].sum(0).astype(float); c = DK[pk].sum(0).astype(float)
        if (k < 1).any(): continue
        p = 100 * c / k; yy = 100 * (DK.sum(0) - c) / (ASK.sum(0) - k); xc = x * max(c.sum(), 1.0) / (k * x / 100).sum(); pbar = 100 * c.sum() / k.sum(); kbar.append(k.mean()); zero.append((c == 0).mean())
        cand = {"forecast alone": x, "pilot alone": p, "pilot pulled to own mean, m=25": (k * p + 25 * pbar) / (k + 25), "forecast rescaled": xc}
        for w in ms: cand[f"blended, m={w}"] = (k * p + w * xc) / (k + w)
        for kx, v in cand.items(): acc.setdefault(kx, []).append((spn(v, yy), np.abs(v - yy).mean()))
    print(f"  -- {n} interviews ({len(acc['pilot alone'])} draws; mean answers per question {np.mean(kbar):.1f}; share of questions with no 'don't know' {np.mean(zero):.2f}) --")
    for kx, v in acc.items():
        r = np.array([a for a, _ in v]); e = np.array([b for _, b in v]); print(f"     {kx:32s} Spearman {np.nanmean(r):.4f} (middle 95% {np.nanpercentile(r, 2.5):.2f} to {np.nanpercentile(r, 97.5):.2f}) | error {e.mean():.2f}")
for m in ("sonnet", "flash_think"):
    x = col("gss2024_interview", m).reindex(ids).to_numpy(float); print(f"  == {m}, re-run forecasts ==")
    for n in ((32, 107, 214) if m == "sonnet" else (32,)): run(x, n, ms=(5, 25, 50, 100) if m == "sonnet" and n in (32, 107) else (25,))

print("\n=== F. Web and telephone questions (Table 6, section 8, Appendix C) ===")
def boot(x, yy, g, n=2000, seed=0):
    rng = np.random.default_rng(seed); x, yy, g = np.asarray(x, float), np.asarray(yy, float), np.asarray(g); ug = np.unique(g); ix = {u: np.where(g == u)[0] for u in ug}; b = []
    for _ in range(n):
        i = np.concatenate([ix[u] for u in rng.choice(ug, len(ug))])
        if np.std(x[i]) > 0 and np.std(yy[i]) > 0: b.append(spearmanr(x[i], yy[i])[0])
    return np.percentile(b, [2.5, 97.5])
W = S["eb_web"].copy(); yw = W.real_dk_pct
for m in ("opus", "sonnet", "flash_think", "flash", "flash_lite"): W[m] = col("eb_web", m)
out = "  EB web 101:"
for c in ("opus", "sonnet", "flash_think", "flash", "qlen_text", "qlen_prompt", "n_options"):
    lo, hi = boot(W[c], yw, W.block); out += f" {c} {sp(W[c], yw):.3f} [{lo:.2f}, {hi:.2f}];"
print(out)
NAR = W.survey.isin(["FL576", "FL585"]); rng = np.random.default_rng(0)
for lab, s in (("general population", W[~NAR]), ("narrower", W[NAR]), ("all web", W)):
    g = s.block.to_numpy(); ug = np.unique(g); ix = {u: np.where(g == u)[0] for u in ug}
    for a in ("opus", "sonnet"):
        dd = []
        for _ in range(2000):
            t = s.iloc[np.concatenate([ix[u] for u in rng.choice(ug, len(ug))])]; dd.append(spn(t[a], t.real_dk_pct) - spn(t.qlen_text, t.real_dk_pct))
        if lab == "all web": print(f"  {lab}: {a} minus question length {spn(s[a], s.real_dk_pct) - spn(s.qlen_text, s.real_dk_pct):+.3f} [{np.nanpercentile(dd, 2.5):+.2f}, {np.nanpercentile(dd, 97.5):+.2f}]")
def inside(s, c, yc="real_dk_pct"):
    d = s[s.groupby("block")[c].transform("size") >= 2]
    a = sp(d[c] - d.groupby("block")[c].transform("mean"), d[yc] - d.groupby("block")[yc].transform("mean"))
    z = lambda v: d.groupby("block")[v].transform(lambda u: (rankdata(u) - (len(u) + 1) / 2) / len(u)); b = sp(z(c), z(yc))
    g4 = [(len(g), sp(g[c], g[yc])) for _, g in d.groupby("block") if len(g) >= 4 and g[c].std() > 0 and g[yc].std() > 0]
    return len(d), d.block.nunique(), a, b, np.average([r for _, r in g4], weights=[n for n, _ in g4])
for c in ("opus", "sonnet", "qlen_text", "qlen_prompt"):
    n, g, a, b, cc = inside(W, c); print(f"  EB web inside blocks ({n} questions, {g} blocks) {c}: (a) {a:.3f} (b) {b:.3f} (c) {cc:.3f}")
lo, hi = boot(W.opus - W.groupby("block").opus.transform("mean"), yw - W.groupby("block").real_dk_pct.transform("mean"), W.block)
d = W[W.groupby("block").opus.transform("size") >= 2]; lo, hi = boot(d.opus - d.groupby("block").opus.transform("mean"), d.real_dk_pct - d.groupby("block").real_dk_pct.transform("mean"), d.block)
print(f"  EB web inside blocks, Opus (a) interval [{lo:.2f}, {hi:.2f}]")
for sv, s in W.groupby("survey"):
    if len(s) >= 8: print(f"  web survey {sv} n {len(s)}: Opus {sp(s.opus, s.real_dk_pct):.2f} | question length {sp(s.qlen_text, s.real_dk_pct):.2f} | prompt length {sp(s.qlen_prompt, s.real_dk_pct):.2f}")
T = S["eb_telephone"].copy(); T["opus"] = col("eb_telephone", "opus")
for sv, s in T.groupby("survey"): print(f"  telephone survey {sv} n {len(s)}: Opus {sp(s.opus, s.real_dk_pct):.2f}")
G = S["gss2024_web"]; yg = G.real_dk_pct
print("  GSS 2024 web 47: " + " ".join(f"{m} {sp(col('gss2024_web', m), yg):.3f};" for m in M) + f" Opus first call {sp(col('gss2024_web', 'opus', 0), yg):.3f}; question length {sp(G.question_chars, yg):.3f}; options {sp(G.n_options, yg):.3f}")
print(f"  GSS 2024 web question length, two definitions: question text {sp(G.qtext.str.len(), yg):.3f} (= question_chars) | question + options {sp(G.qtext.str.len() + G.options_text.str.len(), yg):.3f}")
print(f"  EB telephone question length, two definitions: question line {sp(T.qlen_text, T.real_dk_pct):.3f} | whole prompt {sp(T.qlen_prompt, T.real_dk_pct):.3f}")
xi1 = forecasts(*ASRUN["opus"], 0, issp="as run")[issp]; xi3 = forecasts(*ASRUN["opus"], issp="as run")[issp]; xs = forecasts(*ASRUN["sonnet"], issp="as run")[issp]
print(f"  same 47 ISSP, interview prompt vs interview shares: Opus first call {sp(xi1, yi[issp]):.3f}, mean of 3 {sp(xi3, yi[issp]):.3f}; Sonnet {sp(xs, yi[issp]):.3f}")

print("\n=== G. Whole surveys, Eurobarometer face-to-face, Opus 5.5 (section 7); single questions (section 8) ===")
D = E.copy(); D["opus"] = col("eb_face_to_face", "opus")
tab = D.groupby("survey").apply(lambda s: pd.Series({"n": len(s), "real": s.real_dk_pct.mean(), "opus": s.opus.mean(), "ratio": s.opus.mean() / s.real_dk_pct.mean(), "rho_in": sp(s.opus, s.real_dk_pct) if len(s) >= 8 else np.nan}))
print(tab.round(2).sort_values("ratio").to_string())
D["res"] = D.opus.rank(pct=True) - D.real_dk_pct.rank(pct=True); D["agree"] = D.qtext.str.contains("agree", case=False)
o = D.sort_values("res"); under, over = o.head(20), o.tail(20)
print(f"  20 most under-ranked by survey: {under.survey.value_counts().to_dict()}")
print(f"  agree/disagree wording among the 40 largest rank misses: {int(under.agree.sum() + over.agree.sum())} of 40 | among all 374: {int(D.agree.sum())}")
# The largest misses on single questions, once whole-survey offsets are set aside (added after the scores were seen): ranks
# taken inside each survey with at least eight questions; the question the forecast ranks furthest below and above its real rank.
W8 = D[D.groupby("survey").survey.transform("size") >= 8].copy()
W8["res_in"] = W8.groupby("survey").opus.rank(pct=True) - W8.groupby("survey").real_dk_pct.rank(pct=True)
for lab, part in (("ranked furthest below their real rank (forecast too low)", W8.nsmallest(6, "res_in")), ("ranked furthest above (forecast too high)", W8.nlargest(6, "res_in"))):
    print(f"  inside surveys, questions {lab}:")
    for q, r in part.iterrows(): print(f"     {r.survey:8s} real {r.real_dk_pct:4.1f}% forecast {r.opus:5.1f}% | {r.qtext[:150]}")

print("\n=== H. Abstract, Introduction (Table 1), Conclusion ===")
print(f"  questions {len(Q)} | programmes {Q.programme.nunique()} {sorted(Q.programme.unique())} | sets {Q.set.value_counts().to_dict()}")
for m in ("opus", "sonnet", "flash_think"): print(f"  EB face-to-face 374 Spearman {m} {sp(col('eb_face_to_face', m), ye):.3f}")
print(f"  EB face-to-face question length: question text {sp(E.qlen_text, ye):.3f}, prompt (prompt_chars) {sp(E.prompt_chars, ye):.3f}")
t1 = E[E.block == "STD105:QA1"].copy(); t1["opus"] = col("eb_face_to_face", "opus")[t1.index]
print(t1[["qtext", "real_dk_pct", "opus"]].round(1).to_string()); print(f"  Table 1 Spearman {sp(t1.opus, t1.real_dk_pct):.3f}")
for m in ("opus", "sonnet", "haiku"): print(f"  EB face-to-face mean forecast {m} {col('eb_face_to_face', m).mean():.2f}% (real {ye.mean():.2f}%)")
