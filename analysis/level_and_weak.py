"""Reproduces paper section 7 (Table 6) with Appendix D.8 (Tables D15, D16), section 8 (Table 7) with Appendix D.9, and the
numbers in the Abstract, Introduction (Table 1) and Conclusion that come from these or from Table 3.
Reads data/questions.csv, data/forecasts_main.csv and, for the real-interview pilots, raw/GSS2024.dta (via raw_file).
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

print("\n=== B. The 47 ISSP questions with the web prompt; the corrected forecasts (Appendix D.8, section 7) ===")
I = S["gss2024_interview"]; yi = I.real_dk_pct
web_ids = S["gss2024_web"].index; issp = [w.replace("gss24_web:", "gss24_int:") for w in web_ids]; core = [v for v in I.index if v not in issp]
print(f"  interview questions {len(I)} | ISSP with a web forecast {len(issp)} | core {len(core)} | real mean ISSP {yi[issp].mean():.2f}%, core {yi[core].mean():.2f}%, all {yi.mean():.2f}%")
def corrected(m, d="mean"):
    xi = col("gss2024_interview", m, d); xw = M[m](d).reindex(web_ids); xw.index = issp
    return pd.concat([xw, xi[core]]).reindex(I.index), xi, xw
for m, d, lab in (("opus", 0, "Opus 5.5 first call"), ("opus", "mean", "Opus 5.5 mean of 3"), ("sonnet", "mean", "Sonnet 5.5 mean of 3"), ("haiku", "mean", "Haiku 5.5"),
                  ("flash", "mean", "Gemini Flash"), ("flash_think", "mean", "Gemini Flash thinking")):
    xc, xi, xw = corrected(m, d)
    print(f"  {lab:22s} 47 ISSP: interview prompt mean {xi[issp].mean():.2f}% err {mae(xi[issp], yi[issp]):.2f} Spearman {sp3(xi[issp], yi[issp])} | web prompt mean {xw.mean():.2f}% err {mae(xw, yi[issp]):.2f} Spearman {sp3(xw, yi[issp])}")
    print(f"  {'':22s} 55: as run mean {xi.mean():.2f}% err {mae(xi, yi):.2f} Spearman {sp3(xi, yi)} | corrected mean {xc.mean():.2f}% err {mae(xc, yi):.2f} Spearman {sp3(xc, yi)} | core 8 mean {xi[core].mean():.2f}% err {mae(xi[core], yi[core]):.2f}")

print("\n=== C. Level from other surveys of the programme (Table D16, section 7) ===")
y22 = y("gss2022"); print(f"  real mean 2022 {y22.mean():.2f}% | 2024 {yi.mean():.2f}% | always guessing the 2022 mean on 2024: {mae(np.full(len(yi), y22.mean()), yi):.2f}")
for m, d in (("opus", "mean"), ("sonnet", "mean"), ("flash", "mean")):
    k = y22.mean() / col("gss2022", m).mean()
    for lab, x in (("as run", col("gss2024_interview", m, d)), ("corrected", corrected(m, d)[0])):
        print(f"  {m:7s} {lab:9s} as is {mae(x, yi):.2f} | factor from 2022 {k:.2f} -> {mae(k * x, yi):.2f} | own 2024 mean {mae(yi.mean() / x.mean() * x, yi):.2f}")
for m in ("opus", "sonnet", "flash_think", "haiku"):
    x = col("eb_face_to_face", m); pr = pd.Series(index=E.index, dtype=float); cst = pr.copy(); own = pr.copy()
    for sv, dd in E.groupby("survey"):
        o = E.survey != sv; pr[dd.index] = x[dd.index] * ye[o].mean() / x[o].mean(); cst[dd.index] = ye[o].mean(); own[dd.index] = x[dd.index] * ye[dd.index].mean() / x[dd.index].mean()
    per = pd.Series({sv: mae(pr[dd.index], ye[dd.index]) < mae(x[dd.index], ye[dd.index]) for sv, dd in E.groupby("survey")})
    print(f"  EB {m:11s} as is {mae(x, ye):.2f} | factor from the other surveys {mae(pr, ye):.2f} | own mean {mae(own, ye):.2f} | constant at other surveys' mean {mae(cst, ye):.2f} | surveys improved {int(per.sum())} of {len(per)}")
# A designer has only the surveys already published. Same factor, taken only from the surveys first published before this
# survey's fieldwork began (data/eb_surveys.csv). The earliest survey (SP567, 43 questions) has none and keeps its raw forecast.
DATES = pd.read_csv(DATA / "eb_surveys.csv", parse_dates=["fieldwork_start", "first_published"]).set_index("survey")
print("  -- factor from earlier surveys only (first published before the survey's fieldwork began) --")
for m in ("opus", "sonnet", "flash_think", "haiku"):
    x = col("eb_face_to_face", m); ea = x.copy(); pr = x.copy(); has = pd.Series(False, index=E.index)
    for sv, dd in E.groupby("survey"):
        o = E.survey != sv; pr[dd.index] = x[dd.index] * ye[o].mean() / x[o].mean()
        prior = [t for t in E.survey.unique() if DATES.loc[t, "first_published"] < DATES.loc[sv, "fieldwork_start"]]
        if prior:
            pm = E.survey.isin(prior); ea[dd.index] = x[dd.index] * ye[pm].mean() / x[pm].mean(); has[dd.index] = True
    print(f"  EB {m:11s} {int(has.sum())} questions with an earlier survey: as is {mae(x[has], ye[has]):.2f} | earlier surveys only {mae(ea[has], ye[has]):.2f} | all other surveys {mae(pr[has], ye[has]):.2f}"
          f" || all 374 (raw forecast where no earlier survey): {mae(ea, ye):.2f}")
for m in ("sonnet", "opus"):
    kE = ye.mean() / col("eb_face_to_face", m).mean(); kW = y("wvs7").mean() / col("wvs7", m).mean(); kG = y22.mean() / col("gss2022", m).mean(); xe = col("eb_face_to_face", m)
    print(f"  {m:7s} factors EB {kE:.2f}, WVS {kW:.2f}, GSS 2022 {kG:.2f} | EB error as is {mae(xe, ye):.2f}, with WVS factor {mae(kW * xe, ye):.2f}, with GSS 2022 factor {mae(kG * xe, ye):.2f}")

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

print("\n=== E. Real-interview pilots, General Social Survey 2024 (Table 6, Table D15) ===")
vs = sorted(v.split(":", 1)[1] for v in I.index); ids = ["gss24_int:" + v for v in vs]
raw = pd.read_stata(raw_file("GSS2024.dta"), columns=["id", "mode"] + vs, convert_missing=True, convert_categoricals=False); it = raw[raw["mode"].isin([1, 2])].reset_index(drop=True)
isnum = lambda t: isinstance(t, (int, float, np.number)) and not pd.isna(t)
DK = np.column_stack([(it[v].astype(str) == ".d").to_numpy() for v in vs]); ASK = np.column_stack([it[v].map(isnum).to_numpy() for v in vs]) | DK
full = 100 * DK.sum(0) / ASK.sum(0); print(f"  interviews {len(it)} | full-sample shares equal questions.csv: {np.allclose(full, yi[ids].to_numpy())}")
def spn(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); return spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
for m in ("sonnet", "flash_think"):
    x0 = col("gss2024_interview", m).reindex(ids).to_numpy(float); x = corrected(m)[0].reindex(ids).to_numpy(float)
    print(f"  {m}: full-sample Spearman as run {spn(x0, full):.3f} -> corrected {spn(x, full):.3f} | error corrected {np.abs(x - full).mean():.2f} | mean {x0.mean():.2f}% -> {x.mean():.2f}%")
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
    x = corrected(m)[0].reindex(ids).to_numpy(float); print(f"  == {m}, corrected ==")
    for n in ((32, 107, 214) if m == "sonnet" else (32,)): run(x, n, ms=(5, 25, 50, 100) if m == "sonnet" and n in (32, 107) else (25,))

print("\n=== F. Web and telephone questions (Table 7, section 8, Appendix D.9) ===")
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
xi1 = col("gss2024_interview", "opus", 0)[issp]; xi3 = col("gss2024_interview", "opus")[issp]; xs = col("gss2024_interview", "sonnet")[issp]
print(f"  same 47 ISSP, interview prompt vs interview shares: Opus first call {sp(xi1, yi[issp]):.3f}, mean of 3 {sp(xi3, yi[issp]):.3f}; Sonnet {sp(xs, yi[issp]):.3f}")

print("\n=== G. Whole surveys, Eurobarometer face-to-face, Opus 5.5 (section 8, Appendix D.9) ===")
D = E.copy(); D["opus"] = col("eb_face_to_face", "opus")
tab = D.groupby("survey").apply(lambda s: pd.Series({"n": len(s), "real": s.real_dk_pct.mean(), "opus": s.opus.mean(), "ratio": s.opus.mean() / s.real_dk_pct.mean(), "rho_in": sp(s.opus, s.real_dk_pct) if len(s) >= 8 else np.nan}))
print(tab.round(2).sort_values("ratio").to_string())
D["res"] = D.opus.rank(pct=True) - D.real_dk_pct.rank(pct=True); D["agree"] = D.qtext.str.contains("agree", case=False)
o = D.sort_values("res"); under, over = o.head(20), o.tail(20)
print(f"  20 most under-ranked by survey: {under.survey.value_counts().to_dict()}")
print(f"  agree/disagree wording among the 40 largest rank misses: {int(under.agree.sum() + over.agree.sum())} of 40 | among all 374: {int(D.agree.sum())}")

print("\n=== H. Abstract, Introduction (Table 1), Conclusion ===")
print(f"  questions {len(Q)} | programmes {Q.programme.nunique()} {sorted(Q.programme.unique())} | sets {Q.set.value_counts().to_dict()}")
for m in ("opus", "sonnet", "flash_think"): print(f"  EB face-to-face 374 Spearman {m} {sp(col('eb_face_to_face', m), ye):.3f}")
print(f"  EB face-to-face question length: question text {sp(E.qlen_text, ye):.3f}, prompt (prompt_chars) {sp(E.prompt_chars, ye):.3f}")
t1 = E[E.block == "STD105:QA1"].copy(); t1["opus"] = col("eb_face_to_face", "opus")[t1.index]
print(t1[["qtext", "real_dk_pct", "opus"]].round(1).to_string()); print(f"  Table 1 Spearman {sp(t1.opus, t1.real_dk_pct):.3f}")
for m in ("opus", "sonnet", "haiku"): print(f"  EB face-to-face mean forecast {m} {col('eb_face_to_face', m).mean():.2f}% (real {ye.mean():.2f}%)")
