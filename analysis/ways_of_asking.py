"""Section 6 (Table 5: ways of asking 1, 1b, 2, 3 on 359 questions) and Appendix D.7 (Tables D12, D13, D14, the real-people
benchmark, the profile checks, the open-weight models and the first run on the General Social Survey 2024).
Reads data/ and the respondent-level profile files that build/ways_of_asking_profiles.py writes to raw/derived/.
Ways: 1 answer as the person; 1b the same, told "don't know" is acceptable; 2 state your chance of "don't know";
3 the population forecast of section 5 (data/forecasts_main.csv)."""
from common import *
from scipy.stats import rankdata
Q = questions(); SV = [("gss2024", "gss2024_interview"), ("gss2022", "gss2022"), ("wvs7", "wvs7")]
Q = pd.concat([Q[Q.set == s] for _, s in SV]).reset_index(drop=True); Q["survey"] = Q.set.map({s: k for k, s in SV}); real = Q.set_index("qid").real_dk_pct
print(f"questions {len(Q)} {Q.survey.value_counts().to_dict()} | blocks {Q.groupby('survey').block.nunique().to_dict()} | real mean share {Q.real_dk_pct.mean():.2f}%")
MODELS = ["Opus 5.5", "Sonnet 5.5", "Gemini Flash"]
# ---- way 3: the main forecasts. Opus 5.5: mean of its calls (three on GSS 2024, one on WVS 7 and GSS 2022); Sonnet 5.5: mean of its calls;
# Gemini Flash: no thinking (GSS 2024) and the earlier no-thinking run (WVS 7, GSS 2022).
W3 = {"Opus 5.5": forecasts("Opus 5.5", "thinking"), "Sonnet 5.5": forecasts("Sonnet 5.5", "no thinking"),
      "Gemini Flash": pd.concat([forecasts("Gemini Flash", "no thinking"), forecasts("Gemini Flash", "no thinking, earlier run")])}
W3["Gemini Flash"] = W3["Gemini Flash"][~W3["Gemini Flash"].index.duplicated()]
# ---- ways 1, 1b, 2: one value per call (100 = "don't know", 0 = an answer; way 2 the stated chance 0-100; unreadable = NaN)
C = pd.read_csv(DATA / "replies/ways_of_asking_calls.csv"); C = C.merge(Q[["qid", "survey"]], on="qid")
C["x"] = np.where(C.way == "2", C.number.where(C.number.between(0, 100)), np.where(C.dk_bracket == 1, 100.0, np.where(C.number.notna(), 0.0, np.nan)))
print(f"calls for ways 1, 1b and 2: {len(C)} | per model and way {C.groupby(['model', 'way']).size().to_dict()}")
print("profiles per question:", C.groupby(["model", "way"]).profile.nunique().to_dict(), "| unreadable", C.groupby("model").x.apply(lambda v: int(v.isna().sum())).to_dict())

def qmean(model, way, sv=None):
    """Per-question average over calls (NaN where the way was not run)."""
    if way == "3": return W3[model].reindex(Q.qid if sv is None else Q[Q.survey == sv].qid)
    c = C[(C.model == model) & (C.way == way)]
    return c.groupby("qid").x.mean().reindex(Q.qid if sv is None else Q[Q.survey == sv].qid)

def level(model, way, sv=None):
    if way == "3": x = qmean(model, "3", sv); return x.mean()
    c = C[(C.model == model) & (C.way == way)]; c = c if sv is None else c[c.survey == sv]; return c.x.mean()

WAYS = [w for w in ("1", "1b", "2", "3")]; ran = lambda m, w: w == "3" or ((C.model == m) & (C.way == w)).any()
print("\n=== Table D12: Spearman with the real shares by survey (average share or stated chance) ===")
for m in MODELS:
    for w in WAYS:
        if not ran(m, w): continue
        cells = []
        for sv, _ in SV:
            x = qmean(m, w, sv); r = spearman(x, real.reindex(x.index)); cells.append("none said" if np.isnan(r) else f"{r:.3f} ({level(m, w, sv):.2f}%)")
        said = "" if w not in ("1", "1b") else " | questions with any 'don't know': " + ", ".join(str(int((qmean(m, w, sv) > 0).sum())) for sv, _ in SV)
        print(f"  {m:12s} way {w:2s} " + " | ".join(cells) + said)

# ---- Table 5 / Table D13: mean over the three surveys; blocks resampled 2,000 times inside each survey (seed 0, surveys in the order above)
svix = [np.where(Q.survey == sv)[0] for sv, _ in SV]; tb = (Q.survey + Q.block).to_numpy(); y = Q.real_dk_pct.to_numpy(); rng = np.random.default_rng(0)
blocks = [{u: ix[tb[ix] == u] for u in np.unique(tb[ix])} for ix in svix]
draws = [[np.concatenate([b[u] for u in rng.choice(list(b), len(b))]) for b in blocks] for _ in range(2000)]
XQ = {(m, w): qmean(m, w).to_numpy() for m in MODELS for w in WAYS if ran(m, w)}
def means(idx, m):
    out = {}
    for w in WAYS:
        if (m, w) in XQ:
            v = [spearman(XQ[(m, w)][ix], y[ix]) for ix in idx]; out[w] = np.nanmean(v) if np.isfinite(v).any() else np.nan
    return out
print("\n=== Table 5 and Table D13: mean over the three surveys (way 1: over the surveys that can be scored), 95% interval; level = all calls pooled ===")
for m in MODELS:
    base = means(svix, m); bs = [means(d, m) for d in draws]; ci = lambda k: np.nanpercentile([b[k] for b in bs], [2.5, 97.5])
    print(f"  {m}: " + " | ".join(f"way {w} {base[w]:.3f} [{ci(w)[0]:.3f}, {ci(w)[1]:.3f}] ({level(m, w):.2f}%)" for w in base))
    for a, b in (("2", "1"), ("3", "2")):
        d = np.nanpercentile([x[a] - x[b] for x in bs], [2.5, 97.5]); print(f"      way {a} minus way {b}: {base[a] - base[b]:+.3f} [{d[0]:+.4f}, {d[1]:+.4f}]")
    s1 = {sv: spearman(qmean(m, "1", sv), real.reindex(Q[Q.survey == sv].qid)) for sv, _ in SV}; s2 = {sv: spearman(qmean(m, "2", sv), real.reindex(Q[Q.survey == sv].qid)) for sv, _ in SV}
    both = [k for k in s1 if not np.isnan(s1[k])]
    print(f"      way 2 minus way 1 on only the surveys where way 1 can be scored ({', '.join(both)}): {np.mean([s2[k] for k in both]) - np.mean([s1[k] for k in both]):+.3f}")
    print(f"      way 2 level / real mean share: {level(m, '2') / Q.real_dk_pct.mean():.2f}")
W3["Opus 5.5"] = forecasts("Opus 5.5", "thinking", 0); XQ[("Opus 5.5", "3")] = qmean("Opus 5.5", "3").to_numpy()
base = means(svix, "Opus 5.5"); bs = [means(d, "Opus 5.5") for d in draws]; d = np.nanpercentile([x["3"] - x["2"] for x in bs], [2.5, 97.5])
print(f"  Check: Opus 5.5 way 3 with only the first of its three GSS 2024 calls: way 3 {base['3']:.3f} ({level('Opus 5.5', '3'):.2f}%) | way 3 minus way 2 {base['3'] - base['2']:+.3f} [{d[0]:+.4f}, {d[1]:+.4f}]")
W3["Opus 5.5"] = forecasts("Opus 5.5", "thinking"); XQ[("Opus 5.5", "3")] = qmean("Opus 5.5", "3").to_numpy()

# ---- Table D14: do the stated chances of way 2 pick out the person-question pairs with a real "don't know"?
def auc(lab, s):
    lab, s = np.asarray(lab, bool), np.asarray(s, float); r = rankdata(s); n1, n0 = lab.sum(), (~lab).sum(); return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
RA = pd.read_csv(raw_file("derived/ways_of_asking_real_answers.csv")); PR = pd.read_csv(raw_file("derived/ways_of_asking_profiles.csv"))
print("\n=== Table D14: way 2, person-question pairs with a real answer, three surveys pooled ===")
for m in MODELS:
    rr = C[(C.model == m) & (C.way == "2")].merge(RA.rename(columns={"dk": "real"}), on=["survey", "qid", "profile"]).dropna(subset=["real", "x"])
    wq = rr.x - rr.groupby("qid").x.transform("mean")
    print(f"  {m:12s} pairs {len(rr)} (real 'don't know' {int(rr.real.sum())}) | AUC {auc(rr.real, rr.x):.3f} | with each question's average removed {auc(rr.real, wq):.3f}")

# ---- the same real people as a benchmark for way 1
print("\n=== Real people behind the profiles, scored like way 1 (first n profiles; Spearman with the full-sample shares) ===")
RB = {}
for n in (10, 20, 50):
    for sv, _ in SV:
        r = RA[(RA.survey == sv) & (RA.profile < n)].dropna(subset=["dk"]); x = (100 * r.groupby("qid").dk.mean()).reindex(Q[Q.survey == sv].qid); RB[(n, sv)] = spearman(x, real.reindex(x.index))
    pooled = 100 * RA[RA.profile < n].dk.mean()
    print(f"  first {n:2d}: " + ", ".join(f"{sv} {RB[(n, sv)]:.3f}" for sv, _ in SV) + f" | mean of gss2022 and wvs7 {np.mean([RB[(n, 'gss2022')], RB[(n, 'wvs7')]]):.3f} | mean of all three {np.mean([RB[(n, sv)] for sv, _ in SV]):.3f} | their 'don't know' rate, pooled {pooled:.2f}%")
print("  profiles used in way 1 per model:", {m: C[(C.model == m) & (C.way == "1")].profile.max() + 1 for m in MODELS})
for sv, _ in SV:
    r = RA[RA.survey == sv]; print(f"  {sv}: the 50 people gave {int(r.dk.notna().sum())} answers, {int(r.dk.sum())} 'don't know' ({100 * r.dk.mean():.2f}%), {int((r.groupby('profile').dk.max() > 0).sum())} of 50 at least once")
p24 = PR[PR.survey == "gss2024"].sort_values("profile"); print(f"  GSS 2024 profiles by mode: first 20 {p24['mode'].iloc[:20].value_counts().to_dict()}, other 30 {p24['mode'].iloc[20:].value_counts().to_dict()} (1 face-to-face, 2 telephone, 4 web)")

# ---- the profile adds little (way 2)
print("\n=== Way 2: one profile alone, agreement between profiles, agreement with way 3 ===")
for m in MODELS:
    c = C[(C.model == m) & (C.way == "2")]
    def r_(cc): return np.mean([spearman(cc[cc.survey == sv].groupby("qid").x.mean(), real.reindex(cc[cc.survey == sv].groupby("qid").x.mean().index)) for sv, _ in SV])
    piv = c[c.survey == "gss2024"].pivot(index="qid", columns="profile", values="x"); cm = piv.corr(method="spearman").to_numpy()
    w3 = np.mean([spearman(qmean(m, "2", sv), qmean(m, "3", sv)) for sv, _ in SV])
    print(f"  {m:12s} all profiles {r_(c):.3f} | first profile alone {r_(c[c.profile == 0]):.3f} | GSS 2024 mean pairwise agreement between profiles {cm[np.triu_indices_from(cm, 1)].mean():.3f} | way 2 against way 3 {w3:.3f}")

# ---- GSS 2024 first run (55 interviewer-run and 47 web questions)
codes = pd.read_csv(DATA / "gss2024_answer_codes.csv").set_index("qid"); allq = questions().set_index("qid")
FR = pd.read_csv(DATA / "replies/ways_of_asking_first_run.csv", keep_default_na=False)
def ans(t, qid):
    if t == "": return "unreadable"
    if t.upper() in ("DK", "DON'T", "DONT"): return "dk"
    if t.lstrip("-").isdigit():
        web = qid.startswith("gss24_web"); sub = [int(s) for s in str(codes.subst_codes[qid]).split()]
        return "dk" if (web and int(t) == codes.dk_code_web[qid]) else ("answer" if int(t) in sub else "unreadable")
    return "unreadable"
FR["a"] = [ans(t, q) for t, q in zip(FR.token, FR.qid)]; FR["arm"] = np.where(FR.qid.str.startswith("gss24_web"), "web", "interview")
print("\n=== GSS 2024 first run: simulated respondents ===")
for (run, m), g in FR.groupby(["run", "model"], sort=False):
    print(f"  {run:24s} {m:10s} answers {len(g)} | " + " | ".join(f"{arm}: {len(a)} calls, 'don't know' {int((a.a == 'dk').sum())} ({100 * (a.a == 'dk').mean():.2f}%), on {a[a.a == 'dk'].qid.nunique()} questions, unreadable {int((a.a == 'unreadable').sum())}" for arm, a in g.groupby("arm")))
print(f"  real mean share: 55 interviewer-run {allq.real_dk_pct.reindex(codes.index[codes.index.str.startswith('gss24_int')]).mean():.2f}%, 47 web {allq.real_dk_pct.reindex(codes.index[codes.index.str.startswith('gss24_web')]).mean():.2f}%")

# ---- open-weight models on GSS 2024 (20 profiles of the first run)
OW = pd.read_csv(DATA / "replies/ways_of_asking_open_weight.csv"); OW["arm"] = np.where(OW.qid.str.startswith("gss24_web"), "web", "interview")
OW["said"] = np.where(OW.arm == "interview", OW.dk_bracket == 1, OW.number == OW.qid.map(codes.dk_code_web))
print("\n=== Open-weight models as simulated respondents (interviewer-run questions for the token probabilities) ===")
for (m, run), g in OW[OW.run != "way 2"].groupby(["model", "run"]):
    i = g[g.arm == "interview"]; q = i.groupby("qid").dk_token_prob.mean(); out = i[i.dk_in_top20 == 0]
    bound = np.minimum(out.top20_min_prob, np.maximum(0, 1 - out.top20_sum)).max() if len(out) else np.nan
    print(f"  {m:16s} {run:24s} answers {len(g)}, 'don't know' {int(g.said.sum())} | interviewer calls {len(i)}: DK among the 20 likeliest on {100 * i.dk_in_top20.mean():.1f}%, mean probability {100 * i.dk_token_prob.mean():.7f}% | "
          f"largest possible probability on the other calls {100 * bound:.2e}% | Spearman of the question mean with real shares {spearman(q, real.reindex(q.index)):.3f}")
L = OW[(OW.run == "way 2")].copy(); L["x"] = L.number.where(L.number.between(0, 100)); q24 = Q[Q.survey == "gss2024"]
m2 = L.groupby("qid").x.mean().reindex(q24.qid); lo, hi = interval(m2, q24.real_dk_pct, q24.block)
print(f"  Llama 3.3 70B way 2: calls {len(L)} (readable {int(L.x.notna().sum())}, profiles {L.profile.nunique()}) | Spearman {spearman(m2, q24.real_dk_pct):.3f} [{lo:.2f}, {hi:.2f}] | mean stated chance {m2.mean():.3f}%")
fl = pd.read_csv(DATA / "forecasts_llama_gss2024.csv").set_index("qid").llama33_70b.reindex(q24.qid); lo, hi = interval(fl, q24.real_dk_pct, q24.block)
print(f"  Llama 3.3 70B way 3 (mean of 3 forecast calls): Spearman {spearman(fl, q24.real_dk_pct):.3f} [{lo:.2f}, {hi:.2f}] | mean forecast {fl.mean():.2f}% | real mean {q24.real_dk_pct.mean():.2f}% | blocks {q24.block.nunique()}")
