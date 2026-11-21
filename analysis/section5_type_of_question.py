"""Reproduces section 5.2 and Appendix D.4: Table D8 (the forecast inside question blocks, versions (a)-(c)), the comparison with
regressions on past results (scored from their saved leave-one-survey-out predictions in data/section5/regression_predictions.csv;
the fits themselves need scikit-learn and are not rerun here), and the comparison with rated familiarity and knowledge demand
(ridge regression on the four ratings, written out in numpy with the same leave-one-out choice of penalty as scikit-learn's RidgeCV)."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import rankdata, pearsonr
from section5_common import *

T = test_sets(); F = T["Eurobarometer face-to-face"].reset_index(drop=True); y = F.real_dk_pct.to_numpy(); blk = F.block.to_numpy()
FC = {"Opus 5.5": forecasts("Opus 5.5", "thinking"), "Sonnet 5.5": forecasts("Sonnet 5.5", "no thinking"), "Gemini Flash thinking": forecasts("Gemini Flash", "thinking"), "Haiku 5.5": forecasts("Haiku 5.5", "no thinking")}
for k, v in FC.items(): F[k] = v.reindex(F.qid).to_numpy()
R = pd.read_csv(S5 / "regression_predictions.csv").set_index("qid"); F = F.join(R, on="qid")

print("=== Table D8: inside question blocks, 374 face-to-face Eurobarometer questions ===")
for c in ["Opus 5.5", "Sonnet 5.5", "Gemini Flash thinking", "prompt_chars_len", "question_chars"]:
    x = F.prompt_chars if c == "prompt_chars_len" else F[c]; a, b, cc, n, g, k, pos = inside_blocks(x, y, blk)
    lab = "prompt length (question + options)" if c == "prompt_chars_len" else ("question length" if c == "question_chars" else c)
    print(f"  {lab:36s} (a) {a:.3f} | (b) {b:.3f} | (c) {cc:.3f} | {n} questions, {g} blocks; (c) uses {k} blocks, positive in {pos}")

print("\n=== Regressions on past results, 374 face-to-face questions (leave one survey out, repeats removed from training) ===")
WORDS = ["words_ridge250", "words_ridge3000", "words_forest"]; EMB = ["emb_ridge", "emb_ridge_simple", "emb_svr", "emb_svr_simple", "emb_knn"]
SAME = ["emb_samemode_ridge", "emb_samemode_svr", "emb_samemode_kridge", "emb_samemode_knn", "emb_samemode_avg3"]
N = F[F.new_question == 1].reset_index(drop=True); O = F[F.new_question != 1].reset_index(drop=True)
for lab, cols in (("word counts", WORDS), ("embeddings, all modes", EMB), ("embeddings, face-to-face surveys only", SAME)):
    for sname, s in (("374", F), ("100 new", N), ("274 repeated", O)):
        print(f"  {lab:38s} {sname:12s} " + " | ".join(f"{c} {fmt_ci(spearman(s[c], s.real_dk_pct), interval(s[c], s.real_dk_pct, s.block))}" for c in cols))
print("  forecast minus best all-mode embedding variant (emb_svr) on the 374, paired over block resamples:")
for m in ("Opus 5.5", "Sonnet 5.5", "Gemini Flash thinking", "Haiku 5.5"): print(f"    {m:22s} {fmt_ci(*paired_diff(F[m], F.emb_svr, y, blk), signed=True)}")
print(f"  inside blocks, emb_svr: {inside_blocks(F.emb_svr, y, blk)[0]:.3f}; Opus 5.5 {inside_blocks(F['Opus 5.5'], y, blk)[0]:.3f}; Sonnet 5.5 {inside_blocks(F['Sonnet 5.5'], y, blk)[0]:.3f}")
print("  forecast minus face-to-face-only embedding variants on the 100 new questions, paired over block resamples:")
for m in ("Opus 5.5", "Sonnet 5.5", "Gemini Flash thinking"): print(f"    {m:22s} " + " | ".join(f"minus {c} {fmt_ci(*paired_diff(N[m], N[c], N.real_dk_pct, N.block), signed=True)}" for c in ("emb_samemode_ridge", "emb_samemode_svr", "emb_samemode_kridge")))
dms = lambda s, c: s[c] - s.groupby("survey")[c].transform("mean")
print("  inside surveys on the 100 (survey means removed): " + ", ".join(f"{c} {spearman(dms(N, c), dms(N, 'real_dk_pct')):.3f}" for c in SAME + ["Opus 5.5", "Sonnet 5.5", "Gemini Flash thinking"]))
for sname, s in (("78 grouped new questions", N), ("326 grouped face-to-face questions", F)):
    print(f"  inside blocks, {sname}:")
    for c in SAME + ["Opus 5.5", "Sonnet 5.5", "Gemini Flash thinking"]:
        a, b, cc, n, g, k, pos = inside_blocks(s[c], s.real_dk_pct, s.block); print(f"    {c:22s} (a) {a:.3f} (b) {b:.3f} (c) {cc:.3f} | {n} questions, {g} blocks")
E = pd.read_csv(S5 / "eb_clean_rows.csv"); E["qk"] = E.question.str.lower().str.replace(r"[^a-z]", "", regex=True) + "|" + E["mode"]
def train_size(sv, same_mode=False):
    te = E.survey == sv; ok = (~te) & (~E.qk.isin(set(E.qk[te]))) & E.unique
    if same_mode: ok &= E["mode"] == E.loc[te, "mode"].iloc[0]
    return int(ok.sum())
a = [train_size(sv) for sv in E.survey.unique()]; b = [train_size(sv, True) for sv in E[E["mode"] == "face-to-face"].survey.unique()]
print(f"  training questions per held-out survey: all modes median {np.median(a):.0f} (range {min(a)} to {max(a)}); face-to-face only, face-to-face surveys, median {np.median(b):.0f} (range {min(b)} to {max(b)})")
U = E[E.unique]; cnt = E.groupby("qk").survey.nunique(); print(f"  questions (counted once) that appear in two or more surveys: {int((U.qk.map(cnt) >= 2).sum())} of {len(U)} ({int((U[U['mode'] == 'face-to-face'].qk.map(cnt) >= 2).sum())} face-to-face); clean rows {len(E)}, removed as repeats {len(E) - len(U)}")

# ---------------- rated familiarity and knowledge demand ----------------
print("\n=== Rated familiarity, knowledge needed, wording difficulty, personal relevance (Sonnet 5.5 and Opus 5.5 raters) ===")
FEAT = ["familiarity", "knowledge", "difficulty", "relevance"]; SIGN = {"familiarity": -1, "knowledge": 1, "difficulty": 1, "relevance": -1}
RT = pd.read_csv(S5 / "familiarity_ratings.csv"); RS = RT[RT.rater == "Sonnet 5.5"].set_index("qid")[FEAT]; RO = RT[RT.rater == "Opus 5.5"].set_index("qid")[FEAT]
print("  agreement of the two raters per rating, matched by question: " + ", ".join(f"{f} {spearman(RS[f], RO.reindex(RS.index)[f]):.3f}" for f in FEAT))
print("  the same, pairing the two reply files row by row without matching questions (as the scoring script did): " + ", ".join(f"{f} {spearman(RS[f].to_numpy(), RO[f].to_numpy()):.3f}" for f in FEAT) + f" | files in the same question order: {bool((RS.index == RO.index).all())}")
parts = []
for name in ("World Values Survey 7", "General Social Survey 2022", "General Social Survey 2024", "Eurobarometer face-to-face"):
    s = T[name]; first = name == "General Social Survey 2024"
    parts.append(pd.DataFrame({"qid": s.qid, "set": np.where(s.programme == "Eurobarometer", np.where(s.new_question == 1, "eb_new", "eb_rep"), s.set.str.replace("_interview", "")), "unit": np.where(s.programme == "Eurobarometer", "eb:" + s.survey, s.set.str.replace("_interview", "")),
                               "block": s.block, "y": s.real_dk_pct, "qlen": s.question_chars,
                               "opus": forecasts("Opus 5.5", "thinking").reindex(s.qid).to_numpy(), "sonnet": forecasts("Sonnet 5.5", "no thinking").reindex(s.qid).to_numpy()}))
D = pd.concat(parts, ignore_index=True); assert len(D) == 733
for nm, RR in (("son", RS), ("opu", RO)):
    for f in FEAT: D[f"{nm}_{f}"] = D.qid.map(RR[f])
for f in FEAT: D[f"avg_{f}"] = (D[f"son_{f}"] + D[f"opu_{f}"]) / 2
for r in ("son", "opu", "avg"): D[f"{r}_sum"] = sum(SIGN[f] * D[f"{r}_{f}"] for f in FEAT)
ALPHAS = np.logspace(-3, 2, 16)
def ridge_cv_predict(Xa, ya, Xb):
    """Ridge with an unpenalised intercept; the penalty is chosen from ALPHAS by exact leave-one-out squared error (as RidgeCV)."""
    mx, my = Xa.mean(0), ya.mean(); Xc, yc = Xa - mx, ya - my; n = len(ya); best = None
    for a in ALPHAS:
        A = np.linalg.solve(Xc.T @ Xc + a * np.eye(Xc.shape[1]), Xc.T); w = A @ yc; h = 1 / n + np.einsum("ij,ji->i", Xc, A)
        loo = np.mean(((yc - Xc @ w) / (1 - h)) ** 2)
        if best is None or loo < best[0]: best = (loo, w)
    return (Xb - mx) @ best[1] + my
def fit_out(rater, extra=()):
    cols = [f"{rater}_{f}" for f in FEAT] + list(extra); out = pd.Series(np.nan, index=D.index)
    for u in D.unit.unique():
        tr, te = D[(D.unit != u) & D[cols].notna().all(1)], D[D.unit == u]
        out[te.index] = ridge_cv_predict(tr[cols].to_numpy(float), np.log1p(tr.y.to_numpy(float)), te[cols].fillna(tr[cols].mean()).to_numpy(float))
    return out
for r in ("son", "opu", "avg"): D[f"{r}_fit"] = fit_out(r); D[f"{r}_fitlen"] = fit_out(r, ["qlen"])
def partial(x, yv, Z):
    x, yv = rankdata(x), rankdata(yv); Z = np.column_stack([rankdata(z) for z in Z]); A = np.column_stack([np.ones(len(x)), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]; ry = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]; return pearsonr(rx, ry)[0]
def inside_all(s, c): return spearman(s[c] - s.groupby("block")[c].transform("mean"), s.y - s.groupby("block").y.transform("mean"))  # blocks of one included, as scored
SETS = {"World Values Survey 7": D.set == "wvs7", "General Social Survey 2022": D.set == "gss2022", "General Social Survey 2024": D.set == "gss2024", "Eurobarometer face-to-face (374)": D.set.str.startswith("eb"), "Eurobarometer new (100)": D.set == "eb_new"}
for nm, m in SETS.items():
    s = D[m]; print(f"\n  == {nm}: {len(s)} questions, {s.block.nunique()} blocks ==")
    print("     single ratings (signed), Sonnet-rated | Opus-rated: " + "; ".join(f"{f} {SIGN[f] * spearman(s['son_' + f], s.y):+.3f} | {SIGN[f] * spearman(s['opu_' + f], s.y):+.3f}" for f in FEAT))
    for r in ("son", "opu", "avg"): print(f"     {r}: unweighted sum {spearman(s[r + '_sum'], s.y):+.3f} | trained on other surveys {spearman(s[r + '_fit'], s.y):+.3f} | with question length {spearman(s[r + '_fitlen'], s.y):+.3f}")
    best = max(("avg_fit", "avg_fitlen", "avg_sum"), key=lambda c: spearman(s[c], s.y))
    print(f"     forecasts Opus {spearman(s.opus, s.y):.3f}, Sonnet {spearman(s.sonnet, s.y):.3f} | best rating baseline {best} {spearman(s[best], s.y):.3f} | partial Spearman with the four averaged ratings held fixed: Opus {partial(s.opus, s.y, [s[f'avg_{f}'] for f in FEAT]):.3f}")
    if s.block.nunique() > 20: print(f"     inside blocks (block means removed, blocks of one included): Opus {inside_all(s, 'opus'):.3f}, Sonnet {inside_all(s, 'sonnet'):.3f}, trained ratings {inside_all(s, 'avg_fit'):.3f}, unweighted sum {inside_all(s, 'avg_sum'):.3f}")
    rng = np.random.default_rng(3); gq = s.block.to_numpy(); ug = np.unique(gq); ix = {u: np.where(gq == u)[0] for u in ug}
    for fc in ("opus", "sonnet"):
        dd = []
        for _ in range(2000):
            k = np.concatenate([ix[u] for u in rng.choice(ug, len(ug))]); t = s.iloc[k]; dd.append(spearman(t[fc], t.y) - spearman(t[best], t.y))
        print(f"     {fc} minus best rating baseline {spearman(s[fc], s.y) - spearman(s[best], s.y):+.3f} [{np.nanpercentile(dd, 2.5):+.2f}, {np.nanpercentile(dd, 97.5):+.2f}]")
