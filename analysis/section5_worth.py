"""Reproduces section 5.4, Figure 2 and Appendix D.6 (Table D11): the ceiling from the 84 questions repeated in two Standard
Eurobarometer waves, flagging the questions with the highest values (the share of problem questions caught), random flagging,
the cost per 100 questions, and the bar values of Figure 2. Same seed and draw order as the original scoring (one generator, seed 11)."""
import warnings, numpy as np, pandas as pd
from scipy.stats import rankdata
from section5_common import *
warnings.filterwarnings("ignore")

E = pd.read_csv(S5 / "eb_clean_rows.csv"); E["qk"] = E.question.str.lower().str.replace(r"[^a-z]", "", regex=True) + "|" + E["mode"]
d = E[E.survey.isin(["STD104", "STD105"])]; w = d.pivot_table(index="qk", columns="survey", values="real_dk_pct").dropna()
print(f"=== Ceiling: STD104 (autumn 2025) and STD105 (spring 2026) share {len(w)} questions; real share of one wave against the other: Spearman {spearman(w.STD104, w.STD105):.3f}, mean absolute difference {(w.STD104 - w.STD105).abs().mean():.2f} points")

T = test_sets(); F = T["Eurobarometer face-to-face"].reset_index(drop=True)
for lab, (m, st) in {"opus": ("Opus 5.5", "thinking"), "sonnet": ("Sonnet 5.5", "no thinking"), "flash_think": ("Gemini Flash", "thinking"), "haiku": ("Haiku 5.5", "no thinking")}.items():
    F[lab] = forecasts(m, st).reindex(F.qid).to_numpy()
F = F.join(pd.read_csv(S5 / "regression_predictions.csv").set_index("qid")[["emb_svr", "emb_samemode_kridge"]], on="qid")
def auc(bad, v):
    r = rankdata(v); n1 = bad.sum(); n0 = len(v) - n1; return (r[bad].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(11)
def caught(score, y, T_):
    bad = y >= T_; k = int(bad.sum()); order = np.lexsort((rng.random(len(y)), -np.asarray(score, float))); flag = np.zeros(len(y), bool); flag[order[:k]] = True
    return (flag & bad).sum() / k
print("\n=== Table D11: flag as many questions as there are problem questions (the highest values by each method) ===")
for lab, s in (("374 face-to-face questions", F), ("100 new questions", F[F.new_question == 1].reset_index(drop=True))):
    y = s.real_dk_pct.to_numpy(float)
    for T_ in (10, 8):
        bad = y >= T_; k = int(bad.sum()); print(f"  {lab}: {k} questions with a real share of at least {T_}%")
        for nm, col in (("Opus 5.5", "opus"), ("Sonnet 5.5", "sonnet"), ("Gemini Flash, thinking", "flash_think"), ("Haiku 5.5", "haiku"), ("question length (question line)", "question_chars"),
                        ("regression, embeddings, all modes (svr)", "emb_svr"), ("regression, embeddings, face-to-face only (kernel ridge)", "emb_samemode_kridge")):
            v = s[col].to_numpy(float); print(f"     {nm:58s} catches {100 * np.mean([caught(v, y, T_) for _ in range(200)]):5.1f}% | AUC {auc(bad, v):.3f}")
        for kk in (50, 200):
            c, a = [], []
            for _ in range(2000):
                p = rng.binomial(kk, y / 100) / kk; c.append(caught(p, y, T_)); a.append(auc(bad, p + 1e-9 * rng.random(len(y))))
            print(f"     {'sampled pilot of ' + str(kk) + ' answers per question':58s} catches {100 * np.mean(c):5.1f}% | AUC {np.mean(a):.3f}")
        print(f"     {'random flagging (expected share)':58s} catches {100 * k / len(s):5.1f}%")
# whole-prompt length, as a separate run so the draws above stay in the original order
rng = np.random.default_rng(11)
for lab, s in (("374", F), ("100 new", F[F.new_question == 1].reset_index(drop=True))):
    y = s.real_dk_pct.to_numpy(float); print(f"  whole-prompt length (question + options), {lab}, 10%: catches {100 * np.mean([caught(s.prompt_chars.to_numpy(float), y, 10) for _ in range(200)]):.1f}%")

print("\n=== Cost of the first Eurobarometer run (753 calls per model; list prices per million tokens as in the run script: Sonnet 5.5 $2 in / $10 out, Opus 5.5 $4 in / $20 out) ===")
TK = pd.read_csv(S5 / "eb_first_run_tokens.csv"); PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
for m, g in TK.groupby("model", sort=False):
    g = g.iloc[:753]; usd = (g.input_tokens.sum() * PRICE[m][0] + g.output_tokens.sum() * PRICE[m][1]) / 1e6
    print(f"  {m}: {len(g)} calls, ${usd:.4f}, i.e. {100 * usd / len(g) * 100:.1f} US cents per 100 questions at one call each")

print("\n=== Figure 2 bar values (374 face-to-face questions) ===")
y = F.real_dk_pct.to_numpy(); R = pd.read_csv(S5 / "regression_predictions.csv").set_index("qid").reindex(F.qid)
print(f"  question length: question line {spearman(F.question_chars, y):.3f}, whole prompt {spearman(F.prompt_chars, y):.3f}")
wc = [spearman(R[c], y) for c in ("words_ridge250", "words_ridge3000", "words_forest")]; em = [spearman(R[c], y) for c in ("emb_ridge", "emb_ridge_simple", "emb_svr", "emb_svr_simple", "emb_knn")]
sm = [spearman(R[c], y) for c in ("emb_samemode_ridge", "emb_samemode_svr", "emb_samemode_kridge", "emb_samemode_knn", "emb_samemode_avg3")]
print(f"  word-count regressions {min(wc):.3f} to {max(wc):.3f}; embedding regressions, all modes {min(em):.3f} to {max(em):.3f}; face-to-face only {min(sm):.3f} to {max(sm):.3f}")
print(f"  forecasts (mean of three calls): Sonnet 5.5 {spearman(F.sonnet, y):.3f}, Opus 5.5 {spearman(F.opus, y):.3f}; pilots: see table3_intervals_and_pilots.py (k=50 0.749, k=200 0.916); ceiling above")
