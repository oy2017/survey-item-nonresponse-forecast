"""Reproduces Appendix D.3 Tables D3 (four rewritten prompts), D4 (ten examples in the prompt), D5 (all models on the 374 face-to-face
Eurobarometer questions), D6 (open-weight models), D7 (each country's own shares) and the notes around them; also the 5.1 sentences on
rewording and on the spread across models. Intervals are 2,000 resamples (seed 0). The scoring scripts behind the paper resampled
whole surveys for Table D3 and for the Eurobarometer rows of Table D4; both that unit and question blocks (the rule stated in the paper)
are printed."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from section5_common import *

def boot_ci(x, y, units, n=2000, seed=0):
    """Percentile interval over resamples of whole units; resamples where either side does not vary are skipped."""
    x, y = np.asarray(x, float), np.asarray(y, float); v = [spearman(x[i], y[i]) for i in block_resamples(units, n, seed)]; return np.nanpercentile(v, [2.5, 97.5])
def diff_ci(a, b, y, units, n=2000, seed=0):
    a, b, y = (np.asarray(v, float) for v in (a, b, y)); d = [spearman(a[i], y[i]) - spearman(b[i], y[i]) for i in block_resamples(units, n, seed)]; return np.nanpercentile(d, [2.5, 97.5])

T = test_sets(); F = T["Eurobarometer face-to-face"].reset_index(drop=True); y = F.real_dk_pct.to_numpy(); blk = F.block.to_numpy(); srv = F.survey.to_numpy()
new = (F.new_question == 1).to_numpy(); ing = (F.groupby("block").qid.transform("size") >= 2).to_numpy()
dm = lambda v: np.asarray(v, float) - pd.Series(np.asarray(v, float)).groupby(blk).transform("mean").to_numpy()
SC = pd.read_csv(S5 / "eb_single_calls.csv")

# ---------------- Table D3 ----------------
print("=== Table D3: four rewritten prompts, 374 face-to-face Eurobarometer questions, one call per question ===")
PV = pd.read_csv(S5 / "forecasts_prompt_variants.csv"); BASE = {"Opus 5.5": ("Opus 5.5", "thinking"), "Sonnet 5.5": ("Sonnet 5.5", "no thinking"), "Gemini Flash thinking": ("Gemini Flash", "thinking")}
lo_all, hi_all = [], []
for lab, (m, st) in BASE.items():
    z1 = one_call(m, st, F.qid).to_numpy()  # first call
    if lab != "Gemini Flash thinking": z1true = SC[(SC.model == m) & (SC.draw == 0)].set_index("qid").forecast_pct.reindex(F.qid).to_numpy()
    else: z1true = z1
    print(f"\n  {lab}: original, one call {spearman(z1, y):.3f} | survey-resampled [{', '.join(f'{v:.2f}' for v in boot_ci(z1, y, srv))}] | block-resampled [{', '.join(f'{v:.2f}' for v in boot_ci(z1, y, blk))}]"
          f" | true first call on all 374 {spearman(z1true, y):.3f} | inside blocks {spearman(dm(z1)[ing], dm(y)[ing]):.3f} | mean forecast {np.nanmean(z1):.2f}% (real {y.mean():.2f}%) | average of calls {spearman(forecasts(m, st).reindex(F.qid), y):.3f}")
    for v in ("norole", "nomode", "paraphrase", "minimal"):
        x = PV[(PV.model == lab) & (PV.variant == v)].set_index("qid").forecast_pct.reindex(F.qid).to_numpy(); bad = int(np.isnan(x).sum()); xf = np.where(np.isnan(x), np.nanmedian(x), x)
        r = spearman(x, y); lo_all.append(r); hi_all.append(r)
        print(f"    {v:10s} {r:.3f} | survey-resampled [{', '.join(f'{c:.2f}' for c in boot_ci(xf, y, srv))}] | block-resampled [{', '.join(f'{c:.2f}' for c in boot_ci(xf, y, blk))}]"
              f" | minus original one call {r - spearman(z1, y):+.3f} survey-resampled [{', '.join(f'{c:+.2f}' for c in diff_ci(xf, z1, y, srv))}] block-resampled [{', '.join(f'{c:+.3f}' for c in diff_ci(xf, z1, y, blk))}]"
              f" | inside blocks {spearman(dm(xf)[ing], dm(y)[ing]):.3f} | mean forecast {np.nanmean(x):.2f}% | no readable number {bad}")
print(f"\n  range over the 12 rewritten-prompt scores: {min(lo_all):.3f} to {max(hi_all):.3f}")
# Gemini gave no readable number on some questions, which are left out of its scores above. The same comparison on the
# questions that every version of the prompt answered (added after the scores were seen).
GV = PV[PV.model == "Gemini Flash thinking"].pivot_table(index="qid", columns="variant", values="forecast_pct", aggfunc="first").reindex(F.qid)
cq = GV.dropna().index; yq = F.set_index("qid").real_dk_pct.reindex(cq)
print(f"  Gemini Flash thinking on the {len(cq)} questions every version answered: original {spearman(forecasts('Gemini Flash', 'thinking', 0).reindex(cq), yq):.3f} | "
      + " | ".join(f"{v} {spearman(GV.loc[cq, v], yq):.3f}" for v in ("norole", "nomode", "paraphrase", "minimal")))

# ---------------- Table D4 ----------------
print("\n=== Table D4: ten example questions with their real shares in the prompt (mean of three calls) ===")
FS = pd.read_csv(S5 / "forecasts_fewshot.csv")
TGT = {"gss2024": T["General Social Survey 2024"].reset_index(drop=True), "ebnew": T["Eurobarometer, new questions"].reset_index(drop=True)}
for lab, (m, st) in (("Opus 5.5", BASE["Opus 5.5"]), ("Sonnet 5.5", BASE["Sonnet 5.5"]), ("Haiku 5.5", ("Haiku 5.5", "no thinking"))):
    for tg, t in TGT.items():
        yy = t.real_dk_pct.to_numpy(); z = forecasts(m, st).reindex(t.qid).to_numpy(); units_old = t.block.to_numpy() if tg == "gss2024" else t.survey.to_numpy()
        print(f"  {lab:10s} {tg:8s} no examples {spearman(z, yy):.3f}, error {mae(z, yy):.2f}, mean forecast {np.nanmean(z):.2f}% (real {yy.mean():.2f}%)")
        for cond in ("same", "other"):
            r = FS[(FS.model == lab) & (FS.target == tg) & (FS.examples_from == cond)]; x = r.pivot(index="qid", columns="call", values="forecast_pct").reindex(t.qid).mean(axis=1).to_numpy()
            print(f"      examples from the {cond:5s} programme: {spearman(x, yy):.3f} | as scored (GSS blocks / EB surveys) [{', '.join(f'{c:.2f}' for c in boot_ci(x, yy, units_old))}] change {spearman(x, yy) - spearman(z, yy):+.3f} [{', '.join(f'{c:+.2f}' for c in diff_ci(x, z, yy, units_old))}]"
                  f" | blocks [{', '.join(f'{c:.2f}' for c in boot_ci(x, yy, t.block))}] change [{', '.join(f'{c:+.2f}' for c in diff_ci(x, z, yy, t.block))}] | error {mae(x, yy):.2f} | mean of examples shown {r.mean_of_examples_shown.mean():.2f}%")

# ---------------- Table D5 and D6 ----------------
print("\n=== Table D5: all models on the 374 face-to-face Eurobarometer questions (interval: blocks) | inside blocks, version (a) ===")
OW = pd.read_csv(S5 / "forecasts_open_weight.csv"); ow = lambda m: OW[OW.model == m].groupby("qid").forecast_pct.mean()
ROWS = [("Haiku 5.5", forecasts("Haiku 5.5", "no thinking")), ("Llama 3.3 70B", ow("Llama 3.3 70B")), ("Llama 4 Maverick", ow("Llama 4 Maverick")),
        ("Sonnet 4.5", forecasts("Sonnet 4.5", "no thinking")), ("Gemini Flash-Lite", forecasts("Flash-Lite", "no thinking")), ("Opus 4.5", forecasts("Opus 4.5", "no thinking")),
        ("Sonnet 5.5", forecasts("Sonnet 5.5", "no thinking")), ("Gemini Flash without thinking", forecasts("Gemini Flash", "no thinking")),
        ("Gemini Flash with thinking", forecasts("Gemini Flash", "thinking")), ("Opus 5.5", forecasts("Opus 5.5", "thinking"))]
for lab, f in ROWS:
    x = f.reindex(F.qid).to_numpy(); a = inside_blocks(x, y, blk)
    print(f"  {lab:32s} {fmt_ci(spearman(x, y), boot_ci(x, y, blk))} | inside blocks {a[0]:.3f} ({a[3]} questions, {a[4]} blocks)")
g = T["General Social Survey 2024"]; gy = g.real_dk_pct.to_numpy()
fl, fn = forecasts("Gemini Flash", "thinking").reindex(g.qid).to_numpy(), forecasts("Gemini Flash", "no thinking").reindex(g.qid).to_numpy()
print(f"  thinking on General Social Survey 2024: Gemini Flash with {spearman(fl, gy):.3f}, without {spearman(fn, gy):.3f}, difference {fmt_ci(*paired_diff(fl, fn, gy, g.block), signed=True)}")
fl, fn = forecasts("Gemini Flash", "thinking").reindex(F.qid).to_numpy(), forecasts("Gemini Flash", "no thinking").reindex(F.qid).to_numpy()
print(f"  thinking on the 374: inside blocks with {inside_blocks(fl, y, blk)[0]:.3f}, without {inside_blocks(fn, y, blk)[0]:.3f}")
print("\n=== Table D6: open-weight models (mean of three calls; interval: blocks) ===")
N = T["Eurobarometer, new questions"]
for name, s in (("General Social Survey 2024, interviewer (55)", g), ("Eurobarometer face-to-face (374)", F), ("Eurobarometer, new questions (100)", N)):
    yy = s.real_dk_pct.to_numpy(); out = f"  {name:46s}"
    for m in ("Llama 3.3 70B", "Llama 4 Maverick"):
        x = ow(m).reindex(s.qid).to_numpy(); out += f" | {m} {fmt_ci(spearman(x, yy), boot_ci(x, yy, s.block))}"
        if name.startswith("Euro"): out += f" inside blocks {inside_blocks(x, yy, s.block)[0]:.3f}"
    print(out + f" | Opus 5.5 mean {spearman(forecasts('Opus 5.5', 'thinking').reindex(s.qid), yy):.3f}, first call {spearman(one_call('Opus 5.5', 'thinking', s.qid), yy):.3f}")

# ---------------- Table D7 ----------------
print("\n=== Table D7: the EU-wide forecast against each country's own shares ===")
E = pd.read_csv(S5 / "eb_clean_rows.csv"); has = E[COUNTRIES].notna().sum(1) >= 25; C = E[has]
print(f"  clean rows {len(E)}; country rows read {len(C)} ({C['mode'].value_counts().to_dict()}); not read {int((~has).sum())} ({E[~has]['mode'].value_counts().to_dict()}, {E[~has].survey.nunique()} surveys)")
def ctab(lab, S):
    M = {"Opus 5.5": "opus", "Sonnet 5.5": "sonnet", "Gemini Flash thinking": "gemini_flash_thinking", "options": "n_opts", "question length": "question_chars", "prompt length": "prompt_chars"}
    Tb = pd.DataFrame({k: {c: spearman(S.dropna(subset=[c])[v], S.dropna(subset=[c])[c]) for c in COUNTRIES} for k, v in M.items()})
    up = pd.Series({c: spearman(S.dropna(subset=[c]).real_dk_pct, S.dropna(subset=[c])[c]) for c in COUNTRIES})
    out = f"  {lab:34s} n {len(S):3d} |"
    for k in ("Opus 5.5", "Sonnet 5.5", "Gemini Flash thinking"): out += f" {k} median {Tb[k].median():.3f} (lowest {Tb[k].min():.3f}), beats both simple baselines in {int(((Tb[k] > Tb['options']) & (Tb[k] > Tb['question length'])).sum())} of 27 |"
    print(out + f" question length median {Tb['question length'].median():.3f}, prompt length median {Tb['prompt length'].median():.3f} | real EU-wide share against each country: median {up.median():.3f}, lowest {up.min():.3f}, highest {up.max():.3f}")
ctab("face-to-face, all rows", C[C["mode"] == "face-to-face"]); ctab("new questions, all rows", C[(C["mode"] == "face-to-face") & (C.new_question == 1)]); ctab("web", C[C["mode"] == "web"])
ctab("face-to-face, each question once", C[(C["mode"] == "face-to-face") & C.unique])
