"""Appendix D.5, the paragraph on Sonnet 5.5 naming Eurobarometer: when Sonnet 5.5 names the survey in its reply, is its
forecast closer to the real share, and more so where it could have seen the result?  Reads data/ only; no model calls.

Sonnet 5.5 was called three times per question.  On the 374 face-to-face Eurobarometer questions some of its replies have
a sentence before the number, and every such sentence names Eurobarometer.  On questions where only some of the three
calls name it, the calls that name it are compared with the calls that do not, on the same question.

Parts:
  A. Counts: replies with text, replies naming Eurobarometer, questions where some but not all calls name it.
  B. Same question: error and ordering of the calls that name it against the calls that do not.
  C. The same comparison split by whether Sonnet 5.5 could have seen the result (training cutoff June 2026).
"""
import numpy as np, pandas as pd
from common import DATA, questions, spearman, block_resamples

Q = questions()
f2f = Q[Q.set == "eb_face_to_face"].set_index("qid")
R = pd.read_json(DATA / "replies" / "main_forecast_replies.jsonl", lines=True)
R = R[(R.model == "Sonnet 5.5") & R.qid.isin(f2f.index)][["qid", "draw", "raw"]]
F = pd.read_csv(DATA / "forecasts_main.csv").query("model == 'Sonnet 5.5'")[["qid", "draw", "forecast_pct"]]
S = R.merge(F, on=["qid", "draw"])
S["named"] = S.raw.str.contains("eurobarometer", case=False)
S["has_text"] = S.raw.str.strip().str.len() > len("[100.0]")
S["real"] = S.qid.map(f2f.real_dk_pct)
S["error"] = (S.forecast_pct - S.real).abs()
S["signed"] = S.forecast_pct - S.real

def boot(fn, d, n=2000):
    """95% interval over 2,000 resamples of whole question blocks."""
    return np.nanpercentile([fn(d.iloc[i]) for i in block_resamples(d.block.values, n)], [2.5, 97.5])

print("=== A. Counts (face-to-face Eurobarometer, Sonnet 5.5, three calls per question) ===")
k = S.groupby("qid").named.sum()
print(f"replies {len(S)}; with a sentence before the number {S.has_text.sum()}; naming Eurobarometer {S.named.sum()};"
      f" with a sentence but not naming it {(S.has_text & ~S.named).sum()}")
print(f"questions: {len(k)}; no call names it {(k == 0).sum()}; all three name it {(k == 3).sum()}; one or two name it {k.between(1, 2).sum()}")
print(f"mean signed error (forecast minus real), all Sonnet 5.5 replies: {S.signed.mean():+.2f} points")

print("\n=== B. Same question: calls that name Eurobarometer against calls that do not ===")
mixed = S[S.qid.isin(k.index[k.between(1, 2)])]
W = pd.DataFrame({
    "error_named": mixed[mixed.named].groupby("qid").error.mean(), "error_not": mixed[~mixed.named].groupby("qid").error.mean(),
    "fc_named": mixed[mixed.named].groupby("qid").forecast_pct.mean(), "fc_not": mixed[~mixed.named].groupby("qid").forecast_pct.mean()})
W["diff"] = W.error_named - W.error_not
W["real"] = f2f.real_dk_pct; W["block"] = f2f.block
lo, hi = boot(lambda d: d["diff"].mean(), W)
print(f"{len(W)} questions in {W.block.nunique()} blocks")
print(f"error (points), named minus not named: {W['diff'].mean():+.2f} [{lo:+.2f}, {hi:+.2f}]; named call closer on {(W['diff'] < 0).sum()},"
      f" further on {(W['diff'] > 0).sum()}, equal on {(W['diff'] == 0).sum()}")
print(f"forecast (points), named minus not named: {(W.fc_named - W.fc_not).mean():+.2f}")
o = lambda d: spearman(d.fc_named, d.real) - spearman(d.fc_not, d.real)
lo, hi = boot(o, W)
print(f"Spearman correlation with the real share: named {spearman(W.fc_named, W.real):.3f}, not named {spearman(W.fc_not, W.real):.3f},"
      f" difference {o(W):+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n=== C. Split by whether Sonnet 5.5 could have seen the result (training cutoff June 2026) ===")
pub = f2f.survey.map(pd.read_csv(DATA / "eb_surveys.csv").set_index("survey").first_published)
f2f["group"] = np.where(f2f.new_question == 1, np.where(pub > "2026-06-30", "new, first published after June 2026", "new, first published by June 2026"),
                        "asked in earlier Eurobarometer waves")
W["group"] = f2f.group; S["group"] = S.qid.map(f2f.group)
for g, d in W.groupby("group"):
    lo, hi = boot(lambda x: x["diff"].mean(), d) if d.block.nunique() > 3 else (np.nan, np.nan)
    share = S[S.group == g].named.mean()
    print(f"{g:38s} replies naming it {share:.2f}; questions where some calls name it {len(d):3d}; error, named minus not named {d['diff'].mean():+.2f}"
          + ("" if np.isnan(lo) else f" [{lo:+.2f}, {hi:+.2f}]"))
unseen = lambda d: d.group == "new, first published after June 2026"
gap = lambda d: d[d.group == "asked in earlier Eurobarometer waves"]["diff"].mean() - d[unseen(d)]["diff"].mean()
lo, hi = boot(gap, W)
print(f"asked in earlier waves minus new and first published after June 2026: {gap(W):+.2f} [{lo:+.2f}, {hi:+.2f}]"
      " (negative would mean naming helps more where the result could have been seen)")
