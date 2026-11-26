"""Section 5.3 (Table 4) and Appendix D.5 (Tables D6, D7 and the text around them): is the forecast memorization of
published results?  Reads data/ only (and the two GSS data files in raw/ for the GSS 2024 dates and the overlap with 2022).

Parts:
  A. Eurobarometer cleaning: rows with table numbers inside the question are dropped; each question is counted once.
  B. Dating: a question is dated by its earliest fieldwork start and earliest publication among all clean rows with the same wording.
  C. Older models (Table D6, Table 4 row 1) and paired differences.
  D. Results published after the June 2026 cutoff (Table D7, Table 4 row 2) and the further checks on the 53 questions.
  E. The note on counts (104 rows, 100 questions).
  F. The earlier World Values Survey / GSS 2022 run with a sentence of reasoning (403 questions, 304 after cleaning).
  G. The 80 reworded questions.
  H. Llama on the General Social Survey 2024.
"""
import re, difflib
import numpy as np, pandas as pd
from common import DATA, questions, forecasts, spearman, interval, block_resamples, raw_file

def eb_clean_rows():
    """All clean Eurobarometer rows (a question asked in two surveys appears twice), with the dating columns."""
    R = pd.read_csv(DATA / "eb_rows.csv").sort_values("row_order")
    R["table_numbers"] = R.question.str.contains(r"^[A-Z_]{2,8} -?\d+ ") | R.question.str.contains(r"\d+ -?\d+ -?\d+")
    C = R[~R.table_numbers].copy()
    C["key"] = C.question.str.lower().str.replace(r"[^a-z]", "", regex=True) + "|" + C["mode"]
    C["counted_once"] = ~C.duplicated("key")
    S = pd.read_csv(DATA / "eb_surveys.csv").set_index("survey")
    C["fieldwork_start"] = C.survey.map(S.fieldwork_start); C["published"] = C.survey.map(S.first_published)
    C["fs_min"] = C.groupby("key").fieldwork_start.transform("min"); C["pub_min"] = C.groupby("key").published.transform("min")
    C["n_surveys"] = C.groupby("key").survey.transform("nunique"); C["q_chars"] = C.question.str.len()
    return R, C

def demeaned(s, by, cols):
    d = s[s.groupby(by)[by].transform("size") >= 2]
    return len(d), d[by].nunique(), {c: spearman(d[c] - d.groupby(by)[c].transform("mean"), d.real - d.groupby(by).real.transform("mean")) for c in cols}

M = {"opus45": ("Opus 4.5", "no thinking"), "sonnet45": ("Sonnet 4.5", "no thinking"), "opus": ("Opus 5.5", "thinking"), "sonnet": ("Sonnet 5.5", "no thinking")}

def new_questions():
    """The 100 new face-to-face questions with forecasts, dates and both length measures."""
    Q = questions(); N = Q[(Q.set == "eb_face_to_face") & (Q.new_question == 1)].copy().set_index("qid")
    for c, (m, st) in M.items(): N[c] = forecasts(m, st).reindex(N.index)
    f = pd.read_csv(DATA / "forecasts_memorization.csv"); N["opus45_think"] = f.groupby("qid").forecast_pct.mean().reindex(N.index)
    _, C = eb_clean_rows(); C = C.set_index("row_id")
    for c in ("fs_min", "pub_min", "n_surveys", "q_chars", "survey", "question"): N["row_" + c if c in ("survey", "q_chars") else c] = C.loc[N.index, c]
    N["real"] = N.real_dk_pct; N["qtext_chars"] = N.row_q_chars  # prompt_chars comes from data/questions.csv
    # the survey a question is dated by: the survey of its earliest publication
    first = C.sort_values("published").drop_duplicates("key").set_index("key").survey
    N["dated_by"] = C.loc[N.index, "key"].map(first).to_numpy()
    return N.reset_index()

def row(lab, s, cols):
    out = f"  {lab:50s} n {len(s):3d} blocks {s.block.nunique():2d} surveys (rows) {s.survey.nunique():2d} (dated by) {s.dated_by.nunique():2d} |"
    for c in cols:
        lo, hi = interval(s[c], s.real, s.block); out += f" {c} {spearman(s[c], s.real):.3f} [{lo:.2f}, {hi:.2f}]"
    print(out)

if __name__ == "__main__":
    print("=== A. Eurobarometer cleaning ===")
    R, C = eb_clean_rows(); Q = questions()
    print(f"  parsed table rows {len(R)}; with table numbers inside the question {int(R.table_numbers.sum())} {R[R.table_numbers].survey.value_counts().to_dict()}; clean rows {len(C)}; each question once {int(C.counted_once.sum())} {C[C.counted_once]['mode'].value_counts().to_dict()}")
    eq = set(Q[Q.programme == "Eurobarometer"].qid); print("  questions counted once equal the Eurobarometer qids of questions.csv:", set(C[C.counted_once].row_id) == eq)
    for c, (m, st) in M.items():
        u = C[C.counted_once].set_index("row_id"); col = {"opus": "opus55", "sonnet": "sonnet55"}.get(c, c)
        print(f"  {m}: forecasts in eb_rows equal forecasts_main on the 502: {np.allclose(u[col], forecasts(m, st).reindex(u.index), equal_nan=True)}")

    print("\n=== B. Dating ===")
    N = new_questions(); S = pd.read_csv(DATA / "eb_surveys.csv")
    print(f"  new questions {len(N)}; appearing in more than one survey: {int((N.n_surveys > 1).sum())}", N[N.n_surveys > 1][["qid", "row_survey", "dated_by", "fs_min", "pub_min"]].to_dict("records"))
    print(f"  earliest first publication among the 22 surveys: {S.first_published.min()} ({len(S)} surveys); among the 100: {N.pub_min.min()}; all 100 published after 31 Aug 2025: {bool((N.pub_min > '2025-08-31').all())}")
    e = N[N.fs_min <= "2025-08-31"]; print(f"  fielded by 31 Aug 2025: {len(e)} questions, fieldwork starts {sorted(e.fs_min.unique())} in surveys {e.dated_by.value_counts().to_dict()}")
    print(f"  fielded after 31 Aug 2025: {int((N.fs_min > '2025-08-31').sum())}")

    print("\n=== C. Older models: Table D6 and Table 4 row 1 ===")
    cols = ["opus45", "opus45_think", "sonnet45", "opus", "sonnet"]
    A87 = N[N.fs_min > "2025-08-31"]
    row("fielded after August 2025", A87, cols); row("all", N, cols)
    print(f"  question length (question text characters): 87 {spearman(A87.qtext_chars, A87.real):.3f}, all {spearman(N.qtext_chars, N.real):.3f} | prompt characters: 87 {spearman(A87.prompt_chars, A87.real):.3f}, all {spearman(N.prompt_chars, N.real):.3f}")
    n, g, v = demeaned(A87, "block", cols); print(f"  inside blocks, fielded after August 2025 ({n} questions, {g} blocks): " + ", ".join(f"{c} {x:.3f}" for c, x in v.items()))
    f = pd.read_csv(DATA / "forecasts_memorization.csv"); print(f"  Opus 4.5 thinking: calls {len(f)}, calls per question {f.groupby('qid').size().unique().tolist()}, unreadable {int(f.forecast_pct.isna().sum())}, calls with thinking {int((f.thinking_tokens > 0).sum())}, model {f.model_id_returned.unique().tolist()}")
    # paired differences on the 87: one random generator, 2,000 resamples per comparison, in this order (as first computed)
    res = list(block_resamples(A87.block.to_numpy(), n=8000, seed=0)); x = A87.reset_index(drop=True)
    for k, (a, b) in enumerate((("opus45_think", "opus"), ("opus45", "opus"), ("sonnet45", "sonnet"), ("opus45_think", "opus45"))):
        dd = [spearman(x[a].to_numpy()[i], x.real.to_numpy()[i]) - spearman(x[b].to_numpy()[i], x.real.to_numpy()[i]) for i in res[2000 * k: 2000 * (k + 1)]]
        print(f"  paired difference on the 87, {a} minus {b}: {spearman(x[a], x.real) - spearman(x[b], x.real):+.3f} [{np.nanpercentile(dd, 2.5):+.3f}, {np.nanpercentile(dd, 97.5):+.3f}]")

    print("\n=== D. Published after the June 2026 cutoff: Table D7 and Table 4 row 2 ===")
    late, early = N[N.pub_min > "2026-06-30"], N[N.pub_min <= "2026-06-30"]
    row("first published after 30 June 2026", late, ["opus", "sonnet", "opus45", "sonnet45", "n_options"]); row("first published by 30 June 2026", early, ["opus", "sonnet", "opus45", "sonnet45", "n_options"])
    for lab, s in (("after", late), ("by", early)): print(f"  {lab}: question length (text) {spearman(s.qtext_chars, s.real):.3f}, prompt characters {spearman(s.prompt_chars, s.real):.3f} | dated by {s.dated_by.value_counts().to_dict()} | rows sit in {s.row_survey.value_counts().to_dict()}")
    print(f"  fieldwork end of the surveys dating the 53: " + ", ".join(f"{sv} {S.set_index('survey').fieldwork_end[sv]}" for sv in sorted(late.dated_by.unique())))
    l2 = late[late.pub_min > "2026-07-01"]; print(f"  surveys published on 1 July 2026 among the five: {sorted(late[late.pub_min == '2026-07-01'].dated_by.unique())}; without it n {len(l2)} | opus {spearman(l2.opus, l2.real):.3f} sonnet {spearman(l2.sonnet, l2.real):.3f}")
    n, g, v = demeaned(late, "block", ["opus", "sonnet"]); print(f"  the 53 inside blocks ({n} questions, {g} blocks): " + ", ".join(f"{c} {x:.3f}" for c, x in v.items()))
    n, g, v = demeaned(late, "survey", ["opus", "sonnet"]); print(f"  the 53 inside surveys ({n} questions, {g} surveys): " + ", ".join(f"{c} {x:.3f}" for c, x in v.items()))
    big = late.dated_by.value_counts().idxmax(); w = late[late.dated_by != big]
    print(f"  largest survey among the 53: {big} with {int((late.dated_by == big).sum())}; without it n {len(w)} | " + ", ".join(f"{c} {spearman(w[c], w.real):.3f} [{interval(w[c], w.real, w.block)[0]:.2f}, {interval(w[c], w.real, w.block)[1]:.2f}]" for c in ("opus", "sonnet")))
    # wording of the 53 against every clean row of the surveys published by 30 June 2026
    norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()); pool = C[C.published <= "2026-06-30"]
    sim = []
    for _, r in late.iterrows():
        best = max(((difflib.SequenceMatcher(None, norm(r.question), norm(q2)).ratio(), j2, q2) for j2, q2 in zip(pool.row_id, pool.question)), key=lambda t: t[0]); sim.append(best)
    late = late.assign(sim=[b[0] for b in sim]); print(f"  wording: 53 questions against {len(pool)} clean rows published by 30 June 2026; character similarity of the closest earlier wording, top 6:")
    for (sv, j2, q2), (_, r) in sorted(zip(sim, late.iterrows()), key=lambda t: -t[0][0])[:6]: print(f"     {sv:.2f} | {r.qid}: {r.question[:80]}\n            ~ {j2}: {q2[:80]}")
    k = late[late.sim < 0.75]; print(f"  without the questions at similarity 0.75 or above ({len(late) - len(k)} dropped): n {len(k)} | opus {spearman(k.opus, k.real):.3f} sonnet {spearman(k.sonnet, k.real):.3f}")

    print("\n=== E. Note on counts ===")
    nr = C[(C["mode"] == "face-to-face") & (C.prior_wave_column == "new")]; print(f"  clean face-to-face rows without a change-since-last-wave column: {len(nr)}; counted once {int(nr.counted_once.sum())}")
    for _, r in nr[~nr.counted_once].iterrows():
        o = C[(C.key == r.key) & C.counted_once].iloc[0]; print(f"     {r.row_id} (new) is counted as {o.row_id} in {o.survey}, marked '{o.prior_wave_column}'; published {r.published} vs {o.published}")

    print("\n=== F. The earlier run with a sentence of reasoning (WVS 7 and GSS 2022) ===")
    I = pd.read_csv(DATA / "wvs_gss2022_403.csv"); print(f"  questions {len(I)} {I.dataset.value_counts().to_dict()}; kept {int(I.kept.sum())} {I[I.kept].dataset.value_counts().to_dict()}")
    print("  dropped by rule:", I[~I.kept].groupby(["dataset", "excluded_reason"]).size().to_dict())
    print("  kept questions equal the WVS 7 + GSS 2022 qids of questions.csv:", set(I[I.kept].qid if "qid" in I else I[I.kept].item_id) == set(Q[Q.set.isin(["wvs7", "gss2022"])].qid))
    rp = pd.read_json(DATA / "replies/wvs_gss2022_reasoning_run_sonnet55.jsonl", lines=True)
    pat_name = re.compile(r"World Values Survey|\bWVS\b|General Social Survey|\bGSS\b|\bNORC\b", re.I)
    pat_item = re.compile(r"\b(item|variable|question)\s+(V|Q)?\d+|\b[A-Z]{3,9}\d?\b\s+item|GSS (item|variable|mnemonic)|\((?:GSS|WVS)[^)]*\)", re.I)
    rp["text"] = rp.raw.str.len() > 12; rp["names_survey"] = rp.raw.apply(lambda t: bool(pat_name.search(t))); rp["names_item"] = rp.raw.apply(lambda t: bool(pat_item.search(t)))
    print(f"  Sonnet 5.5 replies {len(rp)}: with text besides the number {int(rp.text.sum())}; naming the survey {int(rp.names_survey.sum())} ({100 * rp.names_survey.mean():.0f}%); naming a specific item {int(rp.names_item.sum())}")
    k = rp[rp.qid.isin(I[I.kept].item_id)]; print(f"  on the 304 kept: naming the survey {int(k.names_survey.sum())} of {len(k)}")

    print("\n=== G. The 80 reworded questions ===")
    T = pd.read_csv(DATA / "paraphrase80.csv"); print(f"  {len(T)} questions {T.dataset.value_counts().to_dict()}; options count kept equal: {int((T.options_original.str.count('=') == T.options_reworded.str.count('=')).sum())} of {len(T)}")
    print("  " + " | ".join(f"{m} {w} {spearman(T[f'{m}55_{w}'], T.real_dk_pct):.3f}" for m in ("opus", "sonnet") for w in ("original", "reworded")))

    print("\n=== H. Llama on the General Social Survey 2024 ===")
    L = pd.read_csv(DATA / "forecasts_llama_gss2024.csv").merge(Q[["qid", "real_dk_pct", "block"]], on="qid")
    print(f"  55 questions by module: {L.module.value_counts().to_dict()}")
    d = L[L.module == "issp_digsoc_2024"]; print(f"  digital societies module: n {len(d)}, blocks {d.block.nunique()} | Llama 3.3 70B {spearman(d.llama33_70b, d.real_dk_pct):.3f} | Llama 4 Maverick {spearman(d.llama4_maverick, d.real_dk_pct):.3f} | calls per question {d.llama33_70b_calls.unique().tolist()}")
    with pd.read_stata(raw_file("GSS2022.dta"), iterator=True, convert_categoricals=False) as rd22: v22 = {v.lower() for v in rd22.read(1).columns}
    print(f"  of the 55 question variables, present in the 2022 data file ({len(v22)} variables): {sorted(set(L['var'].str.lower()) & v22)}")
    g24 = pd.read_stata(raw_file("GSS2024.dta"), columns=["year", "dateintv"], convert_categoricals=False)
    di = pd.to_numeric(g24.dateintv, errors="coerce"); print(f"  2024 file: {len(g24)} respondents, year {sorted(g24.year.dropna().unique().tolist())}, interview date (month*100 + day) from {int(di.min())} to {int(di.max())}, missing for {int(di.isna().sum())}")
