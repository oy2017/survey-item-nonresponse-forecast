"""Table 1 (example block from Standard Eurobarometer 105) and Table 2 (question counts per set), sections 1 and 3."""
from common import *
Q = questions()
print("Table 2: questions per set"); print(Q.groupby("set", sort=False).agg(questions=("qid", "size"), blocks=("block", "nunique"), surveys=("survey", "nunique")).to_string()); print("total", len(Q))
e = Q[(Q.set == "eb_face_to_face") & (Q.survey == "STD105") & Q.question_text.str.contains("How would you judge the current situation", regex=False)]
opus = forecasts("Opus 5.5", "thinking")
for b, g in e.groupby("block"):
    if len(g) != 7: continue
    f = opus.reindex(g.qid).to_numpy(); print(f"\nTable 1: block {b}, {len(g)} questions, Spearman {spearman(f, g.real_dk_pct):.2f}")
    for (_, r), x in sorted(zip(g.iterrows(), f), key=lambda t: -t[0][1].real_dk_pct):
        item = r.question_text[:90]
        print(f"  real {r.real_dk_pct:4.0f}%  forecast {x:5.1f}%  {r.qid}  {item}")
