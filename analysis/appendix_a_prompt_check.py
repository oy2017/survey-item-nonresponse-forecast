"""Appendix A: check that every prompt sent for the main forecast fits the templates printed in the paper."""
import re
from common import *
P = pd.read_csv(DATA / "prompts_main.csv").merge(questions()[["qid", "set"]], on="qid")
SYS_I = "You are a survey methodologist estimating item non-response in live household interviews. Reply with ONLY one percentage between 0.0 and 100.0 inside square brackets, such as [0.6] or [9.5]."
SYS_W = SYS_I.replace("live household interviews", "self-administered web surveys")
EU = "adults in the 27 member states of the European Union. Results are for the European Union as a whole; '(OUR COUNTRY)' or '(NATIONALITY)' in a question stands for the respondent's own country"
POP = {"wvs7": "a cross-national face-to-face household interview survey of adults across 60 countries",
       "gss2022": "a nationally representative in-person and telephone interview survey of adults in the United States",
       "gss2024_interview": "a nationally representative in-person and telephone interview survey of adults in the United States",
       "eb_face_to_face": "a face-to-face interview survey of " + EU, "eb_telephone": "a telephone interview survey of " + EU}
INT = re.compile(r"Population and mode: (?P<pop>.*)\. Interviewers read the question and substantive choices aloud; 'Don't know' is NOT read aloud, but recorded if volunteered by the respondent\.\n\nQuestion: (?P<q>.*)\nSubstantive options: (?P<o>.*)\n\nWhat percentage \(0\.0 to 100\.0\) of respondents asked this question volunteer 'Don't know'\? Output ONLY one number in brackets, e\.g\. \[2\.4\]\.", re.S)
WEB = re.compile(r"Population and mode: a self-administered web survey of " + re.escape(EU) + r"\. Respondents read the question on screen, and 'Don't know' is explicitly displayed in the list of answer options\.\n\nQuestion: (?P<q>.*)\nOn-screen options \(in order\): (?P<o>.*); Don't know\n\nWhat percentage \(0\.0 to 100\.0\) of respondents asked this question select 'Don't know'\? Output ONLY one number in brackets, e\.g\. \[2\.4\]\.", re.S)
bad = {}
for _, r in P.iterrows():
    if r.set == "eb_web": ok = r.system == SYS_W and WEB.fullmatch(r.user)
    elif r.set == "gss2024_web": ok = None  # template checked separately below
    else: m = INT.fullmatch(r.user); ok = r.system == SYS_I and m and m["pop"] == POP[r.set]
    if ok is False or (ok is not None and not ok): bad.setdefault(r.set, []).append(r.qid)
print("prompts per set:", P.groupby("set").size().to_dict()); print("not matching the Appendix A template:", {k: len(v) for k, v in bad.items()} or "none")
for k, v in bad.items(): print(k, v[:3]); print(P.set_index("qid").loc[v[0], "user"][:700]); print("---")
gw = P[P.set == "gss2024_web"].iloc[0]; print("GSS 2024 web prompt (example):\nSYSTEM:", gw.system, "\nUSER:", gw.user)
