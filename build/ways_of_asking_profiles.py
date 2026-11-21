"""Recreate the simulated-respondent profiles and the real answers of the people behind them (section 6, Appendix D.7)
from the respondent-level files in raw/. The output is respondent-level, so it is written to raw/derived/ (git-ignored),
never to data/:
  raw/derived/ways_of_asking_profiles.csv      survey, profile, resp_id, mode, text   (50 per survey)
  raw/derived/ways_of_asking_real_answers.csv  survey, profile, qid, dk  (1 = "don't know", 0 = an answer, empty = not asked)

Sampling (seeds and order as in the original build):
- General Social Survey 2024: respondents complete on ten background fields; 25 drawn among interviewer-run cases (mode 1 or 2)
  and 25 among web cases (mode 4), seed 20261115; the order of the file is kept inside each group.
- General Social Survey 2022: 50 drawn among interviewer-run cases (mode 1 or 2) complete on the same ten fields, then
- World Values Survey 7: 50 drawn among face-to-face cases (Q_MODE 1 or 2) complete on the profile fields,
  both from one generator with seed 20261117 (2022 first). Interest in politics (Q199) is a target question and not in the profile.
Usage: python build/ways_of_asking_profiles.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis")); from common import questions, raw_file, RAW
OUT = RAW / "derived"; OUT.mkdir(exist_ok=True)
Q = questions(); var = lambda s: list(Q[Q.set == s].qid.str.split(":").str[1])
isnum = lambda t: isinstance(t, (int, float, np.number)) and not pd.isna(t)
F = ["age", "sex", "race", "hispanic", "degree", "wrkstat", "marital", "region", "attend", "partyid"]
PROF, REAL = [], []

def gss_read(name, cols):
    """Codes (Stata missing values kept as objects, '.d' = don't know) and, row-aligned, the value labels of the background fields."""
    codes = pd.read_stata(raw_file(name), columns=cols, convert_missing=True, convert_categoricals=False)
    with pd.io.stata.StataReader(raw_file(name)) as r: r.read(nrows=1); labels, lbl = r.value_labels(), dict(zip(r._varlist, r._lbllist))
    vl = {f: labels.get(lbl[f], {}) for f in F}
    return codes, vl

def gss_text(r, vl):
    lab = lambda f, v: "independent" if (f == "partyid" and int(v) == 3) else (str(vl[f][int(v)]) if int(v) in vl[f] else str(int(v)))
    age = "89 or older" if int(r.age) >= 89 else str(int(r.age))
    return (f"About you: you are {age} years old, {lab('sex', r.sex)}. Race: {lab('race', r.race)}. Hispanic origin: {lab('hispanic', r.hispanic)}. Highest degree: {lab('degree', r.degree)}. "
            f"Work status: {lab('wrkstat', r.wrkstat)}. Marital status: {lab('marital', r.marital)}. Region of the United States: {lab('region', r.region)}. "
            f"Attends religious services: {lab('attend', r.attend)}. Party: {lab('partyid', r.partyid)}.")

gss_dk = lambda t: 1 if str(t) == ".d" else 0 if isnum(t) else np.nan

# ---------- General Social Survey 2024
v24 = var("gss2024_interview"); raw, vl = gss_read("GSS2024.dta", ["id", "mode"] + F + v24)
ok = np.ones(len(raw), bool)
for f in F: ok &= raw[f].apply(isnum).to_numpy()
pool_i, pool_w = raw[ok & raw["mode"].isin([1, 2])], raw[ok & (raw["mode"] == 4)]; rng = np.random.default_rng(20261115)
S = pd.concat([pool_i.iloc[np.sort(rng.choice(len(pool_i), 25, replace=False))], pool_w.iloc[np.sort(rng.choice(len(pool_w), 25, replace=False))]]).reset_index(drop=True)
for i, r in S.iterrows():
    PROF.append(dict(survey="gss2024", profile=i, resp_id=int(r.id), mode=int(r["mode"]), text=gss_text(r, vl)))
    REAL += [dict(survey="gss2024", profile=i, qid="gss24_int:" + v, dk=gss_dk(r[v])) for v in v24]

# ---------- General Social Survey 2022
rng = np.random.default_rng(20261117); v22 = var("gss2022"); raw, vl = gss_read("GSS2022.dta", ["id", "mode"] + F + v22)
ok = raw["mode"].isin([1, 2]).to_numpy().copy()
for f in F: ok &= raw[f].apply(isnum).to_numpy()
S = raw[ok].iloc[rng.choice(int(ok.sum()), 50, replace=False)].reset_index(drop=True)
for i, r in S.iterrows():
    PROF.append(dict(survey="gss2022", profile=i, resp_id=int(r.id), mode=int(r["mode"]), text=gss_text(r, vl)))
    REAL += [dict(survey="gss2022", profile=i, qid="gss2022:" + v, dk=gss_dk(r[v])) for v in v22]

# ---------- World Values Survey 7 (labels from the WVS 7 codebook; country names as ISO 3166 short names)
C = {"Q260": {1: "Male", 2: "Female"},
     "Q275": {0: "Early childhood education (ISCED 0) / no education", 1: "Primary education (ISCED 1)", 2: "Lower secondary education (ISCED 2)", 3: "Upper secondary education (ISCED 3)",
              4: "Post-secondary non-tertiary education (ISCED 4)", 5: "Short-cycle tertiary education (ISCED 5)", 6: "Bachelor or equivalent (ISCED 6)", 7: "Master or equivalent (ISCED 7)"},
     "Q273": {1: "Married", 2: "Living together as married", 3: "Divorced", 4: "Separated", 5: "Widowed", 6: "Single"},
     "Q279": {1: "Full time (30 hours a week or more)", 2: "Part time (less than 30 hours a week)", 3: "Self employed", 4: "Retired/pensioned", 5: "Homemaker not otherwise employed", 6: "Student", 7: "Unemployed", 8: "Other"},
     "Q287": {1: "Upper class", 2: "Upper middle class", 3: "Lower middle class", 4: "Working class", 5: "Lower class"},
     "H_URBRURAL": {1: "Urban", 2: "Rural"}}
REL = {0: "none", 1: "Roman Catholic", 2: "Protestant", 3: "Orthodox", 4: "Jewish", 5: "Muslim", 6: "Hindu", 7: "Buddhist", 8: "other Christian", 9: "other"}
COUNTRY = {"AND": "Andorra", "ARG": "Argentina", "ARM": "Armenia", "AUS": "Australia", "BGD": "Bangladesh", "BOL": "Bolivia, Plurinational State of", "BRA": "Brazil", "CAN": "Canada", "CHL": "Chile",
           "CHN": "China", "COL": "Colombia", "CYP": "Cyprus", "CZE": "Czechia", "DEU": "Germany", "ECU": "Ecuador", "EGY": "Egypt", "ETH": "Ethiopia", "GBR": "United Kingdom", "GRC": "Greece",
           "GTM": "Guatemala", "HKG": "Hong Kong", "IDN": "Indonesia", "IND": "India", "IRN": "Iran, Islamic Republic of", "IRQ": "Iraq", "JOR": "Jordan", "JPN": "Japan", "KAZ": "Kazakhstan",
           "KEN": "Kenya", "KGZ": "Kyrgyzstan", "KOR": "Korea, Republic of", "LBN": "Lebanon", "LBY": "Libya", "MAC": "Macao", "MAR": "Morocco", "MDV": "Maldives", "MEX": "Mexico", "MMR": "Myanmar",
           "MNG": "Mongolia", "MYS": "Malaysia", "NGA": "Nigeria", "NIC": "Nicaragua", "NIR": "NIR", "NLD": "Netherlands", "NZL": "New Zealand", "PAK": "Pakistan", "PER": "Peru", "PHL": "Philippines",
           "PRI": "Puerto Rico", "ROU": "Romania", "RUS": "Russian Federation", "SGP": "Singapore", "SRB": "Serbia", "SVK": "Slovakia", "THA": "Thailand", "TJK": "Tajikistan", "TUN": "Tunisia",
           "TUR": "Türkiye", "TWN": "Taiwan, Province of China", "UKR": "Ukraine", "URY": "Uruguay", "USA": "United States", "UZB": "Uzbekistan", "VEN": "Venezuela, Bolivarian Republic of",
           "VNM": "Viet Nam", "ZWE": "Zimbabwe"}
vw = var("wvs7"); demo = ["B_COUNTRY_ALPHA", "Q_MODE", "Q260", "Q262", "Q275", "Q273", "Q279", "Q287", "Q288", "Q289", "H_URBRURAL"]
wv = pd.read_csv(raw_file("WVS_Cross-National_Wave_7_csv_v6_0.csv"), usecols=demo + vw, low_memory=False); f2f = wv[wv["Q_MODE"].isin([1, 2])]
ok = (f2f.Q260.isin(C["Q260"])) & (f2f.Q262 > 0) & (f2f.Q275.isin(C["Q275"])) & (f2f.Q273.isin(C["Q273"])) & (f2f.Q279.isin(C["Q279"])) & (f2f.Q287.isin(C["Q287"])) & (f2f.Q288 > 0) & (f2f.Q289.isin(REL)) & (f2f.H_URBRURAL.isin(C["H_URBRURAL"]))
S = f2f[ok].iloc[rng.choice(int(ok.sum()), 50, replace=False)].reset_index(drop=True)
for i, r in S.iterrows():
    PROF.append(dict(survey="wvs7", profile=i, resp_id=i, mode=int(r.Q_MODE), text=(
        f"About you: you live in {COUNTRY.get(str(r.B_COUNTRY_ALPHA), str(r.B_COUNTRY_ALPHA))}. You are {int(r.Q262)} years old, {C['Q260'][int(r.Q260)].lower()}. Education: {C['Q275'][int(r.Q275)]}. "
        f"Marital status: {C['Q273'][int(r.Q273)]}. Employment: {C['Q279'][int(r.Q279)]}. Social class (your own view): {C['Q287'][int(r.Q287)]}. "
        f"Household income step (1 = lowest tenth, 10 = highest tenth): {int(r.Q288)}. Religion: {REL[int(r.Q289)]}. Residence: {C['H_URBRURAL'][int(r.H_URBRURAL)].lower()}.")))
    REAL += [dict(survey="wvs7", profile=i, qid="wvs7:" + v, dk=(1 if r[v] == -1 else 0 if (isnum(r[v]) and r[v] > 0) else np.nan)) for v in vw]

P, R = pd.DataFrame(PROF), pd.DataFrame(REAL)
P.to_csv(OUT / "ways_of_asking_profiles.csv", index=False); R.to_csv(OUT / "ways_of_asking_real_answers.csv", index=False)
print(f"profiles {len(P)} ({P.survey.value_counts().to_dict()}), distinct texts {P.text.nunique()} | real answers {len(R)} rows, {int(R.dk.notna().sum())} answered, {int(R.dk.sum())} 'don't know'")
print("written to", OUT)
