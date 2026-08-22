from pathlib import Path
from itertools import product
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

FILE=Path("segments/segment_NA.csv")
PAIR_FILE=Path("difficult_cell_types_nearest_neighbor_genes.csv")
OUT=Path("calibrated_margin_tuning"); OUT.mkdir(parents=True,exist_ok=True)

TARGET,VOL,TEST,REPEATS,SEED="MERFISH_cell_type_annotation","volume",.20,10,42
ASTRO_T,OLIGO_T=.70,.30
MARGINS=[.20,.25,.30,.35,.40]

ASTRO=["astrocyte_1","astrocyte_2"]
OLIGO=["oligodendrocyte_1","oligodendrocyte_2","oligodendrocyte_precursor_cell",
       "oligodendrocyte_progenitor_1","oligodendrocyte_progenitor_2"]

PAIR1=tuple(sorted(["oligodendrocyte_1","oligodendrocyte_progenitor_2"]))
PAIR2=tuple(sorted(["oligodendrocyte_precursor_cell","oligodendrocyte_progenitor_1"]))

# ---------------- DATA ----------------

df=pd.read_csv(FILE); df=df[df[TARGET].notna()].reset_index(drop=True)
meta=["Unnamed: 0","Datasets",TARGET,VOL,"center_x","center_y","Region","Excitatory_vs_Inhibitory",
      "Segment","Gender","Mouse_ID","AP_position","Section_ID"]

genes=[c for c in df if c not in meta]
for g in genes: df[g]=pd.to_numeric(df[g],errors="coerce").fillna(0)
genes=[g for g in genes if df[g].nunique()>1]

G=np.log1p(df[genes].astype(float))
types=sorted(df[TARGET].astype(str).unique())

pg=pd.read_csv(PAIR_FILE); pg=pg[(pg.FDR<.05)&pg.gene.isin(genes)]
pg["pair"]=pg.apply(lambda r:tuple(sorted([r.cell_type,r.nearest_cell_type])),axis=1)
PAIR_GENES={p:pg[pg.pair==p].sort_values("FDR").gene.drop_duplicates().tolist() for p in [PAIR1,PAIR2]}

# ---------------- MODELS ----------------

def forest(s):
    return ExtraTreesClassifier(n_estimators=1000,class_weight="balanced",
        max_features="sqrt",random_state=s,n_jobs=-1)

def calibrated(s):
    return CalibratedClassifierCV(estimator=forest(s),method="sigmoid",cv=3,n_jobs=-1)

def feat(i,med):
    x=G.loc[i].copy()
    x["volume"]=pd.to_numeric(df.loc[i,VOL],errors="coerce").fillna(med)
    return x

# ---------------- FREQUENCY CORRECTION ----------------

def counts(f,n):
    x=f*n; c=np.floor(x).astype(int)
    for i in np.argsort(-(x-c))[:n-c.sum()]: c[i]+=1
    return c

def assign(P,labels,c):
    slots=np.repeat(np.arange(len(labels)),c)
    r,s=linear_sum_assignment(-np.log(np.clip(P[:,slots],1e-12,1)))
    out=np.empty(len(P),object); out[r]=np.array(labels)[slots[s]]
    return out

def correct(out,P,classes,y,members,pos):
    if not len(pos): return
    cols=[np.where(classes==m)[0][0] for m in members]
    f=y[y.isin(members)].value_counts(normalize=True)
    out[pos]=assign(P[pos][:,cols],members,
        counts(np.array([f.get(m,0) for m in members]),len(pos)))

def frequency(P,classes,base,y):
    out=base.copy(); conf=P.max(1)
    correct(out,P,classes,y,ASTRO,np.where(np.isin(base,ASTRO)&(conf<ASTRO_T))[0])
    correct(out,P,classes,y,OLIGO,np.where(np.isin(base,OLIGO)&(conf<OLIGO_T))[0])
    return out

# ---------------- SPECIALISTS ----------------

def train_specialists(tr):
    sp={}
    for pair in [PAIR1,PAIR2]:
        gs=PAIR_GENES.get(pair,[])
        idx=tr[df.loc[tr,TARGET].isin(pair).values]
        if gs and df.loc[idx,TARGET].nunique()==2:
            m=make_pipeline(StandardScaler(),
                LogisticRegression(class_weight="balanced",max_iter=3000))
            m.fit(G.loc[idx,gs],df.loc[idx,TARGET].astype(str))
            sp[pair]=(m,gs)
    return sp

def specialize(P,classes,pred,te,sp,m1,m2):
    out=pred.copy(); top=np.argsort(P,axis=1)[:,-2:][:,::-1]
    margins={PAIR1:m1,PAIR2:m2}

    for i,(a,b) in enumerate(top):
        pair=tuple(sorted([classes[a],classes[b]]))
        gap=P[i,a]-P[i,b]

        if pair not in sp or gap>margins[pair] or out[i] not in pair: continue

        m,gs=sp[pair]
        out[i]=m.predict(G.loc[[te[i]],gs])[0]

    return out

def score(y,p):
    return accuracy_score(y,p),balanced_accuracy_score(y,p),f1_score(y,p,average="macro",zero_division=0)

# ---------------- VALIDATION ----------------

results,preds=[],[]

for r in range(REPEATS):
    print(f"Repeat {r+1}/{REPEATS}")

    tr,te=train_test_split(df.index.to_numpy(),test_size=TEST,random_state=SEED+r,stratify=None)
    ytr,yte=df.loc[tr,TARGET].astype(str),df.loc[te,TARGET].astype(str)
    med=pd.to_numeric(df.loc[tr,VOL],errors="coerce").median()

    m=calibrated(SEED+r).fit(feat(tr,med),ytr)
    P,classes=m.predict_proba(feat(te,med)),m.classes_
    raw=classes[P.argmax(1)]

    freq=frequency(P,classes,raw,ytr)
    sp=train_specialists(tr)

    methods={"calibrated_raw":raw,"frequency_only":freq}

    for m1,m2 in product(MARGINS,MARGINS):
        methods[f"M1_{m1:.2f}_M2_{m2:.2f}"]=specialize(P,classes,freq,te,sp,m1,m2)

    for name,p in methods.items():
        a,b,f=score(yte,p)
        results.append([r+1,name,a,b,f])
        preds += [[r+1,i,name,t,q] for i,t,q in zip(te,yte,p)]

# ---------------- SUMMARY ----------------

res=pd.DataFrame(results,columns=["repeat","model","accuracy","balanced_accuracy","macro_f1"])

summary=res.groupby("model").agg(
    accuracy_mean=("accuracy","mean"),
    accuracy_std=("accuracy","std"),
    balanced_accuracy_mean=("balanced_accuracy","mean"),
    balanced_accuracy_std=("balanced_accuracy","std"),
    macro_f1_mean=("macro_f1","mean"),
    macro_f1_std=("macro_f1","std")
).sort_values("macro_f1_mean",ascending=False)

res.to_csv(OUT/"all_runs.csv",index=False)
summary.to_csv(OUT/"model_summary.csv")

best=summary.drop(index=["calibrated_raw","frequency_only"],errors="ignore").index[0]
best_m1=float(best.split("_")[1])
best_m2=float(best.split("_")[3])

print("\n==============================")
print("TOP 15")
print("==============================")
print(summary.head(15).round(4))

print(f"\nBEST MARGINS")
print(f"Oligo1 / Progenitor2: {best_m1:.2f}")
print(f"Precursor / Progenitor1: {best_m2:.2f}")

print("\nBEST PERFORMANCE")
print(summary.loc[best].round(4))

# ---------------- PER-CELL-TYPE ----------------

pred=pd.DataFrame(preds,columns=["repeat","cell_index","model","true","predicted"])
pred.to_csv(OUT/"all_predictions.csv",index=False)

rows=[]
for name in ["calibrated_raw","frequency_only",best]:
    d=pred[pred.model==name]
    rep=classification_report(d.true,d.predicted,labels=types,output_dict=True,zero_division=0)
    rows += [[name,c,rep[c]["precision"],rep[c]["recall"],rep[c]["f1-score"],rep[c]["support"]]
             for c in types]

perf=pd.DataFrame(rows,columns=["model","cell_type","precision","recall","f1","support"])
perf.to_csv(OUT/"per_cell_type_performance.csv",index=False)

f1=perf.pivot(index="cell_type",columns="model",values="f1")
f1["change_vs_calibrated_raw"]=f1[best]-f1["calibrated_raw"]
f1["change_vs_frequency_only"]=f1[best]-f1["frequency_only"]
f1.to_csv(OUT/"per_cell_type_f1.csv")

# ---------------- GRID CSV ----------------

grid=summary.drop(index=["calibrated_raw","frequency_only"],errors="ignore").reset_index()
grid["margin_1"]=grid.model.str.extract(r"M1_([\d.]+)").astype(float)
grid["margin_2"]=grid.model.str.extract(r"M2_([\d.]+)").astype(float)

for metric in ["accuracy_mean","balanced_accuracy_mean","macro_f1_mean"]:
    grid.pivot(index="margin_1",columns="margin_2",values=metric).to_csv(OUT/f"{metric}_grid.csv")

with open(OUT/"best_settings.txt","w") as f:
    f.write(
        f"Astro frequency threshold: {ASTRO_T:.2f}\n"
        f"Oligo frequency threshold: {OLIGO_T:.2f}\n"
        f"Peripheral correction: OFF\n"
        f"Oligo1/Progenitor2 margin: {best_m1:.2f}\n"
        f"Precursor/Progenitor1 margin: {best_m2:.2f}\n\n"
        f"{summary.loc[best].to_string()}"
    )

print("\nPER-CELL-TYPE EFFECT")
print(f1.sort_values("change_vs_frequency_only",ascending=False).round(3))

print(f"\nSaved to: {OUT.resolve()}")