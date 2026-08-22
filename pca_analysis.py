from pathlib import Path
from itertools import product
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report

FILE=Path("segments/segment_NA.csv")
PAIR_FILE=Path("difficult_cell_types_nearest_neighbor_genes.csv")
OUT=Path("ependymal_specialist_results"); OUT.mkdir(parents=True,exist_ok=True)

TARGET,VOL,X,Y="MERFISH_cell_type_annotation","volume","center_x","center_y"
TEST,REPEATS,SEED=.20,10,42
ASTRO_T,OLIGO_T=.70,.30
M1,M2=.35,.25
EP_MARGINS=[.05,.10,.15,.20,.25,.30]

ASTRO=["astrocyte_1","astrocyte_2"]
OLIGO=["oligodendrocyte_1","oligodendrocyte_2","oligodendrocyte_precursor_cell",
       "oligodendrocyte_progenitor_1","oligodendrocyte_progenitor_2"]

PAIR1=tuple(sorted(["oligodendrocyte_1","oligodendrocyte_progenitor_2"]))
PAIR2=tuple(sorted(["oligodendrocyte_precursor_cell","oligodendrocyte_progenitor_1"]))

# ---------- DATA ----------

df=pd.read_csv(FILE); df=df[df[TARGET].notna()].reset_index(drop=True)
meta=["Unnamed: 0","Datasets",TARGET,VOL,X,Y,"Region","Excitatory_vs_Inhibitory",
      "Segment","Gender","Mouse_ID","AP_position","Section_ID"]

genes=[c for c in df if c not in meta]
for g in genes: df[g]=pd.to_numeric(df[g],errors="coerce").fillna(0)
genes=[g for g in genes if df[g].nunique()>1]
G=np.log1p(df[genes].astype(float))
types=sorted(df[TARGET].astype(str).unique())

for c in [VOL,X,Y]: df[c]=pd.to_numeric(df[c],errors="coerce")

pg=pd.read_csv(PAIR_FILE); pg=pg[(pg.FDR<.05)&pg.gene.isin(genes)]
pg["pair"]=pg.apply(lambda r:tuple(sorted([r.cell_type,r.nearest_cell_type])),axis=1)
PAIR_GENES={p:pg[pg.pair==p].sort_values("FDR").gene.drop_duplicates().tolist() for p in [PAIR1,PAIR2]}

# ---------- MAIN MODEL ----------

def forest(s): return ExtraTreesClassifier(n_estimators=1000,class_weight="balanced",max_features="sqrt",random_state=s,n_jobs=-1)
def calibrated(s): return CalibratedClassifierCV(estimator=forest(s),method="sigmoid",cv=3,n_jobs=-1)

def feat(i,med):
    z=G.loc[i].copy(); z[VOL]=df.loc[i,VOL].fillna(med); return z

# ---------- FREQUENCY ----------

def counts(f,n):
    x=f*n; c=np.floor(x).astype(int)
    for i in np.argsort(-(x-c))[:n-c.sum()]: c[i]+=1
    return c

def assign(P,l,c):
    slots=np.repeat(np.arange(len(l)),c)
    r,s=linear_sum_assignment(-np.log(np.clip(P[:,slots],1e-12,1)))
    out=np.empty(len(P),object); out[r]=np.array(l)[slots[s]]
    return out

def correct(out,P,classes,y,members,pos):
    if not len(pos): return
    cols=[np.where(classes==m)[0][0] for m in members]
    f=y[y.isin(members)].value_counts(normalize=True)
    out[pos]=assign(P[pos][:,cols],members,counts(np.array([f.get(m,0) for m in members]),len(pos)))

def frequency(P,classes,base,y):
    out=base.copy(); conf=P.max(1)
    correct(out,P,classes,y,ASTRO,np.where(np.isin(base,ASTRO)&(conf<ASTRO_T))[0])
    correct(out,P,classes,y,OLIGO,np.where(np.isin(base,OLIGO)&(conf<OLIGO_T))[0])
    return out

# ---------- OLIGO SPECIALISTS ----------

def train_oligo(tr):
    sp={}
    for pair in [PAIR1,PAIR2]:
        gs=PAIR_GENES.get(pair,[]); idx=tr[df.loc[tr,TARGET].isin(pair).values]
        if gs and df.loc[idx,TARGET].nunique()==2:
            sc=StandardScaler().fit(G.loc[idx,gs])
            m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(sc.transform(G.loc[idx,gs]),df.loc[idx,TARGET].astype(str))
            sp[pair]=(m,sc,gs)
    return sp

def oligo_specialize(P,classes,pred,te,sp):
    out=pred.copy(); top=np.argsort(P,axis=1)[:,-2:][:,::-1]; margins={PAIR1:M1,PAIR2:M2}
    for i,(a,b) in enumerate(top):
        pair=tuple(sorted([classes[a],classes[b]]))
        if pair not in sp or P[i,a]-P[i,b]>margins[pair] or out[i] not in pair: continue
        m,sc,gs=sp[pair]; out[i]=m.predict(sc.transform(G.loc[[te[i]],gs]))[0]
    return out

# ---------- PCA / SPATIAL EPENDYMAL SPECIALIST ----------

def make_ep_features(tr,te):
    med=df.loc[tr,[X,Y,VOL]].median()
    Str=df.loc[tr,[X,Y,VOL]].fillna(med).to_numpy()
    Ste=df.loc[te,[X,Y,VOL]].fillna(med).to_numpy()

    gs=StandardScaler().fit(G.loc[tr])
    Ztr,Zte=gs.transform(G.loc[tr]),gs.transform(G.loc[te])

    pca=PCA(n_components=min(20,len(genes),len(tr)-1),random_state=SEED)
    Ptr,Pte=pca.fit_transform(Ztr),pca.transform(Zte)

    return {
        "coords":(Str,Ste),
        "pca20":(Ptr,Pte),
        "pca20_coords":(np.hstack([Ptr,Str]),np.hstack([Pte,Ste]))
    }

def train_ep(Xtr,ytr):
    sc=StandardScaler().fit(Xtr)
    y=(ytr=="ependymal").astype(int)
    m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(sc.transform(Xtr),y)
    return m,sc

def ep_specialize(P,classes,pred,Xte,m,sc,margin):
    out=pred.copy(); top=np.argsort(P,axis=1)[:,-2:][:,::-1]
    ep=np.where(classes=="ependymal")[0][0]

    for i,(a,b) in enumerate(top):
        if ep not in (a,b): continue
        other=b if a==ep else a
        gap=abs(P[i,ep]-P[i,other])

        if gap>margin or out[i] not in [classes[ep],classes[other]]: continue

        is_ep=m.predict(sc.transform(Xte[i:i+1]))[0]
        out[i]="ependymal" if is_ep else classes[other]

    return out

def score(y,p): return accuracy_score(y,p),balanced_accuracy_score(y,p),f1_score(y,p,average="macro",zero_division=0)

# ---------- VALIDATION ----------

results,preds=[] ,[]

for r in range(REPEATS):
    print(f"Repeat {r+1}/{REPEATS}")

    tr,te=train_test_split(df.index.to_numpy(),test_size=TEST,random_state=SEED+r,stratify=None)
    ytr,yte=df.loc[tr,TARGET].astype(str),df.loc[te,TARGET].astype(str)
    med=df.loc[tr,VOL].median()

    m=calibrated(SEED+r).fit(feat(tr,med),ytr)
    P,classes=m.predict_proba(feat(te,med)),m.classes_
    raw=classes[P.argmax(1)]

    current=frequency(P,classes,raw,ytr)
    current=oligo_specialize(P,classes,current,te,train_oligo(tr))

    methods={"current_best":current}
    ep_sets=make_ep_features(tr,te)

    for feature,(Xtr,Xte) in ep_sets.items():
        em,sc=train_ep(Xtr,ytr)

        for margin in EP_MARGINS:
            methods[f"ep_{feature}_M{margin:.2f}"]=ep_specialize(P,classes,current,Xte,em,sc,margin)

    for name,p in methods.items():
        a,b,f=score(yte,p)
        results.append([r+1,name,a,b,f])
        preds += [[r+1,i,name,t,q] for i,t,q in zip(te,yte,p)]

# ---------- RESULTS ----------

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

pred=pd.DataFrame(preds,columns=["repeat","cell_index","model","true","predicted"])
pred.to_csv(OUT/"all_predictions.csv",index=False)

best=summary.drop(index="current_best",errors="ignore").index[0]

rows=[]
for name in ["current_best",best]:
    d=pred[pred.model==name]
    rep=classification_report(d.true,d.predicted,labels=types,output_dict=True,zero_division=0)

    rows += [[name,c,rep[c]["precision"],rep[c]["recall"],rep[c]["f1-score"],rep[c]["support"]]
             for c in types]

perf=pd.DataFrame(rows,columns=["model","cell_type","precision","recall","f1","support"])
perf.to_csv(OUT/"per_cell_type_performance.csv",index=False)

f1=perf.pivot(index="cell_type",columns="model",values="f1")
f1["change_vs_current"]=f1[best]-f1["current_best"]
f1.to_csv(OUT/"per_cell_type_f1.csv")

print("\n==============================")
print("TOP MODELS")
print("==============================")
print(summary.head(12).round(4))

print(f"\nBEST EPENDYMAL MODEL: {best}")

print("\nCURRENT BEST:")
print(summary.loc["current_best"].round(4))

print("\nEPENDYMAL F1")
print(f1.loc["ependymal"].round(4))

print("\nALL CELL-TYPE CHANGES")
print(f1.sort_values("change_vs_current",ascending=False).round(3))

print(f"\nSaved to: {OUT.resolve()}")