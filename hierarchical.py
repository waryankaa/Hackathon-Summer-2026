from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix

FILE,PAIR_FILE=Path("segments/segment_NA.csv"),Path("difficult_cell_types_nearest_neighbor_genes.csv")
OUT=Path("oligo_specialist_results"); OUT.mkdir(parents=True,exist_ok=True)
TARGET,VOL,TEST,REPEATS,SEED="MERFISH_cell_type_annotation","volume",.20,10,42
ASTRO_T,OLIGO_T=.50,.60
ASTRO=["astrocyte_1","astrocyte_2"]
OLIGO=["oligodendrocyte_1","oligodendrocyte_2","oligodendrocyte_precursor_cell","oligodendrocyte_progenitor_1","oligodendrocyte_progenitor_2"]
PERIPHERAL=["Schwann_cell","peripheral_glia"]
SPECIALISTS={
    tuple(sorted(["oligodendrocyte_1","oligodendrocyte_progenitor_2"])):.30,
    tuple(sorted(["oligodendrocyte_precursor_cell","oligodendrocyte_progenitor_1"])):.20
}

df=pd.read_csv(FILE); df=df[df[TARGET].notna()].reset_index(drop=True)
meta=["Unnamed: 0","Datasets",TARGET,VOL,"center_x","center_y","Region","Excitatory_vs_Inhibitory","Segment","Gender","Mouse_ID","AP_position","Section_ID"]
genes=[c for c in df if c not in meta]
for g in genes: df[g]=pd.to_numeric(df[g],errors="coerce").fillna(0)
genes=[g for g in genes if df[g].nunique()>1]; G=np.log1p(df[genes].astype(float)); types=sorted(df[TARGET].astype(str).unique())

pg=pd.read_csv(PAIR_FILE); pg=pg[(pg.FDR<.05)&pg.gene.isin(genes)]
pg["pair"]=pg.apply(lambda r:tuple(sorted([r.cell_type,r.nearest_cell_type])),axis=1)
PAIR_GENES={p:pg[pg.pair==p].sort_values("FDR").gene.drop_duplicates().tolist() for p in SPECIALISTS}
pd.DataFrame([(a,b,g) for (a,b),gs in PAIR_GENES.items() for g in gs],columns=["type1","type2","gene"]).to_csv(OUT/"specialist_genes.csv",index=False)
print("SPECIALIST GENES"); [print(p,gs) for p,gs in PAIR_GENES.items()]

def model(s): return ExtraTreesClassifier(n_estimators=1000,class_weight="balanced",max_features="sqrt",random_state=s,n_jobs=-1)
def feat(i,med):
    x=G.loc[i].copy(); x["volume"]=pd.to_numeric(df.loc[i,VOL],errors="coerce").fillna(med); return x
def counts(f,n):
    x=f*n; c=np.floor(x).astype(int)
    for i in np.argsort(-(x-c))[:n-c.sum()]: c[i]+=1
    return c
def assign(P,l,c):
    slots=np.repeat(np.arange(len(l)),c); r,s=linear_sum_assignment(-np.log(np.clip(P[:,slots],1e-12,1)))
    out=np.empty(len(P),object); out[r]=np.array(l)[slots[s]]; return out
def correct(out,P,classes,y,members,pos):
    if not len(pos): return
    cols=[np.where(classes==m)[0][0] for m in members]; f=y[y.isin(members)].value_counts(normalize=True)
    out[pos]=assign(P[pos][:,cols],members,counts(np.array([f.get(m,0) for m in members]),len(pos)))
def hybrid(P,classes,base,y):
    out=base.copy(); conf=P.max(1)
    correct(out,P,classes,y,ASTRO,np.where(np.isin(base,ASTRO)&(conf<ASTRO_T))[0])
    correct(out,P,classes,y,OLIGO,np.where(np.isin(base,OLIGO)&(conf<OLIGO_T))[0])
    correct(out,P,classes,y,PERIPHERAL,np.where(np.isin(base,PERIPHERAL))[0])
    return out
def train_specialists(tr):
    out={}
    for pair,margin in SPECIALISTS.items():
        gs=PAIR_GENES.get(pair,[]); idx=tr[df.loc[tr,TARGET].isin(pair).values]
        if gs and df.loc[idx,TARGET].nunique()==2:
            m=make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=3000))
            m.fit(G.loc[idx,gs],df.loc[idx,TARGET].astype(str)); out[pair]=(m,gs,margin)
    return out
def specialize(P,classes,hyb,te,sp,log,r):
    out=hyb.copy(); top=np.argsort(P,axis=1)[:,-2:][:,::-1]
    for i,(a,b) in enumerate(top):
        pair=tuple(sorted([classes[a],classes[b]])); gap=P[i,a]-P[i,b]
        if pair not in sp or gap>sp[pair][2] or out[i] not in pair: continue
        m,gs,margin=sp[pair]; new=m.predict(G.loc[[te[i]],gs])[0]
        log.append([r+1,te[i],*pair,margin,gap,out[i],new,df.loc[te[i],TARGET]]); out[i]=new
    return out
def score(y,p): return accuracy_score(y,p),balanced_accuracy_score(y,p),f1_score(y,p,average="macro",zero_division=0)

results,preds,changes=[],[],[]
for r in range(REPEATS):
    tr,te=train_test_split(df.index.to_numpy(),test_size=TEST,random_state=SEED+r,stratify=None)
    ytr,yte=df.loc[tr,TARGET].astype(str),df.loc[te,TARGET].astype(str); med=pd.to_numeric(df.loc[tr,VOL],errors="coerce").median()
    m=model(SEED+r).fit(feat(tr,med),ytr); P,classes=m.predict_proba(feat(te,med)),m.classes_; base=classes[P.argmax(1)]
    hyb=hybrid(P,classes,base,ytr); final=specialize(P,classes,hyb,te,train_specialists(tr),changes,r)
    for name,p in {"baseline":base,"targeted_hybrid":hyb,"targeted_hybrid_oligo_specialists":final}.items():
        a,b,f=score(yte,p); results.append([r+1,name,a,b,f]); preds += [[r+1,i,name,t,q] for i,t,q in zip(te,yte,p)]

res=pd.DataFrame(results,columns=["repeat","model","accuracy","balanced_accuracy","macro_f1"])
summary=res.groupby("model").agg(accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
    balanced_accuracy_mean=("balanced_accuracy","mean"),balanced_accuracy_std=("balanced_accuracy","std"),
    macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std")).sort_values("macro_f1_mean",ascending=False)
res.to_csv(OUT/"model_all_runs.csv",index=False); summary.to_csv(OUT/"model_summary.csv"); print("\nMODEL COMPARISON\n",summary.round(4))

pred=pd.DataFrame(preds,columns=["repeat","cell_index","model","true","predicted"]); pred.to_csv(OUT/"all_predictions.csv",index=False)
chg=pd.DataFrame(changes,columns=["repeat","cell_index","pair1","pair2","margin","probability_gap","before","after","true"]); chg.to_csv(OUT/"specialist_changes.csv",index=False)

rows=[]
for name,d in pred.groupby("model"):
    rep=classification_report(d.true,d.predicted,labels=types,output_dict=True,zero_division=0)
    rows += [[name,c,rep[c]["precision"],rep[c]["recall"],rep[c]["f1-score"],rep[c]["support"]] for c in types]
perf=pd.DataFrame(rows,columns=["model","cell_type","precision","recall","f1","support"]); perf.to_csv(OUT/"per_cell_type_performance.csv",index=False)
f1=perf.pivot(index="cell_type",columns="model",values="f1")
f1["specialist_change"]=f1["targeted_hybrid_oligo_specialists"]-f1["targeted_hybrid"]; f1.to_csv(OUT/"per_cell_type_f1.csv")

if len(chg):
    chg["before_correct"]=chg.before==chg.true; chg["after_correct"]=chg.after==chg.true
    effect=chg.groupby(["pair1","pair2"]).agg(uses=("cell_index","size"),before_accuracy=("before_correct","mean"),after_accuracy=("after_correct","mean"))
    effect["change"]=effect.after_accuracy-effect.before_accuracy; effect.to_csv(OUT/"specialist_effect.csv"); print("\nSPECIALIST EFFECT\n",effect.round(3))

best=summary.index[0]; d=pred[pred.model==best]; cm=confusion_matrix(d.true,d.predicted,labels=types,normalize="true")
pd.DataFrame(cm,index=types,columns=types).to_csv(OUT/"best_confusion_matrix.csv")
fig,ax=plt.subplots(figsize=(14,12)); im=ax.imshow(cm,vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(len(types)),labels=types,rotation=90,fontsize=8); ax.set_yticks(range(len(types)),labels=types,fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Best model: {best}"); fig.colorbar(im,ax=ax)
plt.tight_layout(); plt.savefig(OUT/"best_confusion_matrix.png",dpi=300); plt.close()

print("\nPER-CELL-TYPE F1\n",f1.round(3))
print(f"\nBEST MODEL: {best}\nResults: {OUT.resolve()}")