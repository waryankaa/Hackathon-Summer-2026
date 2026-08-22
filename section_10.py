import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

FILE,OUT,TARGET,GROUP=Path("segments/segment_10.csv"),Path("segment_10_hierarchical_results"),"MERFISH_cell_type_annotation","Excitatory_vs_Inhibitory"; OUT.mkdir(exist_ok=True)
META=["Unnamed: 0","Datasets","volume","center_x","center_y",TARGET,GROUP,"Region","Segment","Gender","Mouse_ID","AP_position","Section_ID"]

df=pd.read_csv(FILE); df=df[df[TARGET].notna()&df[GROUP].notna()].reset_index(drop=True)
genes=[c for c in df.columns if c not in META]; X=df[genes].apply(pd.to_numeric,errors="coerce").fillna(0)
genes=[g for g in genes if X[g].nunique()>1]; X=X[genes]; y=df[TARGET].astype(str); group=df[GROUP].astype(str).str.lower()

norm=lambda x:x.div(x.sum(1).replace(0,1),axis=0)*x.sum(1).median()
mi=lambda X,y:mutual_info_classif(X,y,discrete_features=True,random_state=42)
log=FunctionTransformer(np.log1p,validate=False); normalize=FunctionTransformer(norm,validate=False)

def models(k=20):
    et=ExtraTreesClassifier(n_estimators=500,class_weight="balanced",max_features="sqrt",random_state=42,n_jobs=-1)
    lr=LogisticRegression(max_iter=5000,class_weight="balanced",random_state=42)
    svm=SVC(kernel="rbf",class_weight="balanced",C=1)
    sel=lambda:SelectKBest(mi,k=min(k,len(genes)))
    return {
      "ET_log_MI":Pipeline([("log",log),("mi",sel()),("m",et)]),
      "ET_norm_log_MI":Pipeline([("norm",normalize),("log",log),("mi",sel()),("m",et)]),
      "LR_norm_log":Pipeline([("norm",normalize),("log",log),("scale",StandardScaler()),("m",lr)]),
      "LR_PCA":Pipeline([("norm",normalize),("log",log),("scale",StandardScaler()),("pca",PCA(n_components=10)),("m",lr)]),
      "SVM_norm_log":Pipeline([("norm",normalize),("log",log),("scale",StandardScaler()),("m",svm)]),
      "SVM_PCA":Pipeline([("norm",normalize),("log",log),("scale",StandardScaler()),("pca",PCA(n_components=10)),("m",svm)])
    }

rows=[]
for name,model in models().items():
    scores=[]
    for seed in range(5):
        pred=pd.Series(index=df.index,dtype=object)
        for g in group.unique():
            idx=np.where(group==g)[0]; cv=StratifiedKFold(3,shuffle=True,random_state=42+seed)
            pred.iloc[idx]=cross_val_predict(model,X.iloc[idx],y.iloc[idx],cv=cv,n_jobs=-1)
        scores.append([accuracy_score(y,pred),balanced_accuracy_score(y,pred),f1_score(y,pred,average="macro")])
    s=np.array(scores); rows.append([name,*s.mean(0),*s.std(0)])

r=pd.DataFrame(rows,columns=["model","accuracy","balanced_accuracy","macro_f1","accuracy_sd","balanced_accuracy_sd","macro_f1_sd"]).sort_values(["macro_f1","balanced_accuracy"],ascending=False)
print("\n",r.round(3).to_string(index=False)); r.to_csv(OUT/"model_comparison.csv",index=False)