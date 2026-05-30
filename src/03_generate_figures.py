import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})
BLUE="#2c6fbb"; GREEN="#3a9d5d"; ORANGE="#e08a1e"; RED="#cc3b3b"; GREY="#9aa0a6"
OUT="figures"; import os; os.makedirs(OUT, exist_ok=True)

# ---------- Fig 4: Confusion matrix (XGBoost, derived from real metrics+totals) ----------
cm=np.array([[19459,2978],[1360,8796]])
labels=np.array([["TN = 19,459","FP = 2,978"],["FN = 1,360","TP = 8,796"]])
fig,ax=plt.subplots(figsize=(6,4.6))
cmap=LinearSegmentedColormap.from_list("b",["#f4f8fd","#2c6fbb"])
im=ax.imshow(cm,cmap=cmap)
for i in range(2):
    for j in range(2):
        pct=cm[i,j]/cm.sum()*100
        ax.text(j,i,f"{labels[i,j]}\n({pct:.1f}%)",ha="center",va="center",
                fontsize=12,fontweight="bold",
                color="white" if cm[i,j]>cm.max()*0.5 else "#222")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Non-dropout","Dropout"]); ax.set_yticklabels(["Non-dropout","Dropout"])
ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
ax.set_title("Confusion Matrix — XGBoost (best model), OULAD", fontweight="bold", fontsize=12)
fig.colorbar(im,fraction=0.046,pad=0.04)
plt.savefig(f"{OUT}/fig4_confusion.png"); plt.close()

# ---------- Fig 5: Model performance comparison (real per-model metrics) ----------
models=["Random\nForest","XGBoost","CNN","LSTM","CNN-LSTM"]
acc=[86.29,86.69,86.02,85.42,84.14]
prec=[72.61,74.71,71.29,70.26,68.38]
rec=[89.97,86.61,92.35,92.32,91.48]
auc=[93.91,94.40,94.16,93.61,92.50]  # AUC*100 for same axis
x=np.arange(len(models)); w=0.2
fig,ax=plt.subplots(figsize=(8.2,4.8))
ax.bar(x-1.5*w,acc,w,label="Accuracy",color=BLUE)
ax.bar(x-0.5*w,prec,w,label="Precision",color=ORANGE)
ax.bar(x+0.5*w,rec,w,label="Recall",color=GREEN)
ax.bar(x+1.5*w,auc,w,label="AUC×100",color=GREY)
# highlight best (XGBoost) accuracy bar
ax.bar(x[1]-1.5*w,acc[1],w,color=RED,zorder=3)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(60,100); ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison (OULAD, 5-fold CV)",fontweight="bold")
ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,-0.22),frameon=False)
ax.grid(axis="y",alpha=0.25)
for xi,a in zip(x,auc): ax.text(xi+1.5*w,a+0.4,f"{a/100:.3f}",ha="center",fontsize=7,color="#555")
plt.savefig(f"{OUT}/fig5_performance.png"); plt.close()

# ---------- Fig 6: SHAP top-10 (real values) ----------
feat=["Active span (days)","Assessments taken","Active days","First assessment score",
      "Late-period clicks","Avg. assessment score","Total clicks","Std. daily clicks",
      "Mean daily clicks","Early-period clicks"]
val=[0.2677,0.0430,0.0404,0.0261,0.0252,0.0216,0.0182,0.0176,0.0174,0.0145]
order=np.argsort(val)
fig,ax=plt.subplots(figsize=(8,5))
colors=[RED if val[i]==max(val) else BLUE for i in order]
ax.barh([feat[i] for i in order],[val[i] for i in order],color=colors)
for i,idx in enumerate(order):
    ax.text(val[idx]+0.003,i,f"{val[idx]:.4f}",va="center",fontsize=8,color="#444")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top-10 SHAP Feature Importance (OULAD)",fontweight="bold")
ax.set_xlim(0,0.30); ax.grid(axis="x",alpha=0.25)
plt.savefig(f"{OUT}/fig6_shap.png"); plt.close()

# ---------- Fig 7: EWS risk distribution (real counts) ----------
fig,ax=plt.subplots(figsize=(6.2,4.6))
risk=["Low\n(P<0.30)","Medium\n(0.30–0.60)","High\n(P≥0.60)"]
n=[17410,2579,12604]; pct=[53.4,7.9,38.7]; cols=[GREEN,ORANGE,RED]
bars=ax.bar(risk,n,color=cols)
for b,c,pp in zip(bars,n,pct):
    ax.text(b.get_x()+b.get_width()/2,c+250,f"{c:,}\n({pp}%)",ha="center",fontsize=10,fontweight="bold")
ax.set_ylabel("Number of students")
ax.set_title("EWS Three-Tier Risk Distribution (OULAD, n=32,593)",fontweight="bold")
ax.set_ylim(0,19500); ax.grid(axis="y",alpha=0.25)
plt.savefig(f"{OUT}/fig7_ews.png"); plt.close()

print("4 figures generated:", os.listdir(OUT))
PYEOF
