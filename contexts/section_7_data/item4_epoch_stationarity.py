import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
T=pd.read_pickle(OUT/"_task_table.pkl"); T=T[(T.config=="base")&T.epoch.notna()].copy(); T["E_J"]=T.energy_consumed*3.6e6
SEGS=["rollout","gradient_updates","dynamics_model_update","synthetic_rollout_generation"]
TRAIN=SEGS  # training total per epoch = sum of per-epoch measured segments present (world_model_pretrain is one-off, no epoch index)
rows=[]
def stats(x):  # x: array len 100
    n=len(x); e=np.arange(n); m=x.mean()
    sl=np.polyfit(e,x,1)[0]
    return dict(ratio_0_9_over_10_99=x[:10].mean()/x[10:].mean(), slope_pct_per_100ep=sl*100/m*100/100 if False else sl*100/m*100, cv=x.std(ddof=1)/m)
for (a,env),g in T.groupby(["algo","env"]):
    P=g.pivot_table(index=["seed","epoch"],columns="seg",values="E_J",aggfunc="sum")
    for seg in SEGS+["TRAINING_TOTAL"]:
        if seg=="TRAINING_TOTAL": cols=[c for c in TRAIN if c in P.columns]; ser=P[cols].sum(axis=1)
        else:
            if seg not in P.columns: continue
            ser=P[seg]
        per=[]; 
        for s,x in ser.groupby(level="seed"):
            x=x.sort_index().values; assert len(x)==100
            d=stats(x)
            if seg=="TRAINING_TOTAL":
                cum=np.cumsum(x); steps=1000*(np.arange(100)+1); r=np.corrcoef(steps,cum)[0,1]; d["r2_cum_energy_vs_cum_steps"]=r**2
            per.append(d)
        D=pd.DataFrame(per)
        rows.append(dict(algo=a,env=env,segment=seg,n_seeds=len(D),n_epochs=100,mean_epoch_energy_J=ser.groupby(level="seed").mean().mean(),
            ratio_ep0_9_over_ep10_99_mean_seed=D.ratio_0_9_over_10_99.mean(),
            ratio_ep0_9_over_ep10_99_of_seedmeans=(ser.groupby(level="epoch").mean().iloc[:10].mean()/ser.groupby(level="epoch").mean().iloc[10:].mean()),
            slope_pct_of_mean_per_100_epochs=D.slope_pct_per_100ep.mean(),slope_pct_min_over_seeds=D.slope_pct_per_100ep.min(),slope_pct_max_over_seeds=D.slope_pct_per_100ep.max(),
            cv_within_run_mean=D.cv.mean(),
            r2_cum_energy_vs_cum_steps=D.r2_cum_energy_vs_cum_steps.mean() if "r2_cum_energy_vs_cum_steps" in D else np.nan))
S=pd.DataFrame(rows); S.to_csv(OUT/"epoch_stationarity.csv",index=False)
pd.set_option("display.width",250); print(S.round(4).drop(columns=["n_seeds","n_epochs"]).to_string())
