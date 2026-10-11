import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
R=runs(); recs=[]
for _,r in R.iterrows():
    m=jload(r.path/"training_metrics.json"); ep=[e for e in m["episodes"] if e["phase"]=="train"]
    k=max(1,len(ep)//10); last=ep[-k:]; first=ep[:k]
    recs.append(dict(algo=r.algo,env=r.env,config=r.config,seed=r.seed,n_ep=len(ep),
      last_ret=np.mean([e["return"] for e in last]),first_ret=np.mean([e["return"] for e in first]),
      last_len=np.mean([e["length"] for e in last]),first_len=np.mean([e["length"] for e in first]),
      last_early=np.mean([e["length"]<1000 for e in last]),first_early=np.mean([e["length"]<1000 for e in first]),
      all_len_1000=all(e["length"]==1000 for e in ep)))
P=pd.DataFrame(recs); P.to_csv(OUT/"item2_learning_per_run.csv",index=False)
rows=[]
for (a,e,c),g in P.groupby(["algo","env","config"]):
    ant=e=="Ant-v5"
    d=dict(algo=a,env=e,config=c,n_seeds=len(g),last10_return_mean=g.last_ret.mean(),last10_return_sd=g.last_ret.std(ddof=1),
      first10_return_mean=g.first_ret.mean(),last_minus_first=g.last_ret.mean()-g.first_ret.mean(),
      n_episodes_mean=g.n_ep.mean(),n_episodes_min=g.n_ep.min(),n_episodes_max=g.n_ep.max(),
      frac_early_terminated_last10=g.last_early.mean() if ant else np.nan,
      flag_last_minus_first_le_0=bool(g.last_ret.mean()-g.first_ret.mean()<=0),
      first10_mean_length=g.first_len.mean(),last10_mean_length=g.last_len.mean(),frac_early_terminated_first10=g.first_early.mean() if ant else np.nan,
      n_seeds_last_gt_first=int((g.last_ret>g.first_ret).sum()))
    rows.append(d)
L=pd.DataFrame(rows).sort_values(["env","algo","config"]); L.to_csv(OUT/"learning_check.csv",index=False)
pd.set_option("display.width",250)
print(L[["algo","env","config","n_seeds","last10_return_mean","last10_return_sd","first10_return_mean","last_minus_first","n_episodes_mean","n_episodes_min","n_episodes_max","frac_early_terminated_last10","first10_mean_length","last10_mean_length","n_seeds_last_gt_first"]].round(1).to_string())
print("flagged:",L[L.flag_last_minus_first_le_0][["algo","env","config"]].values.tolist())
print("HC any non-1000 ep:", (~P[P.env=="HalfCheetah-v5"].all_len_1000).sum(), " Ant runs with non-1000:", (~P[P.env=="Ant-v5"].all_len_1000).sum())

# supplementary: length-normalised return (sum return / sum length over the same first/last 10% episodes), Ant only matters
rec2=[]
for _,r in R.iterrows():
    ep=[e for e in jload(r.path/"training_metrics.json")["episodes"] if e["phase"]=="train"]; k=max(1,len(ep)//10)
    f,l=ep[:k],ep[-k:]
    rec2.append(dict(algo=r.algo,env=r.env,config=r.config,seed=r.seed,
      first_per_step=sum(e["return"] for e in f)/sum(e["length"] for e in f),last_per_step=sum(e["return"] for e in l)/sum(e["length"] for e in l)))
Q=pd.DataFrame(rec2).groupby(["algo","env","config"]).agg(first10_return_per_step=("first_per_step","mean"),last10_return_per_step=("last_per_step","mean")).reset_index()
Q["last_minus_first_per_step"]=Q.last10_return_per_step-Q.first10_return_per_step
Q.to_csv(OUT/"learning_check_per_step_supplement.csv",index=False)
print(Q[(Q.env=="Ant-v5")].round(3).to_string()); print("Ant configs with per-step last<=first:",Q[(Q.env=="Ant-v5")&(Q.last_minus_first_per_step<=0)][["algo","config"]].values.tolist())
