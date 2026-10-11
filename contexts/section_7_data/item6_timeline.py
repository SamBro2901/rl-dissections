import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
R=runs().set_index("run_dir")
rows=[]
for p in (REPO/"results").rglob("metadata.json"):
    m=jload(p); rd=p.parent.relative_to(REPO).as_posix()
    rows.append(dict(run_dir=rd,algo=m["algo_name"],env=m["env_id"],seed=m["seed"],start=pd.Timestamp(m["start_time_utc"]),end=pd.Timestamp(m["end_time_utc"]),git_commit=m["git_commit"][:7],
      canonical=m["seed"] in CANON,config=R.config.get(rd,"DEV"),dir_ts=p.parent.name))
A=pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
A["dur_h"]=(A.end-A.start).dt.total_seconds()/3600
A["gap_prev_min"]=(A.start-A.end.shift()).dt.total_seconds()/60
A["overlap"]=A.gap_prev_min<0
A["session"]=(A.gap_prev_min.isna()|(A.gap_prev_min>30)).cumsum()
A.to_csv(OUT/"execution_timeline_runs.csv",index=False)
print("runs",len(A),"canonical",A.canonical.sum(),"overlapping runs:",A.overlap.sum(),"min gap",A.gap_prev_min.min())
print("machine hours canonical (end-start sum):",A[A.canonical].dur_h.sum(),"; all 300:",A.dur_h.sum())
# sessions
S=[]
for s,g in A.groupby("session"):
    gc=g[g.canonical]
    S.append(dict(session=s,first_start_utc=g.start.min(),last_end_utc=g.end.max(),span_h=(g.end.max()-g.start.min()).total_seconds()/3600,n_runs=len(g),n_canonical=len(gc),n_dev=len(g)-len(gc),
      algo_env_configs="; ".join(f"{a}/{e}:{','.join(dict.fromkeys(x.config))}" for (a,e),x in g.groupby(["algo","env"],sort=False)),git_commits=",".join(sorted(g.git_commit.unique())),max_gap_in_session_min=g.gap_prev_min.iloc[1:].max()))
S=pd.DataFrame(S); S.to_csv(OUT/"execution_timeline.csv",index=False)
pd.set_option("display.width",300,"display.max_colwidth",160); print(S.to_string())
# per config dates
C=A[A.canonical].groupby(["algo","env","config"]).agg(first_start=("start","min"),last_end=("end","max"),session=("session","first"),n_sessions=("session","nunique"),commits=("git_commit",lambda x:",".join(sorted(set(x))))).reset_index()
C["date_utc"]=C.first_start.dt.date; C["last_date_utc"]=C.last_end.dt.date; C.to_csv(OUT/"execution_timeline_configs.csv",index=False)
# baseline vs sweep, same day?
pairs=[]
for (a,e),g in C.groupby(["algo","env"]):
    b=g[g.config=="base"].iloc[0]
    for _,r in g[g.config!="base"].iterrows():
        pairs.append(dict(kind="baseline_vs_sweep",algo=a,env=e,config_a="base",config_b=r.config,date_a=b.date_utc,date_b=r.date_utc,same_session=b.session==r.session,days_apart=(r.first_start.normalize()-b.first_start.normalize()).days,hours_apart=(r.first_start-b.first_start).total_seconds()/3600))
for a,g in C.groupby("algo"):
    for c in set(g[g.env=="HalfCheetah-v5"].config)&set(g[g.env=="Ant-v5"].config):
        h=g[(g.env=="HalfCheetah-v5")&(g.config==c)].iloc[0]; n=g[(g.env=="Ant-v5")&(g.config==c)].iloc[0]
        pairs.append(dict(kind="HC_vs_Ant_pair",algo=a,env="HC|Ant",config_a=c,config_b=c,date_a=h.date_utc,date_b=n.date_utc,same_session=h.session==n.session,days_apart=(n.first_start.normalize()-h.first_start.normalize()).days,hours_apart=(n.first_start-h.first_start).total_seconds()/3600))
P=pd.DataFrame(pairs); P["different_day"]=P.days_apart!=0; P.to_csv(OUT/"execution_timeline_pairs.csv",index=False)
print(P.groupby("kind").agg(n=("algo","size"),diff_day=("different_day","sum"),diff_session=("same_session",lambda x:int((~x).sum()))))
print(P[P.different_day].to_string())
# seed order consecutive?
chk=[]
for (a,e,c),g in A[A.canonical].groupby(["algo","env","config"]):
    g=g.sort_values("start"); order=list(g.seed); idx=g.index.tolist()
    consecutive=all(A.index.get_loc(idx[i+1])-A.index.get_loc(idx[i])==1 for i in range(len(idx)-1))
    chk.append(dict(algo=a,env=e,config=c,seed_order=order,in_list_order=order==CANON,consecutive_in_global_order=consecutive,
      contains_dev_between=int(A.loc[idx[0]:idx[-1]].canonical.eq(False).sum())))
K=pd.DataFrame(chk); K.to_csv(OUT/"execution_timeline_seed_order.csv",index=False)
print("configs with seeds in list order:",K.in_list_order.sum(),"/",len(K),"; consecutive:",K.consecutive_in_global_order.sum(),"; with dev run in between:",(K.contains_dev_between>0).sum())
print(K[~K.in_list_order|~K.consecutive_in_global_order].to_string())
