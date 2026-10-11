import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
T=pd.read_pickle(OUT/"_task_table.pkl"); T=T[(T.config=="base")&T.seg.isin(["rollout","gradient_updates","synthetic_rollout_generation","dynamics_model_update"])].copy()
T["E_J"]=T.energy_consumed*3.6e6; T["P_W"]=T.E_J/T.duration
for c in ["cpu","gpu","ram"]: T[f"{c}_J"]=T[f"{c}_energy"]*3.6e6; T[f"{c}_W"]=T[f"{c}_J"]/T.duration
rows=[]
for (a,e,s),g in T.groupby(["algo","env","seg"]):
    d=dict(algo=a,env=e,segment=s,n_tasks=len(g))
    for nm,col in [("duration_s","duration"),("energy_J","E_J"),("power_W","P_W"),("gpu_power_W","gpu_W"),("cpu_power_W","cpu_W"),("ram_power_W","ram_W")]:
        d[f"{nm}_min"],d[f"{nm}_median"],d[f"{nm}_max"]=g[col].min(),g[col].median(),g[col].max()
    d["n_gpu_energy_le0"]=int((g.gpu_energy<=0).sum()); d["n_cpu_energy_le0"]=int((g.cpu_energy<=0).sum()); d["n_ram_energy_le0"]=int((g.ram_energy<=0).sum())
    d["n_total_energy_le0"]=int((g.energy_consumed<=0).sum())
    rows.append(d)
S=pd.DataFrame(rows); S.to_csv(OUT/"item3b_short_task_stats.csv",index=False)
pd.set_option("display.width",250)
print(S[["algo","env","segment","n_tasks","duration_s_min","duration_s_median","duration_s_max","energy_J_min","energy_J_median","energy_J_max","power_W_min","power_W_median","power_W_max","n_gpu_energy_le0","n_cpu_energy_le0","n_total_energy_le0"]].round(3).to_string())
# distinct-value check of GPU energy resolution on short tasks: how many tasks share an identical gpu_energy
g=T[T.seg=="rollout"]; print("rollout tasks share of unique gpu_energy values:", g.gpu_energy.nunique()/len(g))
print("corr(duration, energy) rollout, per algo/env:"); print(g.groupby(["algo","env"]).apply(lambda x: x.duration.corr(x.E_J)).round(3))

# MBPO synthetic_rollout_generation detail
s=T[T.seg=="synthetic_rollout_generation"]
print("\nsynthetic: gpu_energy==0:",int((s.gpu_energy==0).sum()),"<0:",int((s.gpu_energy<0).sum()),"of",len(s))
print(s.groupby("env").agg(n=("duration","size"),dur_med=("duration","median"),gpu_zero=("gpu_energy",lambda x:int((x==0).sum())),E_sum_J=("E_J","sum"),gpu_sum_J=("gpu_J","sum"),cpu_sum_J=("cpu_J","sum"),ram_sum_J=("ram_J","sum")))
tot=pd.read_pickle(OUT/"_task_table.pkl"); tot=tot[(tot.config=="base")&(tot.algo=="mbpo")&~tot.seg.isin(["idle_baseline_head","idle_baseline_tail","warmup"])]
tot["E_J"]=tot.energy_consumed*3.6e6
print((tot[tot.seg=="synthetic_rollout_generation"].groupby("env").E_J.sum()/tot.groupby("env").E_J.sum()*100).rename("synthetic share of mbpo baseline training total %"))
# duration bucket vs gpu energy zero (all tasks, all segments, baselines)
A=pd.read_pickle(OUT/"_task_table.pkl"); A=A[~A.seg.isin(["idle_baseline_head","idle_baseline_tail"])]
A["b"]=pd.cut(A.duration,[0,0.01,0.025,0.05,0.1,0.2,0.5,1,5,1e9])
print(A.groupby("b",observed=True).agg(n=("duration","size"),gpu_le0=("gpu_energy",lambda x:int((x<=0).sum())),cpu_le0=("cpu_energy",lambda x:int((x<=0).sum()))))
A.groupby(["b"],observed=True).agg(n=("duration","size"),gpu_energy_le0=("gpu_energy",lambda x:int((x<=0).sum())),cpu_energy_le0=("cpu_energy",lambda x:int((x<=0).sum()))).reset_index().assign(b=lambda d:d.b.astype(str)).to_csv(OUT/"item3b_gpu_zero_by_duration_bucket_all280.csv",index=False)
A[A.seg=="synthetic_rollout_generation"].groupby("env").agg(n=("duration","size")).to_csv(OUT/"_tmp.csv")
