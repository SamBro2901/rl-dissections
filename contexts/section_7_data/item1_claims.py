import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
from scipy import stats
R = runs(); rows = []
def add(n, claim, found, ok, src): rows.append(dict(claim=n, item=claim, found=found, status="VERIFIED" if ok else "MISMATCH", source=src))
ct = pd.read_csv(REPO/"contexts/section_5.2_data/d1_config_table.csv")
b = ct[ct.config=="base"].set_index(["algo","env"])
def chk(n, algo, env, mu, sd):
    r = b.loc[(algo, env)]
    ok = abs(r.total_E_J-mu)/mu < 0.005 and abs(r.total_E_sd_J-sd)/sd < 0.01
    add(n, f"{algo} {env} total mean±sd J", f"{r.total_E_J:.1f} ± {r.total_E_sd_J:.1f}", ok, "contexts/section_5.2_data/d1_config_table.csv total_E_J/total_E_sd_J")
H, A = "HalfCheetah-v5", "Ant-v5"
chk(1,"sac",H,60910,254.1); chk(1,"sac",A,64380,588.3); chk(1,"td3",H,28110,310.7); chk(1,"td3",A,30520,431.9)
chk(2,"mbpo",H,207400,7451); chk(2,"mbpo",A,253100,20140)
chk(3,"tdmpc2",H,447100,3184); chk(3,"tdmpc2",A,493200,1782)
for env, v in [(H,70.46),(A,74.28)]:
    r=b.loc[("mbpo",env)]; s=r.dynamics_model_update_E_J/r.total_E_J*100
    add(2,f"mbpo {env} dynamics_model_update share %",f"{s:.2f}",abs(s-v)<0.01,"d1_config_table: dynamics_model_update_E_J/total_E_J (ratio of means)")
    # also mean of per-run shares
    g=R[(R.algo=="mbpo")&(R.env==env)&(R.config=="base")]; s2=(g.dynamics_model_update_E/g.total_E).mean()*100
    add(2,f"mbpo {env} dyn share % (mean of per-run shares)",f"{s2:.2f}",abs(s2-v)<0.01,"per-run via section_5_2_analysis.load()")
for env, v, j in [(H,47.83,50.62),(A,51.62,48.95)]:
    r=b.loc[("tdmpc2",env)]; s=r.rollout_E_J/r.total_E_J*100
    add(3,f"tdmpc2 {env} rollout share %",f"{s:.2f}",abs(s-v)<0.01,"d1_config_table rollout_E_J/total_E_J")
    add(3,f"tdmpc2 {env} total J/FLOP pJ",f"{r.total_J_per_F*1e12:.2f}",abs(r.total_J_per_F*1e12-j)<0.01,"d1_config_table total_J_per_F")
# 4,5 pooled
d2=pd.read_csv(REPO/"contexts/section_5.2_data/d2_total_correlation.csv"); print(d2.to_string())
cs=pd.read_csv(REPO/"flop_analysis/output/correlation/correlation_summary.csv"); print(cs.to_string())
E=pd.read_csv(REPO/"flop_analysis/output/correlation/config_means.csv"); print(E.head(3).to_string())
for name,df in [("d2",d2),("corr_summary",cs)]:
    df.to_csv(OUT/f"item1_src_{name}.csv",index=False)
# claim 6: jpf range training total
for env,lo,hi in [(H,19.11,1421),(A,19.82,1232)]:
    s=ct[ct.env==env].sort_values("total_J_per_F"); 
    a,z=s.iloc[0],s.iloc[-1]
    add(6,f"{env} total J/FLOP min pJ",f"{a.total_J_per_F*1e12:.2f} ({a.algo},{a.config})",abs(a.total_J_per_F*1e12-lo)<0.01,"d1_config_table total_J_per_F")
    add(6,f"{env} total J/FLOP max pJ",f"{z.total_J_per_F*1e12:.1f} ({z.algo},{z.config})",abs(z.total_J_per_F*1e12-hi)/hi<0.005,"d1_config_table total_J_per_F")
pd.DataFrame(rows).to_csv(OUT/"item1_claims_partial.csv",index=False)
print(pd.DataFrame(rows).to_string())
