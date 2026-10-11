"""Parse emissions_base_*.csv of all 280 canonical runs into one long table (cache for items 3,4,5,11)."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
import glob, re
R = runs(); out=[]
for _, r in R.iterrows():
    fs = glob.glob(str(r.path/"emissions_base_*.csv"))
    assert len(fs)==1, (r.run_dir, fs)
    d = pd.read_csv(fs[0])
    d["run_dir"]=r.run_dir; d["algo"]=r.algo; d["env"]=r.env; d["config"]=r.config; d["seed"]=r.seed
    out.append(d[["run_dir","algo","env","config","seed","task_name","timestamp","duration","cpu_energy","gpu_energy","ram_energy","energy_consumed","cpu_power","gpu_power","ram_power"]])
T = pd.concat(out, ignore_index=True)
T["seg"] = T.task_name.str.replace(r"_\d+$","",regex=True)
T["epoch"] = T.task_name.str.extract(r"_(\d+)$")[0].astype(float)
T.to_pickle(OUT/"_task_table.pkl")
print(len(T)); print(T.groupby(["algo","seg"]).size().unstack(0).to_string())
