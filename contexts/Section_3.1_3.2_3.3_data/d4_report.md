## d4 report (generated)
per_run_energy_per_flop.csv rows for mbpo/HalfCheetah-v5/seed_0: 22; included_in_cross_seed_avg values: [False]; run_dirs: ['results/mbpo/HalfCheetah-v5/seed_0/20260904_122958', 'results/mbpo/HalfCheetah-v5/seed_0/20260904_130915']
cross_seed_energy_per_flop.csv: n_seeds values for mbpo HalfCheetah-v5 rows: [5]; architecture signatures: ['bs1024_h1024x1024_ens7_mh200x200x200x200_mb256', 'bs256_h1024x1024_ens7_mh200x200x200x200_mb256', 'bs256_h256x256_ens7_mh200x200x200x200_mb256', 'bs256_h512x512_ens7_mh200x200x200x200_mb256', 'bs512_h1024x1024_ens7_mh200x200x200x200_mb256']
20260904_122958: n_epochs=100, n_train_episodes=100, last10pct_mean_return=np.float64(5936.559047246978), final_cumulative_reward=325883.6580644874, sum(model_train_epochs)=1196, sum(synthetic_transitions_generated)=990000
20260904_130915: n_epochs=100, n_train_episodes=100, last10pct_mean_return=np.float64(5936.559047246978), final_cumulative_reward=325883.6580644874, sum(model_train_epochs)=1196, sum(synthetic_transitions_generated)=990000
n differing metadata fields vs primary: {'20260904_122958': 7}
--- differing fields, primary 20260904_130915 vs 20260904_122958:
  end_time_utc: primary='2026-09-04T12:48:34.409574' | 20260904_122958='2026-09-04T11:01:08.461191'
  experiment_config.gpu_max_clock_mhz: primary=200 | 20260904_122958=2000
  experiment_config.gpu_min_clock_mhz: primary=200 | 20260904_122958=2000
  start_time_utc: primary='2026-09-04T11:09:15.306557' | 20260904_122958='2026-09-04T10:29:59.128368'
  thermal_gate.final_power_w: primary=21.006 | 20260904_122958=26.467
  thermal_gate.final_temp_c: primary=44.0 | 20260904_122958=45.0
  thermal_gate.waited_seconds: primary=1.0728836059570312e-05 | 20260904_122958=1.1682510375976562e-05
