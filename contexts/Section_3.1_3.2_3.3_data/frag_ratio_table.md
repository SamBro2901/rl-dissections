| algo | env | sweep | values_numeric_order | rollout_energy_ratio | rollout_flops_ratio | gradient_updates_energy_ratio | gradient_updates_flops_ratio |
|---|---|---|---|---|---|---|---|
| sac | HalfCheetah-v5 | utd | 1 < 2 < 4 | 1.02291 | 1 | 3.97851 | 4 |
| sac | HalfCheetah-v5 | width | 256 < 512 < 1024 | 1.09536 | 14.7789 | 1.29439 | 15.0778 |
| sac | HalfCheetah-v5 | batch_size | 256 < 512 < 1024 | 1.10116 | 1 | 1.36296 | 4 |
| sac | Ant-v5 | utd | 1 < 2 < 4 | 0.991094 | 1 | 3.94723 | 4 |
| sac | Ant-v5 | width | 256 < 512 < 1024 | 1.05642 | 12.1485 | 1.30873 | 12.7506 |
| sac | Ant-v5 | batch_size | 256 < 512 < 1024 | 1.04821 | 1 | 1.40742 | 4 |
| mbpo | HalfCheetah-v5 | utd | 1 < 2 < 4 | 0.980576 | 1 | 3.94953 | 4 |
| mbpo | HalfCheetah-v5 | width | 256 < 512 < 1024 | 1.05 | 14.7789 | 1.31758 | 15.0778 |
| mbpo | HalfCheetah-v5 | batch_size | 256 < 512 < 1024 | 0.973044 | 1 | 1.32981 | 4 |
| mbpo | Ant-v5 | utd | 1 < 2 < 4 | 0.97357 | 1 | 3.94766 | 4 |
| mbpo | Ant-v5 | width | 256 < 512 < 1024 | 1.05367 | 12.1485 | 1.3334 | 12.7506 |
| mbpo | Ant-v5 | batch_size | 256 < 512 < 1024 | 0.973371 | 1 | 1.41627 | 4 |
| mbpo | Ant-v5 | rollout_max_length | 1 < 15 < 25 | 1.00953 | 1 | 0.996954 | 1 |
| td3 | HalfCheetah-v5 | utd | 1 < 2 < 4 | 0.978413 | 1 | 3.96862 | 4 |
| td3 | HalfCheetah-v5 | width | 256 < 512 < 1024 | 1.13766 | 15.0108 | 1.22498 | 15.1691 |
| td3 | HalfCheetah-v5 | batch_size | 100 < 256 < 512 < 1024 | 1.24203 | 1 | 1.73465 | 10.24 |
| td3 | Ant-v5 | utd | 1 < 2 < 4 | 0.981591 | 1 | 3.96689 | 4 |
| td3 | Ant-v5 | width | 256 < 512 < 1024 | 1.08255 | 12.3252 | 1.23291 | 12.8898 |
| td3 | Ant-v5 | batch_size | 100 < 256 < 512 < 1024 | 1.10398 | 1 | 1.81592 | 10.24 |
| tdmpc2 | HalfCheetah-v5 | num_q | 3 < 5 < 7 | 1.33083 | 1.36202 | 1.61187 | 1.84571 |
| tdmpc2 | HalfCheetah-v5 | horizon | 1 < 3 < 5 | 2.13251 | 2.14942 | 1.94656 | 4.02928 |
| tdmpc2 | Ant-v5 | num_q | 3 < 5 < 7 | 1.25952 | 1.29063 | 1.57108 | 1.77704 |
| tdmpc2 | Ant-v5 | horizon | 1 < 3 < 5 | 2.37033 | 2.42952 | 1.92872 | 4.07233 |
