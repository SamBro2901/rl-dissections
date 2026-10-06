| group_id | algo | env | sweep_name | sweep_value | is_baseline | hidden_sizes | utd | batch_size | rollout_max_length | num_q | horizon | episodic | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G01 | sac | HalfCheetah-v5 | baseline |  | True | 1024x1024 | 1 | 256 |  |  |  |  | 5 |
| G02 | sac | HalfCheetah-v5 | utd | 2 | False | 1024x1024 | 2 | 256 |  |  |  |  | 5 |
| G03 | sac | HalfCheetah-v5 | utd | 4 | False | 1024x1024 | 4 | 256 |  |  |  |  | 5 |
| G04 | sac | HalfCheetah-v5 | width | 256 | False | 256x256 | 1 | 256 |  |  |  |  | 5 |
| G05 | sac | HalfCheetah-v5 | width | 512 | False | 512x512 | 1 | 256 |  |  |  |  | 5 |
| G06 | sac | HalfCheetah-v5 | batch_size | 512 | False | 1024x1024 | 1 | 512 |  |  |  |  | 5 |
| G07 | sac | HalfCheetah-v5 | batch_size | 1024 | False | 1024x1024 | 1 | 1024 |  |  |  |  | 5 |
| G08 | sac | Ant-v5 | baseline |  | True | 1024x1024 | 1 | 256 |  |  |  |  | 5 |
| G09 | sac | Ant-v5 | utd | 2 | False | 1024x1024 | 2 | 256 |  |  |  |  | 5 |
| G10 | sac | Ant-v5 | utd | 4 | False | 1024x1024 | 4 | 256 |  |  |  |  | 5 |
| G11 | sac | Ant-v5 | width | 256 | False | 256x256 | 1 | 256 |  |  |  |  | 5 |
| G12 | sac | Ant-v5 | width | 512 | False | 512x512 | 1 | 256 |  |  |  |  | 5 |
| G13 | sac | Ant-v5 | batch_size | 512 | False | 1024x1024 | 1 | 512 |  |  |  |  | 5 |
| G14 | sac | Ant-v5 | batch_size | 1024 | False | 1024x1024 | 1 | 1024 |  |  |  |  | 5 |
| G15 | mbpo | HalfCheetah-v5 | baseline |  | True | 1024x1024 | 1 | 256 | 1 |  |  |  | 5 |
| G16 | mbpo | HalfCheetah-v5 | utd | 2 | False | 1024x1024 | 2 | 256 | 1 |  |  |  | 5 |
| G17 | mbpo | HalfCheetah-v5 | utd | 4 | False | 1024x1024 | 4 | 256 | 1 |  |  |  | 5 |
| G18 | mbpo | HalfCheetah-v5 | width | 256 | False | 256x256 | 1 | 256 | 1 |  |  |  | 5 |
| G19 | mbpo | HalfCheetah-v5 | width | 512 | False | 512x512 | 1 | 256 | 1 |  |  |  | 5 |
| G20 | mbpo | HalfCheetah-v5 | batch_size | 512 | False | 1024x1024 | 1 | 512 | 1 |  |  |  | 5 |
| G21 | mbpo | HalfCheetah-v5 | batch_size | 1024 | False | 1024x1024 | 1 | 1024 | 1 |  |  |  | 5 |
| G22 | mbpo | Ant-v5 | baseline |  | True | 1024x1024 | 1 | 256 | 25 |  |  |  | 5 |
| G23 | mbpo | Ant-v5 | utd | 2 | False | 1024x1024 | 2 | 256 | 25 |  |  |  | 5 |
| G24 | mbpo | Ant-v5 | utd | 4 | False | 1024x1024 | 4 | 256 | 25 |  |  |  | 5 |
| G25 | mbpo | Ant-v5 | width | 256 | False | 256x256 | 1 | 256 | 25 |  |  |  | 5 |
| G26 | mbpo | Ant-v5 | width | 512 | False | 512x512 | 1 | 256 | 25 |  |  |  | 5 |
| G27 | mbpo | Ant-v5 | batch_size | 512 | False | 1024x1024 | 1 | 512 | 25 |  |  |  | 5 |
| G28 | mbpo | Ant-v5 | batch_size | 1024 | False | 1024x1024 | 1 | 1024 | 25 |  |  |  | 5 |
| G29 | mbpo | Ant-v5 | rollout_max_length | 1 | False | 1024x1024 | 1 | 256 | 1 |  |  |  | 5 |
| G30 | mbpo | Ant-v5 | rollout_max_length | 15 | False | 1024x1024 | 1 | 256 | 15 |  |  |  | 5 |
| G31 | td3 | HalfCheetah-v5 | baseline |  | True | 1024x1024 | 1 | 100 |  |  |  |  | 5 |
| G32 | td3 | HalfCheetah-v5 | utd | 2 | False | 1024x1024 | 2 | 100 |  |  |  |  | 5 |
| G33 | td3 | HalfCheetah-v5 | utd | 4 | False | 1024x1024 | 4 | 100 |  |  |  |  | 5 |
| G34 | td3 | HalfCheetah-v5 | width | 256 | False | 256x256 | 1 | 100 |  |  |  |  | 5 |
| G35 | td3 | HalfCheetah-v5 | width | 512 | False | 512x512 | 1 | 100 |  |  |  |  | 5 |
| G36 | td3 | HalfCheetah-v5 | batch_size | 256 | False | 1024x1024 | 1 | 256 |  |  |  |  | 5 |
| G37 | td3 | HalfCheetah-v5 | batch_size | 512 | False | 1024x1024 | 1 | 512 |  |  |  |  | 5 |
| G38 | td3 | HalfCheetah-v5 | batch_size | 1024 | False | 1024x1024 | 1 | 1024 |  |  |  |  | 5 |
| G39 | td3 | Ant-v5 | baseline |  | True | 1024x1024 | 1 | 100 |  |  |  |  | 5 |
| G40 | td3 | Ant-v5 | utd | 2 | False | 1024x1024 | 2 | 100 |  |  |  |  | 5 |
| G41 | td3 | Ant-v5 | utd | 4 | False | 1024x1024 | 4 | 100 |  |  |  |  | 5 |
| G42 | td3 | Ant-v5 | width | 256 | False | 256x256 | 1 | 100 |  |  |  |  | 5 |
| G43 | td3 | Ant-v5 | width | 512 | False | 512x512 | 1 | 100 |  |  |  |  | 5 |
| G44 | td3 | Ant-v5 | batch_size | 256 | False | 1024x1024 | 1 | 256 |  |  |  |  | 5 |
| G45 | td3 | Ant-v5 | batch_size | 512 | False | 1024x1024 | 1 | 512 |  |  |  |  | 5 |
| G46 | td3 | Ant-v5 | batch_size | 1024 | False | 1024x1024 | 1 | 1024 |  |  |  |  | 5 |
| G47 | tdmpc2 | HalfCheetah-v5 | baseline |  | True | 512x512 | 1 | 256 |  | 5 | 3 | False | 5 |
| G48 | tdmpc2 | HalfCheetah-v5 | num_q | 3 | False | 512x512 | 1 | 256 |  | 3 | 3 | False | 5 |
| G49 | tdmpc2 | HalfCheetah-v5 | num_q | 7 | False | 512x512 | 1 | 256 |  | 7 | 3 | False | 5 |
| G50 | tdmpc2 | HalfCheetah-v5 | horizon | 1 | False | 512x512 | 1 | 256 |  | 5 | 1 | False | 5 |
| G51 | tdmpc2 | HalfCheetah-v5 | horizon | 5 | False | 512x512 | 1 | 256 |  | 5 | 5 | False | 5 |
| G52 | tdmpc2 | Ant-v5 | baseline |  | True | 512x512 | 1 | 256 |  | 5 | 3 | True | 5 |
| G53 | tdmpc2 | Ant-v5 | num_q | 3 | False | 512x512 | 1 | 256 |  | 3 | 3 | True | 5 |
| G54 | tdmpc2 | Ant-v5 | num_q | 7 | False | 512x512 | 1 | 256 |  | 7 | 3 | True | 5 |
| G55 | tdmpc2 | Ant-v5 | horizon | 1 | False | 512x512 | 1 | 256 |  | 5 | 1 | True | 5 |
| G56 | tdmpc2 | Ant-v5 | horizon | 5 | False | 512x512 | 1 | 256 |  | 5 | 5 | True | 5 |
