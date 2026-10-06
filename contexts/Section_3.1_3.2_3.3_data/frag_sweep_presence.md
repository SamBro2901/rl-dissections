| sweep | algo | HalfCheetah-v5 (values; *=baseline value; group_ids) | Ant-v5 |
|---|---|---|---|
| utd | sac | 1*, 2, 4 (G01,G02,G03) | 1*, 2, 4 (G08,G09,G10) |
| utd | mbpo | 1*, 2, 4 (G15,G16,G17) | 1*, 2, 4 (G22,G23,G24) |
| utd | td3 | 1*, 2, 4 (G31,G32,G33) | 1*, 2, 4 (G39,G40,G41) |
| utd | tdmpc2 | — absent | — absent |
| width | sac | 256, 512, 1024* (G01,G04,G05) | 256, 512, 1024* (G08,G11,G12) |
| width | mbpo | 256, 512, 1024* (G15,G18,G19) | 256, 512, 1024* (G22,G25,G26) |
| width | td3 | 256, 512, 1024* (G31,G34,G35) | 256, 512, 1024* (G39,G42,G43) |
| width | tdmpc2 | — absent | — absent |
| batch_size | sac | 256*, 512, 1024 (G01,G06,G07) | 256*, 512, 1024 (G08,G13,G14) |
| batch_size | mbpo | 256*, 512, 1024 (G15,G20,G21) | 256*, 512, 1024 (G22,G27,G28) |
| batch_size | td3 | 100*, 256, 512, 1024 (G31,G36,G37,G38) | 100*, 256, 512, 1024 (G39,G44,G45,G46) |
| batch_size | tdmpc2 | — absent | — absent |
| rollout_max_length | sac | — absent | — absent |
| rollout_max_length | mbpo | — absent | 1, 15, 25* (G22,G29,G30) |
| rollout_max_length | td3 | — absent | — absent |
| rollout_max_length | tdmpc2 | — absent | — absent |
| num_q | sac | — absent | — absent |
| num_q | mbpo | — absent | — absent |
| num_q | td3 | — absent | — absent |
| num_q | tdmpc2 | 3, 5*, 7 (G47,G48,G49) | 3, 5*, 7 (G52,G53,G54) |
| horizon | sac | — absent | — absent |
| horizon | mbpo | — absent | — absent |
| horizon | td3 | — absent | — absent |
| horizon | tdmpc2 | 1, 3*, 5 (G47,G50,G51) | 1, 3*, 5 (G52,G55,G56) |
