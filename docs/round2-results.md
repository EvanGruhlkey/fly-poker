# Round-two measured results

Validation experiments only. No new production brain has been promoted.

The strength objective remains unmet. Pressure-balanced imitation learned some folding, but still lost to the equity baseline at every stack size. Its PPO continuation regressed against equity at equal stacks. The retrained architecture controls show no clear FlyWire advantage at this sample size. Further scaling of these candidates is not justified.

Each sample is a seat-swapped pair on heldout deals. Intervals use the saved paired bootstrap. Different sample counts have different uncertainty; point estimates alone do not rank models.

| Candidate | Stacks (chips) | Opponent | Pairs | BB/100 | 95% CI | Gain over v2 BB/100 | Gain 95% CI |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| ppo | 400/400 | random | 128 | -93.95 | [-678.75, 473.27] | -400.20 | [-1050.60, 212.19] |
| ppo | 400/400 | passive | 128 | 177.93 | [-88.09, 449.43] | 207.62 | [-55.30, 467.59] |
| ppo | 400/400 | aggressive | 128 | -834.18 | [-1375.46, -321.86] | -673.63 | [-1211.01, -175.97] |
| ppo | 400/400 | equity | 128 | -1433.01 | [-1841.21, -1016.00] | -155.27 | [-698.08, 397.85] |
| ppo | 400/400 | teacher | 128 | -893.95 | [-1475.79, -356.43] | 20.51 | [-607.64, 628.54] |
| ppo | 400/400 | v2 | 128 | -298.63 | [-760.95, 139.28] | -298.63 | [-760.95, 139.28] |
| ppo | 80/720 | random | 128 | -185.35 | [-299.81, -77.34] | -241.60 | [-374.81, -112.50] |
| ppo | 80/720 | passive | 128 | 39.26 | [-48.24, 124.22] | 50.59 | [-27.93, 127.94] |
| ppo | 80/720 | aggressive | 128 | -193.95 | [-305.27, -90.82] | -141.21 | [-250.99, -35.54] |
| ppo | 80/720 | equity | 128 | -174.22 | [-273.05, -75.19] | -18.55 | [-138.67, 102.15] |
| ppo | 80/720 | teacher | 128 | -109.77 | [-200.20, -20.70] | -38.48 | [-152.74, 72.27] |
| ppo | 80/720 | v2 | 128 | -126.76 | [-234.38, -30.08] | -127.15 | [-231.26, -33.39] |
| ppo | 200/600 | random | 128 | -150.59 | [-417.39, 130.27] | -134.18 | [-395.53, 128.72] |
| ppo | 200/600 | passive | 128 | 107.62 | [-81.06, 293.56] | 66.21 | [-97.28, 230.29] |
| ppo | 200/600 | aggressive | 40 | -80.00 | [-499.38, 331.27] | 1.25 | [-376.91, 391.27] |
| ppo | 200/600 | equity | 0 | — | — | — | — |
| ppo | 200/600 | teacher | 0 | — | — | — | — |
| ppo | 200/600 | v2 | 0 | — | — | — | — |
| imitation | 400/400 | random | 64 | 124.22 | [-540.68, 826.19] | -339.84 | [-1343.75, 637.52] |
| imitation | 400/400 | passive | 64 | 32.81 | [-47.66, 121.88] | 196.88 | [-191.43, 604.71] |
| imitation | 400/400 | aggressive | 64 | -252.34 | [-1041.43, 503.96] | -162.50 | [-970.33, 646.97] |
| imitation | 400/400 | equity | 64 | -906.64 | [-1339.84, -511.27] | 607.81 | [-194.53, 1350.02] |
| imitation | 400/400 | teacher | 64 | -585.94 | [-1357.07, 183.24] | -185.16 | [-1115.64, 725.06] |
| imitation | 400/400 | v2 | 64 | -284.38 | [-1108.59, 510.96] | -284.38 | [-1108.59, 510.96] |
| imitation | 80/720 | random | 64 | -25.78 | [-203.12, 149.61] | -184.38 | [-411.74, 41.41] |
| imitation | 80/720 | passive | 64 | 20.31 | [-49.22, 91.43] | 39.06 | [-32.03, 109.38] |
| imitation | 80/720 | aggressive | 64 | -32.03 | [-188.30, 107.03] | -4.69 | [-171.11, 157.03] |
| imitation | 80/720 | equity | 64 | -423.05 | [-531.25, -314.06] | -210.94 | [-381.27, -46.88] |
| imitation | 80/720 | teacher | 64 | -191.41 | [-322.66, -61.71] | -142.97 | [-332.03, 38.30] |
| imitation | 80/720 | v2 | 64 | -85.94 | [-249.22, 60.16] | -78.91 | [-253.91, 85.16] |
| imitation | 200/600 | random | 64 | 52.34 | [-338.31, 425.81] | -43.75 | [-600.00, 466.41] |
| imitation | 200/600 | passive | 64 | 35.94 | [-60.96, 135.16] | 35.16 | [-164.08, 233.61] |
| imitation | 200/600 | aggressive | 64 | -347.66 | [-803.14, 85.16] | -296.88 | [-767.19, 153.14] |
| imitation | 200/600 | equity | 64 | -540.62 | [-832.05, -255.07] | 340.62 | [-92.21, 757.81] |
| imitation | 200/600 | teacher | 64 | -288.67 | [-713.70, 122.29] | 108.59 | [-356.27, 578.91] |
| imitation | 200/600 | v2 | 64 | -167.19 | [-614.06, 257.81] | -73.44 | [-535.98, 364.86] |
| pressure | 400/400 | random | 64 | 999.22 | [68.73, 1950.85] | 535.16 | [-167.19, 1268.01] |
| pressure | 400/400 | passive | 64 | -112.50 | [-593.03, 358.65] | 51.56 | [-475.06, 550.84] |
| pressure | 400/400 | aggressive | 64 | -221.88 | [-580.47, 0.00] | -132.03 | [-410.16, 14.06] |
| pressure | 400/400 | equity | 64 | -2266.80 | [-2842.22, -1694.53] | -752.34 | [-1404.69, -107.79] |
| pressure | 400/400 | teacher | 64 | -149.22 | [-919.14, 660.99] | 251.56 | [-744.57, 1253.18] |
| pressure | 400/400 | v2 | 64 | 195.31 | [-78.12, 578.12] | 195.31 | [-78.12, 578.12] |
| pressure | 80/720 | random | 64 | 167.19 | [-3.12, 342.59] | 8.59 | [-122.66, 133.59] |
| pressure | 80/720 | passive | 64 | -23.44 | [-124.22, 75.78] | -4.69 | [-103.12, 91.41] |
| pressure | 80/720 | aggressive | 64 | -22.66 | [-67.97, 0.00] | 4.69 | [-46.88, 74.22] |
| pressure | 80/720 | equity | 64 | -287.11 | [-397.27, -172.66] | -75.00 | [-214.06, 59.38] |
| pressure | 80/720 | teacher | 64 | -44.53 | [-190.62, 115.24] | 3.91 | [-107.03, 122.66] |
| pressure | 80/720 | v2 | 64 | -42.19 | [-114.84, 7.81] | -35.16 | [-94.53, 8.59] |
| pressure | 200/600 | random | 64 | 303.12 | [-129.73, 766.41] | 207.03 | [-142.99, 560.18] |
| pressure | 200/600 | passive | 64 | -58.59 | [-318.77, 183.61] | -59.38 | [-332.03, 214.86] |
| pressure | 200/600 | aggressive | 64 | -46.09 | [-138.28, 0.00] | 4.69 | [0.00, 14.06] |
| pressure | 200/600 | equity | 64 | -1171.09 | [-1475.42, -853.90] | -289.84 | [-698.44, 116.45] |
| pressure | 200/600 | teacher | 64 | -289.45 | [-692.58, 143.76] | 107.81 | [-295.35, 503.12] |
| pressure | 200/600 | v2 | 64 | 55.47 | [-107.03, 242.19] | 149.22 | [-79.69, 418.75] |
| pressure-imitation | 400/400 | random | 64 | -25.78 | [-752.04, 712.20] | -489.84 | [-1606.31, 586.74] |
| pressure-imitation | 400/400 | passive | 64 | 41.41 | [-10.16, 98.44] | 205.47 | [-181.25, 616.41] |
| pressure-imitation | 400/400 | aggressive | 64 | -575.78 | [-1253.12, 101.56] | -485.94 | [-1171.09, 194.55] |
| pressure-imitation | 400/400 | equity | 64 | -742.58 | [-1188.28, -308.58] | 771.88 | [-42.21, 1539.84] |
| pressure-imitation | 400/400 | teacher | 64 | -410.16 | [-981.25, 174.63] | -9.38 | [-1046.17, 932.87] |
| pressure-imitation | 400/400 | v2 | 64 | 152.34 | [-561.76, 860.23] | 152.34 | [-561.76, 860.23] |
| pressure-imitation | 80/720 | random | 64 | 15.62 | [-153.92, 187.12] | -142.97 | [-320.31, 32.83] |
| pressure-imitation | 80/720 | passive | 64 | 15.62 | [-55.47, 80.47] | 34.38 | [-39.84, 102.34] |
| pressure-imitation | 80/720 | aggressive | 64 | -89.84 | [-228.91, 37.50] | -62.50 | [-209.41, 81.27] |
| pressure-imitation | 80/720 | equity | 64 | -377.73 | [-473.05, -282.81] | -165.62 | [-331.25, 2.36] |
| pressure-imitation | 80/720 | teacher | 64 | -228.12 | [-352.34, -106.64] | -179.69 | [-354.71, -14.04] |
| pressure-imitation | 80/720 | v2 | 64 | -44.53 | [-200.00, 98.44] | -37.50 | [-212.52, 133.59] |
| pressure-imitation | 200/600 | random | 64 | 169.53 | [-257.03, 585.21] | 73.44 | [-478.12, 617.99] |
| pressure-imitation | 200/600 | passive | 64 | 35.94 | [-21.88, 89.06] | 35.16 | [-156.27, 231.25] |
| pressure-imitation | 200/600 | aggressive | 64 | -218.75 | [-622.66, 149.24] | -167.97 | [-543.79, 188.28] |
| pressure-imitation | 200/600 | equity | 64 | -464.06 | [-703.91, -237.07] | 417.19 | [21.84, 796.89] |
| pressure-imitation | 200/600 | teacher | 64 | -570.70 | [-930.86, -201.52] | -173.44 | [-623.46, 303.14] |
| pressure-imitation | 200/600 | v2 | 64 | -199.22 | [-627.38, 217.21] | -105.47 | [-560.20, 334.38] |

ppo: complete panel = False; evaluation compute = 600.3 seconds.

imitation: complete panel = True; evaluation compute = 432.0 seconds.

pressure: complete panel = True; evaluation compute = 514.5 seconds.

pressure-imitation: complete panel = True; evaluation compute = 445.9 seconds.

## PPO versus matching pressure imitation

Same training seed, imitation steps, data, and heldout deals. Positive values favor PPO. These exploratory intervals do not correct for multiple testing.

| Stacks | Opponent | Matched pairs | PPO gain BB/100 | 95% CI |
| --- | --- | ---: | ---: | --- |
| 400/400 | random | 64 | 1025.00 | [-48.44, 2130.59] |
| 400/400 | passive | 64 | -153.91 | [-621.88, 296.09] |
| 400/400 | aggressive | 64 | 353.91 | [-356.31, 1074.32] |
| 400/400 | equity | 64 | -1524.22 | [-2101.56, -920.31] |
| 400/400 | teacher | 64 | 260.94 | [-653.14, 1208.61] |
| 400/400 | v2 | 64 | 42.97 | [-772.71, 937.50] |
| 80/720 | random | 64 | 151.56 | [-62.52, 364.06] |
| 80/720 | passive | 64 | -39.06 | [-128.91, 53.12] |
| 80/720 | aggressive | 64 | 67.19 | [-71.88, 214.08] |
| 80/720 | equity | 64 | 90.62 | [-43.77, 228.12] |
| 80/720 | teacher | 64 | 183.59 | [27.34, 355.47] |
| 80/720 | v2 | 64 | 2.34 | [-154.69, 162.50] |
| 200/600 | random | 64 | 133.59 | [-411.80, 706.27] |
| 200/600 | passive | 64 | -94.53 | [-330.47, 135.16] |
| 200/600 | aggressive | 64 | 172.66 | [-183.59, 550.80] |
| 200/600 | equity | 64 | -707.03 | [-1057.83, -343.75] |
| 200/600 | teacher | 64 | 281.25 | [-199.24, 775.00] |
| 200/600 | v2 | 64 | 254.69 | [-202.34, 730.49] |

## Retrained architecture controls

One training seed, about 81 seconds training per mode, identical data-sequence prefixes. Different step counts reflect throughput under the same wall budget. These limited controls do not establish biological necessity.

| Mode | Updates | Training seconds | Validation loss |
| --- | ---: | ---: | ---: |
| real | 1933 | 81.01 | 1.05 |
| shuffle | 1971 | 81.03 | 0.92 |
| zero | 10854 | 81.01 | 0.89 |
| mlp | 30173 | 81.00 | 0.84 |

Paired FlyWire gain over each control, equally averaging equity/random/v2 opponents within each heldout deal before bootstrapping. Sample size 32 deals per stack profile. These exploratory comparisons are not corrected for multiple testing.

| Control | Stacks | Real graph gain BB/100 | 95% CI |
| --- | --- | ---: | --- |
| shuffle | 400/400 | 369.79 | [-150.01, 871.91] |
| shuffle | 80/720 | -23.18 | [-145.06, 94.27] |
| shuffle | 200/600 | 205.21 | [-26.58, 445.86] |
| zero | 400/400 | 646.35 | [-223.70, 1487.54] |
| zero | 80/720 | -3.39 | [-184.64, 177.11] |
| zero | 200/600 | 210.42 | [-209.90, 610.44] |
| mlp | 400/400 | 241.15 | [-121.37, 646.38] |
| mlp | 80/720 | -110.16 | [-250.52, 22.14] |
| mlp | 200/600 | 2.60 | [-224.48, 222.93] |

Compute estimate including capped pending/failed jobs: $1.41. Startup, storage and billing rounding are excluded. This is not an account invoice.

[Training curves](../experiments/round2-learning.png). [Methods and limitations](poker-strength-audit.md).
