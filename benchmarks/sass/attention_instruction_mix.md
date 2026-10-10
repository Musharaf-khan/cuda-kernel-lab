| kernel | scope | FFMA | HMMA | LDS.32 | LDS.64 | LDS.128 | LDSM | STS | LDG | BAR | LOCAL(spill) | total | FFMA per shared load |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flash_attention_split<64,128,32,2> | whole | 2058 | 0 | 0 | 0 | 512 | 0 | 62 | 70 | 2 | 0 | 3440 | 4.0 |
| flash_attention_split<64,128,32,2> | hot loop | 2049 | 0 | 0 | 0 | 512 | 0 | 62 | 62 | 2 | 0 | 3220 | 4.0 |
| flash_attention_split<64,64,64,4> | whole | 2058 | 0 | 0 | 0 | 512 | 0 | 62 | 66 | 2 | 0 | 3872 | 4.0 |
| flash_attention_split<64,64,64,4> | hot loop | 2049 | 0 | 0 | 0 | 512 | 0 | 62 | 62 | 2 | 0 | 3691 | 4.0 |
| flash_attention_split<64,64,32,4> | whole | 1034 | 0 | 0 | 0 | 256 | 0 | 62 | 66 | 2 | 0 | 2160 | 4.0 |
| flash_attention_split<64,64,32,4> | hot loop | 1025 | 0 | 0 | 0 | 256 | 0 | 62 | 62 | 2 | 0 | 1983 | 4.0 |
| flash_attention_split<64,64,32,2> | whole | 2058 | 0 | 0 | 0 | 512 | 0 | 62 | 70 | 2 | 0 | 3440 | 4.0 |
| flash_attention_split<64,64,32,2> | hot loop | 2049 | 0 | 0 | 0 | 512 | 0 | 62 | 62 | 2 | 0 | 3220 | 4.0 |
| flash_attention_fwd<64,64,32> | whole | 4106 | 0 | 0 | 0 | 1024 | 0 | 62 | 78 | 2 | 0 | 6040 | 4.0 |
| flash_attention_fwd<64,64,32> | hot loop | 4097 | 0 | 0 | 0 | 1024 | 0 | 62 | 62 | 2 | 0 | 5737 | 4.0 |
| softmax_rows_inplace | whole | 20 | 0 | 4 | 0 | 0 | 0 | 4 | 2 | 3 | 0 | 488 | 5.0 |
| softmax_rows_inplace | hot loop | 7 | 0 | 2 | 0 | 0 | 0 | 2 | 1 | 2 | 0 | 173 | 3.5 |
