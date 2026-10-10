| kernel | scope | FFMA | HMMA | LDS.32 | LDS.64 | LDS.128 | LDSM | STS | LDG | BAR | LOCAL(spill) | total | FFMA per shared load |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgemm_wmma_tiled | whole | 0 | 64 | 24 | 0 | 0 | 0 | 8 | 8 | 2 | 0 | 400 | 2.7 |
| hgemm_wmma_tiled | hot loop | 0 | 64 | 24 | 0 | 0 | 0 | 4 | 4 | 1 | 0 | 165 | 2.7 |
| hgemm_wmma_naive | whole | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 248 | - |
| hgemm_wmma_naive | hot loop | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 32 | 0 | 0 | 120 | - |
