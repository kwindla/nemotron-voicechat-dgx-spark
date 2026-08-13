# Step 1 retained generation-frame evidence — 2026-08-12

This report is an analyzer-rendered view of `step1-retained-evidence-20260812.json`. All latency aggregates use nearest-rank percentiles. Population membership hashes cover ordered row identifiers; the gate table pins every retained source byte-for-byte.

The corpus contains 14 unique retained artifacts. The three schema anchors are the sequential preflight, aged sustained stream, and pair-era trace; the pair trace appears only once even though the task names it both as an anchor and as the pair comparator. No model, container, WebSocket, or browser run was used.

## Corpus inventory

For every population below, `n / mean / p95 / >80` uses the server clock `server_step_ms`; `mhash` is that population's full membership SHA-256.

### `nano-sequential-preflight-20260810-r1`

Source SHA-256: `166ec35a5e74045384c80042e01fe20e4795137719797634145dde6f6d9a74fd`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 34 | 86.168 | 86.636 | 34 | `05295ce9e58ce7bd1685fb9a1a3f0f0e04f9f01d5ac923e0b6fdb13e6a64a155` |
| delivered-nonBOS | 33 | 84.024 | 85.960 | 33 | `a8ec3a705fe4bd0bae0d9f8109a8be4e786525172eb5acf355ece67dba90aeda` |
| BOS-transition | 1 | 156.923 | 156.923 | 1 | `79b267c5aebc0b70a05ff2d8a71aa783b16e3cd6ec9eb5ff80aa3717f923670a` |
| abort-prefill-reset | 1 | 156.923 | 156.923 | 1 | `79b267c5aebc0b70a05ff2d8a71aa783b16e3cd6ec9eb5ff80aa3717f923670a` |
| response-null | 2692 | 70.926 | 73.781 | 24 | `3a5e63cffe022bf56d2d1444a3350a5f5e869e1d3682a4888f8ac2c987c49221` |
| idle | 2692 | 70.926 | 73.781 | 24 | `3a5e63cffe022bf56d2d1444a3350a5f5e869e1d3682a4888f8ac2c987c49221` |
| function-cycle | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `yes`; outcome `REJECT`; worst K-window 1009.934 ms; max gap 151.671 ms. Delivered mhash `05295ce9e58ce7bd1685fb9a1a3f0f0e04f9f01d5ac923e0b6fdb13e6a64a155`.
Queue diagnostics: clock `client_received_monotonic_s (response.output_audio.delta)`; normative `yes`; responses 1; zero-reserve max prefix debt 293.889 ms; at reserve 160 ms, responses starved / total starvation 1 / 142.218 ms. Queue/delivered membership hash `92ac805195825e91d588feff7282b596df6e4dea57da67b9c1048160d2cd0dd0`.

### `sustained-aged-26h-20260812`

Source SHA-256: `acb205f6b6e5c786a039a2e3e238f12a15dc0fc0dcc971341dd8838b33f7353f`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 193 | 85.403 | 100.005 | 144 | `3f832f89b89e367c590e23cacda854b963a9f78a1960f615c85fc7db977ddb5e` |
| delivered-nonBOS | 188 | 82.919 | 93.109 | 139 | `e4afedeb26f6bcd04194100475f4a53c733e50400d4d11ea9fdbcbd5529930fb` |
| BOS-transition | 5 | 178.801 | 186.708 | 5 | `d37fb73dc53b8d38c8009a49bcf7361c03a1c12ee80f1a1dcb80311fd35f09e4` |
| abort-prefill-reset | 5 | 178.801 | 186.708 | 5 | `d37fb73dc53b8d38c8009a49bcf7361c03a1c12ee80f1a1dcb80311fd35f09e4` |
| response-null | 2454 | 70.627 | 74.827 | 31 | `fbe75b398c42f6252b08dba96784724329d3a9867869077f0ed1437e0fb4f366` |
| idle | 2424 | 71.469 | 74.830 | 31 | `98f712d2c6b8e4da32454a687ed2713a637a214f12378d2a83b67e9af8cf78ef` |
| function-cycle | 30 | 2.555 | 5.038 | 0 | `02b104caf316b636297f7a33ceff4d1253ff2c3ddada53495264aabd4f84dd9a` |
| EarTTS-high | 10 | 91.763 | 96.453 | 10 | `b4ed44bee9233860207cb456740877cc8e04304dfcc7d1bccc4acdb46d1690d9` |
| residual-high | 2 | 96.431 | 100.005 | 2 | `7e745eb2e3f64c5576e964a7f965c209e840716a87f0a73e73273c90602e0858` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `yes`; outcome `REJECT`; worst K-window 1023.500 ms; max gap 147.157 ms. Delivered mhash `3f832f89b89e367c590e23cacda854b963a9f78a1960f615c85fc7db977ddb5e`.
Queue diagnostics: clock `client_received_monotonic_s (response.output_audio.delta)`; normative `yes`; responses 5; zero-reserve max prefix debt 352.661 ms; at reserve 160 ms, responses starved / total starvation 3 / 654.966 ms. Queue/delivered membership hash `64e962fa17d878e606ba00376bf24ab6029e444704c612c11d92423c249a5bc4`.

### `nano-sequential-production-r1`

Source SHA-256: `3d3157480b009d107109594953b07a32231150654e7bb2f1d85d072b8a4bef3b`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 34 | 85.048 | 88.071 | 33 | `e215f755ef38dd25e3a905c5ffc27db9b57f1d30530affa44dddbbfb40eeefbb` |
| delivered-nonBOS | 33 | 82.754 | 86.676 | 32 | `328da1b49e8b85748a235f44a8c0813fd326c3058aa4bd563ed03ba70d10d8fa` |
| BOS-transition | 1 | 160.766 | 160.766 | 1 | `fa1c5b56b362429c3f8fe31159f534c47dc57165df5ac1b5a8bcfd3a1f9a3f49` |
| abort-prefill-reset | 1 | 160.766 | 160.766 | 1 | `fa1c5b56b362429c3f8fe31159f534c47dc57165df5ac1b5a8bcfd3a1f9a3f49` |
| response-null | 1190 | 68.724 | 72.225 | 2 | `615ff68ad6e6798976666f009ae12fa35993930b6b89edf3954e7af1eb2346a2` |
| idle | 1190 | 68.724 | 72.225 | 2 | `615ff68ad6e6798976666f009ae12fa35993930b6b89edf3954e7af1eb2346a2` |
| function-cycle | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `yes`; outcome `REJECT`; worst K-window 1009.394 ms; max gap 164.430 ms. Delivered mhash `e215f755ef38dd25e3a905c5ffc27db9b57f1d30530affa44dddbbfb40eeefbb`.
Queue diagnostics: clock `client_received_monotonic_s (response.output_audio.delta)`; normative `yes`; responses 1; zero-reserve max prefix debt 259.706 ms; at reserve 160 ms, responses starved / total starvation 1 / 95.276 ms. Queue/delivered membership hash `8b1eda434717e2a386c1981e2d85304c1bd771845e2ea4ec9fb3429a59c5299d`.

### `209fe177-bbb8-403d-ba50-740d3b6f3bf4`

Source SHA-256: `baabf0a1203e8fa4bdcef9ebcbd9b2aeb7db952ec0eb83c58e18898fdcb2f880`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 55 | 84.763 | 94.383 | 52 | `cdd18c4b7a3ca2c2e5f50088cb80447c25ef1bf6532cde595a6040968fe95703` |
| delivered-nonBOS | 54 | 84.519 | 94.374 | 51 | `7e74a11703d61842fe040b8da97b646a2c834338eba8ef6d2665192f37a2d73a` |
| BOS-transition | 1 | 97.988 | 97.988 | 1 | `c3082c9c6b73f43a6020a351c6fd8cc192cbd7403a71b8ad41b01eeb1c382d7a` |
| abort-prefill-reset | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| response-null | 245 | 74.697 | 91.740 | 52 | `d0ce88e18ddefcb6d7e7b4586a094d8a276b2e375c6c70327387ee0edb8416b1` |
| idle | 190 | 71.784 | 73.553 | 0 | `61460daa009e30dab13715337150be369b2105f9febbb3a5aec7b3fba4b1ae50` |
| function-cycle | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 10 | 93.369 | 94.752 | 10 | `d3bc56bd3fc35cd3e0f1638027836ba437de7de322823e01cb4c0532e6ae0a0e` |
| residual-high | 2 | 93.151 | 93.417 | 2 | `a933691f26c8366faf4e4e1f68fe10f4eba22bc218c81688a31858f34feb2680` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic REJECT)`; worst K-window 995.004 ms; max gap 97.390 ms. Delivered mhash `cdd18c4b7a3ca2c2e5f50088cb80447c25ef1bf6532cde595a6040968fe95703`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 395.647 ms; at reserve 160 ms, responses starved / total starvation 1 / 310.705 ms. Queue/delivered membership hash `cdd18c4b7a3ca2c2e5f50088cb80447c25ef1bf6532cde595a6040968fe95703`.

### `8d4d4699-734c-4327-9848-1477e04a054a`

Source SHA-256: `6ce13165a72e54cb820db11fcba3adaec9ec1a731c597f138d3c658ee7fac331`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 37 | 84.126 | 94.225 | 31 | `6dc2502b8995d9c556d5d8993be4646b166f440c5e2e226bfcdf8e54e9e38875` |
| delivered-nonBOS | 36 | 83.957 | 94.225 | 30 | `6ffae9d748d6c975862483934a2443d2a82d1c5e202613d7ae4f029d36b828d6` |
| BOS-transition | 1 | 90.215 | 90.215 | 1 | `b8b404dec1c1dd43148a2a8542e9be5cbd29dce612088bd9db5760c98e2af8b2` |
| abort-prefill-reset | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| response-null | 88 | 76.940 | 90.215 | 36 | `519afdddb6046e5e8f0eeb12f72b490652024674751e64527fccfb44da8b4a3b` |
| idle | 51 | 71.726 | 82.226 | 5 | `766cc05c2618e80e7e4d1c6a16aa406ee5880b1a336d01508c6080aff34b5615` |
| function-cycle | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 5 | 92.989 | 94.548 | 5 | `c8f3de336733b0e44ba7c49bdf7017c53eb6ffb860c2561e488af87582ad296e` |
| residual-high | 1 | 88.640 | 88.640 | 1 | `1e345bb65330d622ead476432201acf505372ec4765d72cf809dc14a5a6d4228` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic REJECT)`; worst K-window 986.830 ms; max gap 97.949 ms. Delivered mhash `6dc2502b8995d9c556d5d8993be4646b166f440c5e2e226bfcdf8e54e9e38875`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 242.799 ms; at reserve 160 ms, responses starved / total starvation 1 / 159.019 ms. Queue/delivered membership hash `6dc2502b8995d9c556d5d8993be4646b166f440c5e2e226bfcdf8e54e9e38875`.

### `a4d2f917-a906-4498-a6b2-aecf67ef5db2`

Source SHA-256: `1748621f4e4d9e14bfa129d4a3154111a8798b0681a7d0e9224f884112d3d12c`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| delivered-nonBOS | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| BOS-transition | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| abort-prefill-reset | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| response-null | 546 | 68.349 | 71.386 | 1 | `b271c7ace7f3554d1001674fb1e1476d9c882bdd86560b8a604a67c6274b06bf` |
| idle | 546 | 68.349 | 71.386 | 1 | `b271c7ace7f3554d1001674fb1e1476d9c882bdd86560b8a604a67c6274b06bf` |
| function-cycle | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE`; worst K-window — ms; max gap — ms. Delivered mhash `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 0; zero-reserve max prefix debt 0.000 ms; at reserve 160 ms, responses starved / total starvation 0 / 0 ms. Queue/delivered membership hash `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.

### `2a0fa2ae-766d-4455-bfb0-885f67cc958a`

Source SHA-256: `2815a4e99d5ade234ade161cc83acd9e4b7fa93a4297bd55fb5e3f5ae596693d`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 134 | 84.109 | 88.588 | 101 | `be5ec3122c61c8602018746724c9a17ba7c530cf0277fb98f20fff5a4fb25cf7` |
| delivered-nonBOS | 130 | 82.010 | 86.482 | 98 | `de383a589d9b13b237677b042db460ca31e3c300b53c07dbfbc18ee6f6a06eb3` |
| BOS-transition | 4 | 152.351 | 205.267 | 3 | `c598fc6d6439a52109612cbe8712ad2bcb8ce09a7ae5ffc0f810f1cf39d183d4` |
| abort-prefill-reset | 3 | 177.141 | 205.267 | 3 | `209cba67016bc59f950585878e82f82a633d6a0ce7a6bb4c12155bfd2d1252b2` |
| response-null | 490 | 65.812 | 85.430 | 116 | `f8f71404cd105d3ab75d0351e159a9fa6987ceef1cdafa60ae2eeafe2fc06b4d` |
| idle | 326 | 64.280 | 78.462 | 15 | `3cb05f368c57a075363742226382fb3c0867dfbdc9e07720aa4ea120a02fa888` |
| function-cycle | 30 | 0.730 | 1.410 | 0 | `8c9142aadcc4881c2543aad4e222e8b7691d5a40ff9a6b333629b429cd216c57` |
| EarTTS-high | 1 | 91.672 | 91.672 | 1 | `7ee4fb4bc6398449bf29afe09fb6f879b7375ca4582fd7861d1a82cbd1ff230c` |
| residual-high | 1 | 90.691 | 90.691 | 1 | `b7908f96bb0eedfab203e54df198c6786fdd0352e75247fdbd66c6e72560cc45` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic REJECT)`; worst K-window 974.284 ms; max gap 106.526 ms. Delivered mhash `be5ec3122c61c8602018746724c9a17ba7c530cf0277fb98f20fff5a4fb25cf7`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 4; zero-reserve max prefix debt 196.500 ms; at reserve 160 ms, responses starved / total starvation 4 / 255.558 ms. Queue/delivered membership hash `be5ec3122c61c8602018746724c9a17ba7c530cf0277fb98f20fff5a4fb25cf7`.

### `ac88123a-28b6-48e9-bd2d-23dc101c48dc`

Source SHA-256: `442df3de9adec6773debbcc23fe645b51405921f296ffc2860dcd0a41b635509`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 106 | 85.475 | 93.409 | 77 | `e0efc636151a85a28a5473565523a8d1e41edae359b75cfccf1464943a5efcd0` |
| delivered-nonBOS | 102 | 82.843 | 91.611 | 73 | `37f554c28efc7620a614d5615c737f12966156822ad02c4766a25ace4acada0f` |
| BOS-transition | 4 | 152.584 | 201.490 | 4 | `e62d9e241751f882275137e8c0b50377065916b7a08fb50116d3aa4369c9b8e9` |
| abort-prefill-reset | 3 | 174.966 | 201.490 | 3 | `7ba6a698b460c1642904e35d44140d3e982985f34b475f7bf008b71b5649f5bb` |
| response-null | 483 | 66.783 | 88.463 | 120 | `0b96e17bce5111873e51c2950f72fcb40c767cfe1447794be48c2c00d570ccee` |
| idle | 348 | 66.569 | 82.997 | 43 | `3f68dc124066fe48fad3f47b6c0ec514e1a2bb39ba8527025e4df6eb48038c10` |
| function-cycle | 29 | 1.041 | 1.090 | 0 | `d1f3585d9c09abcbcbb2f940825f0aea5f3ce1bb80e99954fae3c56a32e636ad` |
| EarTTS-high | 12 | 91.167 | 98.371 | 12 | `56fe510fbb1e5865c9b7578464bd68854c2a12e81d987f579da80b1bd41a4a14` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic REJECT)`; worst K-window 997.352 ms; max gap 101.421 ms. Delivered mhash `e0efc636151a85a28a5473565523a8d1e41edae359b75cfccf1464943a5efcd0`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 4; zero-reserve max prefix debt 297.574 ms; at reserve 160 ms, responses starved / total starvation 3 / 264.269 ms. Queue/delivered membership hash `e0efc636151a85a28a5473565523a8d1e41edae359b75cfccf1464943a5efcd0`.

### `f110f063-8c9a-46e1-87a9-97dc682b8022`

Source SHA-256: `aba32f50ad8c7c03fce4c3c1663ea0d42da119d4a9a52c1d90379bb4596ea1f3`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 32 | 75.116 | 118.710 | 13 | `95ea829c87154c81b39e1154e6e67e02f22f39cb5ae1748c3746065eb4a67890` |
| delivered-nonBOS | 31 | 72.699 | 117.917 | 12 | `77826b8eff30ecbff4e4759ff7d757a8b82138929e83411db00ce8da45641399` |
| BOS-transition | 1 | 150.050 | 150.050 | 1 | `0c84ab025bd2eee7753107eaba8ae0f91bc1a69217a826632bf89c2ab4539650` |
| abort-prefill-reset | 1 | 150.050 | 150.050 | 1 | `0c84ab025bd2eee7753107eaba8ae0f91bc1a69217a826632bf89c2ab4539650` |
| response-null | 2718 | 68.239 | 84.773 | 356 | `ae69f0123105d9d66dc7ded09fdf65205b0661e52bf9ce194306f5e78ab313c8` |
| idle | 2686 | 68.157 | 84.546 | 343 | `21dbff3b34655660faa6306a09fe3698b27135ea17d3411320cce40f9bca4091` |
| function-cycle | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic ACCEPT)`; worst K-window 877.871 ms; max gap 120.939 ms. Delivered mhash `95ea829c87154c81b39e1154e6e67e02f22f39cb5ae1748c3746065eb4a67890`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 4.530 ms; at reserve 160 ms, responses starved / total starvation 0 / 0.000 ms. Queue/delivered membership hash `95ea829c87154c81b39e1154e6e67e02f22f39cb5ae1748c3746065eb4a67890`.

### `de64cc34-2394-490e-a03c-cd920f96af25`

Source SHA-256: `b04a401967d0e3ebc51d1f95abb294a2225c849a61cde81be9e54b6e1552b4ec`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 46 | 84.253 | 85.714 | 35 | `99a2077318f07ee16410ae2569e3a0caa722494fcae3cd567f9cf54241b1302a` |
| delivered-nonBOS | 45 | 82.577 | 85.457 | 34 | `f33069beb60095691606c7d3cd555d7b04a432a543e81007478bbe78626bd925` |
| BOS-transition | 1 | 159.677 | 159.677 | 1 | `4063274a1e94f2d27105dee9e9adebf0a61e8820917476919994116611d9e909` |
| abort-prefill-reset | 1 | 159.677 | 159.677 | 1 | `4063274a1e94f2d27105dee9e9adebf0a61e8820917476919994116611d9e909` |
| response-null | 405 | 48.287 | 84.308 | 36 | `4467df84af26295d3c69deb0150a9edb43741e795bf1b1b7eb271bcbb9127b68` |
| idle | 282 | 55.391 | 74.775 | 1 | `5fcb83cf1c5c7960c0709e55b82e81763fe6342895fc5b5544ec1ae2f98845eb` |
| function-cycle | 77 | 0.784 | 1.277 | 0 | `d54e4d05f5a91d0815db6625f4aebf89da049c886733d3bc17f76f18633274d9` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic REJECT)`; worst K-window 965.728 ms; max gap 89.432 ms. Delivered mhash `99a2077318f07ee16410ae2569e3a0caa722494fcae3cd567f9cf54241b1302a`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 261.242 ms; at reserve 160 ms, responses starved / total starvation 1 / 179.153 ms. Queue/delivered membership hash `99a2077318f07ee16410ae2569e3a0caa722494fcae3cd567f9cf54241b1302a`.

### `feca4ec0-2962-48df-b219-e175858659c8`

Source SHA-256: `7ea447eb15e28a87bd49ab47f9c8b7eedae16561d853a19e074fc578a91ef44a`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 46 | 84.991 | 85.784 | 43 | `48680fea86fc46e05dcc70e95b62ea42d5df0ce89b6978e0c6740fe4ad1ac1a7` |
| delivered-nonBOS | 45 | 83.245 | 85.729 | 42 | `c6782d8f579ae8de26bafd0f1d3027bf6a2c9cbd0f1e0e2dbd0d689dddbe5ed8` |
| BOS-transition | 1 | 163.551 | 163.551 | 1 | `578f8c7fe771d9339336bce9ee89d8a94263638fa81671d03dccd24973737f97` |
| abort-prefill-reset | 1 | 163.551 | 163.551 | 1 | `578f8c7fe771d9339336bce9ee89d8a94263638fa81671d03dccd24973737f97` |
| response-null | 399 | 48.936 | 84.269 | 45 | `0f72ce9ba3fb601d69304477845500808888eb9a35c89c59a298f95d48c59a5e` |
| idle | 275 | 56.564 | 74.853 | 2 | `9ff3f59ac0d18f25745293125b428ff83d354e5e08be68e0f48e5675590d28f4` |
| function-cycle | 78 | 0.780 | 1.104 | 0 | `e91b6918f021cfe0d34141aeb98162340ec8215ca03328eb7d1158fbb5737694` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic REJECT)`; worst K-window 963.890 ms; max gap 88.960 ms. Delivered mhash `48680fea86fc46e05dcc70e95b62ea42d5df0ce89b6978e0c6740fe4ad1ac1a7`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 298.580 ms; at reserve 160 ms, responses starved / total starvation 1 / 210.689 ms. Queue/delivered membership hash `48680fea86fc46e05dcc70e95b62ea42d5df0ce89b6978e0c6740fe4ad1ac1a7`.

### `97f9bacc-4d2f-4e55-9026-dc2590919023`

Source SHA-256: `39387ec141bd0199a205c80c883192c08b44565a9548b5f7f91117c457ce7211`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 45 | 84.848 | 86.295 | 32 | `24f4f16b563560e9044ba333f4958cd23bd671c0651e62581d15d8e061f667dd` |
| delivered-nonBOS | 44 | 82.005 | 85.652 | 31 | `39b4ff2f178a3cfbefb457b7eb42bf0d82513e4c0913669ce388e24fc4a595fd` |
| BOS-transition | 1 | 209.912 | 209.912 | 1 | `9eec0781ac770f91107f4e404bfae91a056651175bb14d622947961df9fe0c81` |
| abort-prefill-reset | 1 | 209.912 | 209.912 | 1 | `9eec0781ac770f91107f4e404bfae91a056651175bb14d622947961df9fe0c81` |
| response-null | 398 | 47.712 | 84.189 | 34 | `1d6a99a027042e3727c68565821b834ba9a5821feca4d3ddd7d1db3363cdbdbb` |
| idle | 275 | 55.004 | 73.313 | 2 | `31294f4aada4599ba8c4fee364c4b878aef0b1c4faada5b609828a9abf7d1095` |
| function-cycle | 78 | 0.580 | 0.964 | 0 | `edf8ec6d9c279c80d5386b11292167bffcaef30a1113c73ade2599517e67cc98` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic ACCEPT)`; worst K-window 959.650 ms; max gap 90.570 ms. Delivered mhash `24f4f16b563560e9044ba333f4958cd23bd671c0651e62581d15d8e061f667dd`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 234.030 ms; at reserve 160 ms, responses starved / total starvation 1 / 151.279 ms. Queue/delivered membership hash `24f4f16b563560e9044ba333f4958cd23bd671c0651e62581d15d8e061f667dd`.

### `543bfc97-d9d1-40ae-84de-1a26b78a3756`

Source SHA-256: `f6247366c092557c501c8177e7b8cefaf48eee99ffd77e5f47593efa7edcdf10`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 46 | 83.350 | 85.425 | 40 | `b35898f34b8ffaafd6615c9bfb179dc9b76c5d1f0e4e97f4310306cc58c53076` |
| delivered-nonBOS | 45 | 81.591 | 85.021 | 39 | `462596309e9ecdc1b4bdc8879d9482b86eee133afc5174830eb85818a892c289` |
| BOS-transition | 1 | 162.512 | 162.512 | 1 | `2b988779d2d67f8fed06d4d975e551f80a0dbe86cc5c66ba1b14e0fe76b377eb` |
| abort-prefill-reset | 1 | 162.512 | 162.512 | 1 | `2b988779d2d67f8fed06d4d975e551f80a0dbe86cc5c66ba1b14e0fe76b377eb` |
| response-null | 407 | 46.784 | 83.194 | 42 | `0ec1893ff6cab907c9d58e4e6cfcc19322669b0593d1b8a385a823fef46b29ff` |
| idle | 282 | 53.679 | 73.726 | 2 | `74c52b26bb0825b84a7b1d90296fccef7cd7faff5717df66e6c665616ee62b56` |
| function-cycle | 79 | 0.879 | 1.182 | 0 | `bcf4ca14c34bb1c1a5c0230aa61d88bb4fc4c20add66d298d40b5f74c16b628f` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic ACCEPT)`; worst K-window 950.151 ms; max gap 89.002 ms. Delivered mhash `b35898f34b8ffaafd6615c9bfb179dc9b76c5d1f0e4e97f4310306cc58c53076`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 200.605 ms; at reserve 160 ms, responses starved / total starvation 1 / 113.337 ms. Queue/delivered membership hash `b35898f34b8ffaafd6615c9bfb179dc9b76c5d1f0e4e97f4310306cc58c53076`.

### `70d3f7b5-2d3c-4998-846d-c85c539f2c59`

Source SHA-256: `aac38611e28e870e70ef2b7146772b68de22855dd42c08f90a85000808fe805c`

| population | n | mean ms | p95 ms | >80 ms | mhash |
|---|---:|---:|---:|---:|---|
| delivered | 46 | 84.096 | 85.009 | 42 | `f17062a9c85cd22556054ab516840c86b0f0a66c65c9d5add50c5144cbf37f7f` |
| delivered-nonBOS | 45 | 82.344 | 84.625 | 41 | `275db6cca98436bcc7fee7d330ad110ed6823594e69690a04fd3c5ed4306f533` |
| BOS-transition | 1 | 162.938 | 162.938 | 1 | `e9c4937660028ca9ffeb32d77cd8abb5a9bea67420123826d6dddd36beed4c94` |
| abort-prefill-reset | 1 | 162.938 | 162.938 | 1 | `e9c4937660028ca9ffeb32d77cd8abb5a9bea67420123826d6dddd36beed4c94` |
| response-null | 398 | 47.456 | 83.684 | 46 | `d270ec4a8dcc27cb56ee748638fdd46b1f81ee39c340df1f37170599508e1c7c` |
| idle | 274 | 54.588 | 72.388 | 4 | `edcb50c1a879f773259654f06c396409502cc7b15774a6a42972d6b775178109` |
| function-cycle | 78 | 0.797 | 1.041 | 0 | `05178709127b26d686d8b1778d616da83aa0acab17610e2de49a943fdbe91dd2` |
| EarTTS-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

Structural gate: clock `client_received_monotonic_s (response.output_audio.delta)`; eligible `no`; outcome `UNAVAILABLE (server diagnostic ACCEPT)`; worst K-window 951.648 ms; max gap 89.020 ms. Delivered mhash `f17062a9c85cd22556054ab516840c86b0f0a66c65c9d5add50c5144cbf37f7f`.
Queue diagnostics: clock `monotonic_s (delivered model_step fallback)`; normative `no`; responses 1; zero-reserve max prefix debt 237.993 ms; at reserve 160 ms, responses starved / total starvation 1 / 151.269 ms. Queue/delivered membership hash `f17062a9c85cd22556054ab516840c86b0f0a66c65c9d5add50c5144cbf37f7f`.

## Corpus-scale stage-timing strata

Token phase is source-derived: PAD-tail means `agent_token_id == 12`; agent-control means `agent_control` is present or the token matches a non-PAD `control_ids` entry; text-emission means a remaining integer token. BOS remains separate. High modes retain the plan predicates over delivered-nonBOS only. Server columns use `server_step_ms`; stage columns are host-interval means in milliseconds.

| stratum | n | server mean | server p95 | >80 | perception | Nano | EarTTS | codec | residual | wrapper | mhash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| delivered-nonBOS-stage-timed | 831 | 82.420 | 90.990 | 655 | 12.642 | 51.989 | 11.979 | 0.846 | 3.982 | 81.437 | `ecf20a5e761959d0eb28a566e35a89da58e22cffdbe324e8f21351378cb9240b` |
| text-emission | 209 | 82.364 | 90.569 | 156 | 12.616 | 52.560 | 11.805 | 0.617 | 3.807 | 81.406 | `1dbf0780015d9cfad593e1942cc911d43301af6ee5aefcd1cf9fb848bddb0a09` |
| PAD-tail | 615 | 82.440 | 91.611 | 492 | 12.652 | 51.789 | 12.041 | 0.927 | 4.039 | 81.448 | `a0096df73387d278b192740dd57d4d5bfb66260bbc9915d9c3afd443de0e9955` |
| agent-control | 7 | 82.322 | 84.829 | 7 | 12.463 | 52.511 | 11.723 | 0.603 | 4.138 | 81.438 | `1ab7789ee424f4e6f191ef8f37d1badaa9504a1798b3bd5180e08deb9701ff5f` |
| token-unavailable | 0 | — | — | 0 | — | — | — | — | — | — | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| EarTTS-high | 38 | 92.156 | 96.453 | 38 | 12.560 | 52.635 | 21.700 | 0.604 | 3.833 | 91.332 | `7bed86dc518efc1f8de983fb9e3af4f5aad008e43155da3ece8359d0bc09c217` |
| residual-high | 6 | 93.083 | 100.005 | 6 | 12.494 | 53.236 | 11.739 | 0.637 | 14.017 | 92.122 | `c88b2f593949d6477219f0ad34d9e0758ad658b4a1f30b71320469896713f5ed` |
| BOS-transition | 23 | 157.751 | 205.267 | 22 | 12.335 | 52.580 | 87.609 | 0.577 | 3.694 | 156.796 | `276635e5d6d37fa54041f10fc0f2457baca49eee878657a536bced5b70a5273c` |
| abort-prefill-reset | 19 | 172.456 | 209.912 | 19 | 12.378 | 52.551 | 102.261 | 0.582 | 3.672 | 171.443 | `3d52048a99e8853e66843365e15d6240e2dded68b09ead81e7249bbe8e668b55` |
| prepared-reuse-BOS | 4 | 87.904 | 97.988 | 3 | 12.130 | 52.718 | 18.017 | 0.554 | 3.800 | 87.218 | `ea460659ca3156ae7db3ae7be0c3c09a8fb814d84269e4d326f33268770bf269` |

The four delivered-nonBOS token-phase strata reconcile 831/831 stage-timed rows; BOS is reported separately.

## Pair versus sequential retained rollup

Pair artifact: `f110f063-8c9a-46e1-87a9-97dc682b8022`. Sequential side: all 13 other retained artifacts. Both sides use the identical analyzer populations and `server_step_ms` clock.

| population | pair n / mean / p95 / >80 | pair mhash | sequential n / mean / p95 / >80 | sequential mhash |
|---|---|---|---|---|
| delivered | 32 / 75.116 / 118.710 / 13 | `316fde76d6b1900544d99909bdceddb4a0bd5e0fdc9ee437a44d0f9dce91798d` | 822 / 84.812 / 93.262 / 664 | `825b64b01bfe908cda677e5a3a9e4bccbd287004ec41512e94fc8e398abe0c7e` |
| delivered-nonBOS | 31 / 72.699 / 117.917 / 12 | `ed8b6bffd91c01c620fd81b5902294cc95665a0ca36ced4c43114f20152866e2` | 800 / 82.797 / 90.126 / 643 | `2073c4017bd650d47269d0006fc79978d16bb4b2cd2630c4c9ee5fb6bb90df36` |
| BOS-transition | 1 / 150.050 / 150.050 / 1 | `f90bda395c92bbd0980c910efeafbb65a92a66de7751dbc7a9d1e83a8c97b9eb` | 22 / 158.101 / 205.267 / 21 | `80691acfd74e6752d0595da13a62a5aeead0ff98d1b19c6ad66fd32f09340b16` |
| abort-prefill-reset | 1 / 150.050 / 150.050 / 1 | `f90bda395c92bbd0980c910efeafbb65a92a66de7751dbc7a9d1e83a8c97b9eb` | 18 / 173.700 / 209.912 / 18 | `27b2551fa902c7103d2eb34473655aa0aec36719cfaccc0309e3674a19a89894` |
| response-null | 2718 / 68.239 / 84.773 / 356 | `a94336f4d14bfabaf154d59468452266f6561ebbf5e6c5473c76b2a1a3dddda0` | 10195 / 65.613 / 81.110 / 585 | `658b35c00c9d820929a864c807274297c0c4158071a637e8e2b0fb3718e3820b` |
| idle | 2686 / 68.157 / 84.546 / 343 | `a874d3472fc30a6610969393280ddeb62270345ac90bc770262bfb9dfd4c08d5` | 9155 / 67.842 / 74.409 / 132 | `7f4a15613bd72024fa951df8803fcdd31db89ec2a27285cd29f264bc95c4d0a9` |
| function-cycle | 0 / — / — / 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 479 / 0.891 / 1.747 / 0 | `33db1fd75ceed0557bfbe279dd8679c530b54ba906b01d87dff8f1900ce1422f` |
| EarTTS-high | 0 / — / — / 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 38 / 92.156 / 96.453 / 38 | `7bed86dc518efc1f8de983fb9e3af4f5aad008e43155da3ece8359d0bc09c217` |
| residual-high | 0 / — / — / 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 6 / 93.083 / 100.005 / 6 | `c88b2f593949d6477219f0ad34d9e0758ad658b4a1f30b71320469896713f5ed` |

Pair scheduling is classified only from the delta of `turn_state.vllm_request_positions.nano.session_positions`: zero is buffered, one is sequential, and two is packed-pair.

| schedule class | n | server mean | server p95 | >80 | mhash |
|---|---:|---:|---:|---:|---|
| buffered | 12 | 44.999 | 45.959 | 0 | `85ca556f435e17d5a683410bd5c5acdade57efa2dad2f9229930672b9c561cb4` |
| packed-pair | 12 | 97.485 | 118.710 | 12 | `3051a980e30ba5de7a5228b7abaa7117c005ff4736397c4f9e52176e68dbe66d` |
| sequential | 8 | 86.740 | 150.050 | 1 | `04ee166dd4aff2db42e1834c1edd367bfb2807028d3be5ae13b2ac60dc87c65c` |
| unavailable | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| other-position-delta | 0 | — | — | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

The pair-class subsequence contains 24 rows and 23 adjacent transitions: buffered→packed 12, packed→buffered 11. The ordered pair-class membership SHA-256 is `5f6e8ec6c470972c80d1f7246201e869a1d1b1aa51eb8ae135bcb674fcdab28b`; transition counts and the complete per-row schedule are in `step1_details.pair_vs_sequential.pair_schedule`.

## High-mode row inventories for Step 4

### EarTTS-high

Inventory n=38; membership SHA-256 `7bed86dc518efc1f8de983fb9e3af4f5aad008e43155da3ece8359d0bc09c217`. Phase counts: PAD-tail 34, text-emission 4. Position and periodicity diagnostics are analyzer outputs: response ordinal p50/p95 22.000/39.000; distance from response end p50/p95 17.000/38.000; within-response high-row ordinal gap p50/p95 4.000/10.000.

| artifact | frame | phase/token | response pos | server | perception | Nano | EarTTS | codec | residual | wrapper | previous → next |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sustained-aged-26h-20260812` | 59 | PAD-tail/t12 | 12/34 | 88.967 | 11.704 | 52.677 | 21.134 | 0.396 | 2.424 | 88.335 | f58 PAD-tail/t12 77.022ms → f60 PAD-tail/t12 79.150ms |
| `sustained-aged-26h-20260812` | 65 | PAD-tail/t12 | 18/34 | 88.375 | 11.730 | 52.227 | 21.069 | 0.378 | 2.387 | 87.791 | f64 PAD-tail/t12 78.421ms → f66 PAD-tail/t12 81.666ms |
| `sustained-aged-26h-20260812` | 68 | PAD-tail/t12 | 21/34 | 89.879 | 12.782 | 53.116 | 20.670 | 0.372 | 2.356 | 89.297 | f67 PAD-tail/t12 79.718ms → f69 PAD-tail/t12 79.345ms |
| `sustained-aged-26h-20260812` | 75 | PAD-tail/t12 | 28/34 | 91.232 | 11.777 | 53.233 | 22.734 | 0.394 | 2.414 | 90.553 | f74 PAD-tail/t12 80.519ms → f76 PAD-tail/t12 80.415ms |
| `sustained-aged-26h-20260812` | 607 | PAD-tail/t12 | 12/40 | 96.453 | 13.311 | 54.361 | 22.001 | 0.710 | 4.844 | 95.227 | f606 PAD-tail/t12 82.366ms → f608 PAD-tail/t12 84.607ms |
| `sustained-aged-26h-20260812` | 617 | PAD-tail/t12 | 22/40 | 92.691 | 12.775 | 51.966 | 21.151 | 0.710 | 4.864 | 91.466 | f616 PAD-tail/t12 84.782ms → f618 PAD-tail/t12 85.938ms |
| `sustained-aged-26h-20260812` | 623 | PAD-tail/t12 | 28/40 | 93.963 | 13.174 | 52.485 | 21.353 | 0.707 | 4.898 | 92.617 | f622 PAD-tail/t12 83.613ms → f624 PAD-tail/t12 84.995ms |
| `sustained-aged-26h-20260812` | 1139 | PAD-tail/t12 | 24/43 | 93.615 | 13.127 | 52.646 | 21.936 | 0.743 | 4.310 | 92.762 | f1138 PAD-tail/t12 82.137ms → f1140 PAD-tail/t12 82.400ms |
| `sustained-aged-26h-20260812` | 1779 | PAD-tail/t12 | 28/42 | 93.109 | 12.376 | 52.138 | 22.356 | 0.691 | 4.668 | 92.230 | f1778 PAD-tail/t12 83.263ms → f1780 PAD-tail/t12 83.319ms |
| `sustained-aged-26h-20260812` | 2289 | PAD-tail/t12 | 28/34 | 89.344 | 11.666 | 52.478 | 22.159 | 0.382 | 2.087 | 88.770 | f2288 PAD-tail/t12 79.404ms → f2290 PAD-tail/t12 79.313ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 26 | text-emission/t1653 | 4/55 | 93.176 | 13.591 | 51.839 | 22.273 | 0.742 | 3.878 | 92.323 | f25 text-emission/t1710 80.294ms → f27 text-emission/t10234 82.907ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 39 | text-emission/t5675 | 17/55 | 91.740 | 12.386 | 52.373 | 20.547 | 0.667 | 4.903 | 90.877 | f38 text-emission/t1046 83.132ms → f40 text-emission/t1653 82.602ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 42 | text-emission/t2534 | 20/55 | 93.804 | 12.352 | 53.786 | 20.792 | 0.767 | 5.254 | 92.950 | f41 text-emission/t1636 84.209ms → f43 text-emission/t1063 83.191ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 47 | PAD-tail/t12 | 25/55 | 93.262 | 12.303 | 53.432 | 21.448 | 0.706 | 4.561 | 92.450 | f46 PAD-tail/t12 82.669ms → f48 PAD-tail/t12 83.045ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 49 | PAD-tail/t12 | 27/55 | 93.694 | 13.161 | 52.453 | 22.013 | 0.648 | 4.584 | 92.858 | f48 PAD-tail/t12 83.045ms → f50 PAD-tail/t12 84.078ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 52 | PAD-tail/t12 | 30/55 | 92.479 | 12.374 | 52.345 | 21.506 | 0.736 | 4.621 | 91.582 | f51 PAD-tail/t12 83.482ms → f53 PAD-tail/t12 82.456ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 54 | PAD-tail/t12 | 32/55 | 92.026 | 12.349 | 52.958 | 20.756 | 0.623 | 4.550 | 91.236 | f53 PAD-tail/t12 82.456ms → f55 PAD-tail/t12 82.755ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 57 | PAD-tail/t12 | 35/55 | 94.383 | 13.140 | 51.929 | 23.021 | 0.709 | 4.745 | 93.543 | f56 PAD-tail/t12 83.137ms → f58 PAD-tail/t12 82.652ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 59 | PAD-tail/t12 | 37/55 | 94.374 | 12.345 | 53.285 | 22.712 | 0.662 | 4.520 | 93.524 | f58 PAD-tail/t12 82.652ms → f60 PAD-tail/t12 81.865ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 64 | PAD-tail/t12 | 42/55 | 94.752 | 14.208 | 53.012 | 21.450 | 0.682 | 4.580 | 93.932 | f63 PAD-tail/t12 84.298ms → f65 PAD-tail/t12 82.918ms |
| `8d4d4699-734c-4327-9848-1477e04a054a` | 25 | text-emission/t1710 | 5/37 | 89.569 | 11.992 | 52.376 | 22.595 | 0.405 | 1.731 | 89.100 | f24 text-emission/t3075 77.857ms → f26 text-emission/t1362 79.571ms |
| `8d4d4699-734c-4327-9848-1477e04a054a` | 32 | PAD-tail/t12 | 12/37 | 93.559 | 12.770 | 52.512 | 21.824 | 0.732 | 4.821 | 92.659 | f31 PAD-tail/t12 82.490ms → f33 PAD-tail/t12 82.891ms |
| `8d4d4699-734c-4327-9848-1477e04a054a` | 40 | PAD-tail/t12 | 20/37 | 93.042 | 12.343 | 51.950 | 22.546 | 0.713 | 4.646 | 92.198 | f39 PAD-tail/t12 82.876ms → f41 PAD-tail/t12 83.984ms |
| `8d4d4699-734c-4327-9848-1477e04a054a` | 42 | PAD-tail/t12 | 22/37 | 94.225 | 12.319 | 53.734 | 21.860 | 0.741 | 4.720 | 93.374 | f41 PAD-tail/t12 83.984ms → f43 PAD-tail/t12 82.633ms |
| `8d4d4699-734c-4327-9848-1477e04a054a` | 48 | PAD-tail/t12 | 28/37 | 94.548 | 12.872 | 53.312 | 22.094 | 0.716 | 4.650 | 93.643 | f47 PAD-tail/t12 83.163ms → f49 PAD-tail/t12 84.431ms |
| `2a0fa2ae-766d-4455-bfb0-885f67cc958a` | 152 | PAD-tail/t12 | 39/39 | 91.672 | 12.457 | 51.563 | 21.878 | 0.626 | 4.333 | 90.857 | f151 PAD-tail/t12 80.427ms → f153 agent-control/t2 81.971ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 122 | PAD-tail/t12 | 10/38 | 95.630 | 13.132 | 54.289 | 22.352 | 0.743 | 3.879 | 94.395 | f121 PAD-tail/t12 86.512ms → f123 PAD-tail/t12 84.518ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 125 | PAD-tail/t12 | 13/38 | 93.409 | 13.630 | 51.913 | 21.099 | 0.682 | 4.852 | 92.176 | f124 PAD-tail/t12 85.479ms → f126 PAD-tail/t12 83.784ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 129 | PAD-tail/t12 | 17/38 | 88.463 | 11.987 | 51.704 | 20.864 | 0.483 | 2.885 | 87.923 | f128 PAD-tail/t12 79.228ms → f130 PAD-tail/t12 78.586ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 133 | PAD-tail/t12 | 21/38 | 90.051 | 12.019 | 52.203 | 22.094 | 0.421 | 2.645 | 89.383 | f132 PAD-tail/t12 85.828ms → f134 PAD-tail/t12 82.627ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 143 | PAD-tail/t12 | 31/38 | 98.371 | 12.636 | 57.135 | 22.374 | 0.684 | 4.654 | 97.483 | f142 PAD-tail/t12 81.821ms → f144 PAD-tail/t12 80.760ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 150 | PAD-tail/t12 | 38/38 | 90.978 | 12.539 | 51.977 | 20.777 | 0.626 | 4.234 | 90.154 | f149 PAD-tail/t12 80.031ms → f151 agent-control/t2 82.841ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 310 | PAD-tail/t12 | 20/24 | 91.611 | 12.477 | 51.177 | 22.253 | 0.625 | 4.152 | 90.684 | f309 PAD-tail/t12 80.306ms → f311 PAD-tail/t12 82.007ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 408 | PAD-tail/t12 | 17/34 | 88.448 | 12.148 | 51.657 | 20.960 | 0.438 | 2.618 | 87.820 | f407 PAD-tail/t12 79.226ms → f409 PAD-tail/t12 78.986ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 413 | PAD-tail/t12 | 22/34 | 89.257 | 12.058 | 51.951 | 21.754 | 0.399 | 2.488 | 88.650 | f412 PAD-tail/t12 77.713ms → f414 PAD-tail/t12 78.201ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 415 | PAD-tail/t12 | 24/34 | 88.765 | 11.933 | 51.337 | 21.995 | 0.407 | 2.487 | 88.159 | f414 PAD-tail/t12 78.201ms → f416 PAD-tail/t12 80.610ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 418 | PAD-tail/t12 | 27/34 | 90.126 | 12.733 | 52.115 | 21.864 | 0.393 | 2.441 | 89.546 | f417 PAD-tail/t12 80.379ms → f419 PAD-tail/t12 78.753ms |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | 481 | PAD-tail/t12 | 9/10 | 88.899 | 12.613 | 52.496 | 20.334 | 0.684 | 1.967 | 88.095 | f480 text-emission/t1046 76.331ms → f482 PAD-tail/t12 81.736ms |

### residual-high

Inventory n=6; membership SHA-256 `c88b2f593949d6477219f0ad34d9e0758ad658b4a1f30b71320469896713f5ed`. Phase counts: text-emission 6. Position and periodicity diagnostics are analyzer outputs: response ordinal p50/p95 9.000/15.000; distance from response end p50/p95 35.000/43.000; within-response high-row ordinal gap p50/p95 3.000/3.000.

| artifact | frame | phase/token | response pos | server | perception | Nano | EarTTS | codec | residual | wrapper | previous → next |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sustained-aged-26h-20260812` | 598 | text-emission/t42332 | 3/40 | 100.005 | 12.786 | 56.385 | 14.738 | 0.742 | 14.137 | 98.788 | f597 text-emission/t1073 83.867ms → f599 text-emission/t1278 105.139ms |
| `sustained-aged-26h-20260812` | 1758 | text-emission/t16385 | 7/42 | 92.858 | 12.422 | 52.832 | 11.212 | 0.642 | 14.780 | 91.887 | f1757 text-emission/t1395 84.180ms → f1759 text-emission/t63251 84.575ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 34 | text-emission/t48790 | 12/55 | 92.886 | 12.295 | 51.787 | 12.028 | 0.719 | 15.050 | 91.878 | f33 text-emission/t1044 80.225ms → f35 text-emission/t1044 84.520ms |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 37 | text-emission/t11368 | 15/55 | 93.417 | 12.314 | 53.604 | 10.889 | 0.676 | 14.980 | 92.462 | f36 text-emission/t1321 82.597ms → f38 text-emission/t1046 83.132ms |
| `8d4d4699-734c-4327-9848-1477e04a054a` | 29 | text-emission/t9406 | 9/37 | 88.640 | 12.577 | 52.851 | 10.707 | 0.378 | 11.378 | 87.891 | f28 text-emission/t1636 77.708ms → f30 text-emission/t1063 79.174ms |
| `2a0fa2ae-766d-4455-bfb0-885f67cc958a` | 482 | text-emission/t10532 | 9/16 | 90.691 | 12.569 | 51.958 | 10.859 | 0.665 | 13.777 | 89.828 | f481 text-emission/t1317 81.738ms → f483 text-emission/t1395 79.888ms |

### Step 4 targeting summary

EarTTS-high should be profiled first on PAD-tail states: the inventory is 34 PAD-tail versus 4 text-emission rows, with repeated within-response events rather than a single BOS/reset effect. Retain text-position captures as the control because the high predicate also occurs there. Residual-high should be investigated separately on text-emission states (6 of 6 rows); its normal EarTTS column in the row inventory supports keeping tracing/logging overhead as the first residual hypothesis. The non-unit, variable high-row gaps reported above do not support selecting a fixed-period kernel hypothesis from this corpus alone.

## Step 1 gate and source identities

PASS. Every reported aggregate is emitted by the qualified analyzer and carries an ordered membership SHA-256; every member includes its artifact and source SHA-256. Structural and queue rows cite their delivered/reconciled membership. The JSON retains the raw analyzer output for every artifact.

| artifact | source path | source SHA-256 |
|---|---|---|
| `nano-sequential-preflight-20260810-r1` | `/home/khkramer/.local/state/nemotron-voicechat/qualification/nano-sequential-preflight-20260810-r1/events.jsonl` | `166ec35a5e74045384c80042e01fe20e4795137719797634145dde6f6d9a74fd` |
| `sustained-aged-26h-20260812` | `/home/khkramer/.local/state/nemotron-voicechat/qualification/sustained-aged-26h-20260812/events.jsonl` | `acb205f6b6e5c786a039a2e3e238f12a15dc0fc0dcc971341dd8838b33f7353f` |
| `nano-sequential-production-r1` | `/home/khkramer/.local/state/nemotron-voicechat/qualification/nano-sequential-production-r1/events.jsonl` | `3d3157480b009d107109594953b07a32231150654e7bb2f1d85d072b8a4bef3b` |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/209fe177-bbb8-403d-ba50-740d3b6f3bf4/events.jsonl` | `baabf0a1203e8fa4bdcef9ebcbd9b2aeb7db952ec0eb83c58e18898fdcb2f880` |
| `8d4d4699-734c-4327-9848-1477e04a054a` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/8d4d4699-734c-4327-9848-1477e04a054a/events.jsonl` | `6ce13165a72e54cb820db11fcba3adaec9ec1a731c597f138d3c658ee7fac331` |
| `a4d2f917-a906-4498-a6b2-aecf67ef5db2` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/a4d2f917-a906-4498-a6b2-aecf67ef5db2/events.jsonl` | `1748621f4e4d9e14bfa129d4a3154111a8798b0681a7d0e9224f884112d3d12c` |
| `2a0fa2ae-766d-4455-bfb0-885f67cc958a` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/2a0fa2ae-766d-4455-bfb0-885f67cc958a/events.jsonl` | `2815a4e99d5ade234ade161cc83acd9e4b7fa93a4297bd55fb5e3f5ae596693d` |
| `ac88123a-28b6-48e9-bd2d-23dc101c48dc` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/ac88123a-28b6-48e9-bd2d-23dc101c48dc/events.jsonl` | `442df3de9adec6773debbcc23fe645b51405921f296ffc2860dcd0a41b635509` |
| `f110f063-8c9a-46e1-87a9-97dc682b8022` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/f110f063-8c9a-46e1-87a9-97dc682b8022/events.jsonl` | `aba32f50ad8c7c03fce4c3c1663ea0d42da119d4a9a52c1d90379bb4596ea1f3` |
| `de64cc34-2394-490e-a03c-cd920f96af25` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/de64cc34-2394-490e-a03c-cd920f96af25/events.jsonl` | `b04a401967d0e3ebc51d1f95abb294a2225c849a61cde81be9e54b6e1552b4ec` |
| `feca4ec0-2962-48df-b219-e175858659c8` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/feca4ec0-2962-48df-b219-e175858659c8/events.jsonl` | `7ea447eb15e28a87bd49ab47f9c8b7eedae16561d853a19e074fc578a91ef44a` |
| `97f9bacc-4d2f-4e55-9026-dc2590919023` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/97f9bacc-4d2f-4e55-9026-dc2590919023/events.jsonl` | `39387ec141bd0199a205c80c883192c08b44565a9548b5f7f91117c457ce7211` |
| `543bfc97-d9d1-40ae-84de-1a26b78a3756` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/543bfc97-d9d1-40ae-84de-1a26b78a3756/events.jsonl` | `f6247366c092557c501c8177e7b8cefaf48eee99ffd77e5f47593efa7edcdf10` |
| `70d3f7b5-2d3c-4998-846d-c85c539f2c59` | `/home/khkramer/.local/state/nemotron-voicechat/traces/model/70d3f7b5-2d3c-4998-846d-c85c539f2c59/events.jsonl` | `aac38611e28e870e70ef2b7146772b68de22855dd42c08f90a85000808fe805c` |

The retained negative controls remain `reports/step8-overlap-ab-20260807.md` and `reports/step5-host-barriers-20260807.md`; they were not rerun or altered during this offline inventory.
