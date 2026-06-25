# Offline RL — sonuç tablosu (ortak iş, Bölüm 20)

Havuz: `offline_pool.npz` = 420,103 transition / 3969 episode (3 üyenin npz'leri birleşik).
Eğitim env'e hiç dokunmadan sabit veride; eval seeds 0,1,2. Kod: `code/offline_rl.py`.

Seedli koşu (`torch.manual_seed(0)`) — sayılar birebir tekrar üretilir.

| Yöntem | cost_per_order ↓ | success | final max-Q | yorum |
|---|---|---|---|---|
| BC baseline | 22.47 | 0.50 | — | karışık-kalite veriyi klonlamak zayıf (random'dan kötü) |
| Naive offline DQN | 17.44 | 0.47 | **6785** | Q patlıyor 61→6785 = overestimation |
| **CQL** | **8.42** | 0.68 | 839 | konservatiflik Q'yu sınırlıyor, hem naive hem BC'yi geçer |
| ref: greedy_nearest | 4.57 | — | — | çıta |
| ref: online DQN (1M) | 6.76 | — | — | Role A en iyi |

Ham veri: `logs/offline_qstats.csv` (Q eğrisi), `logs/offline_results.json` (metrikler).
Weight: `weights/offline_cql.pt` (run_all.py bunu yükleyip aynı env'de doğruluyor).

**Sonuç:** (i) naive offline DQN'de overestimation çöküşü ölçülerek gösterildi (Q → 6785),
(ii) CQL düzeltti ve hem naive-offline-DQN'i (17.44) hem BC'yi (22.47) geçti → 8.42.
Greedy/online seviyesine ulaşmıyor (model-free hiçbir metot burada greedy'yi geçmiyor) ama
ödevin iki şartı da sağlandı. Kaynak: Kumar ve ark., Conservative Q-Learning, NeurIPS 2020.

Veri seti: `offline_*.npz` repo'ya konmadı (büyük); `offline_dlogs.zip` ayrı branch'te,
üye npz'leri kendi repolarında. Havuzu yeniden üret: `python pool_offline.py offline_pool.npz offline_dlogs.npz offline_ozan_karhan.npz offline_runa.npz`.
