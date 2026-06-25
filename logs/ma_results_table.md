# Multi-agent IDQN — sonuç tablosu (ortak iş, Bölüm 21)

8 drone, her biri bağımsız DQN ama tek paylaşılan ağ (parameter sharing) +
ortak replay. Env: `DroneDispatchMA-v0` (per-agent 59-dim obs, 4 aksiyon).
Kod: `code/train_ma_idqn.py` (60k adım). Eval seeds 0,1,2.

cost_per_order MA env'de reward'dan türetildi: `cost = 10·delivered + 5·ontime − return`
(teslim, TO_DROPOFF→IDLE geçişinden kesin sayılır).

| Politika | cost_per_order ↓ | teslim/ep | return | yorum |
|---|---|---|---|---|
| random (MA) | 8.80 | 85.3 | 433.8 | güçlü baseline (accept+move zaten teslim ediyor) |
| **IDQN (param sharing)** | **6.49** | 100.7 | 793.8 | random'ı geçer, merkezi referansı da geçer |
| ref: merkezi Double DQN | 6.76 | — | — | farklı env/aksiyon; paradigma referansı |

**Non-stationarity (eğri):** politika önce kötüleşiyor (return −1303→−1388, cost 66→82,
12k–36k adım — 8 ajan aynı anda değişiyor, hedef kayıyor), ε azaldıkça toparlıyor
(−913 @48k), geç yakınsıyor (+794, cost 6.49 @60k). Ham eğri: `logs/ma_idqn.csv`.

Kaynak: Tampuu ve ark. 2017 (IDQN); Gupta ve ark. AAMAS 2017 (parameter sharing).
