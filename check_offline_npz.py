"""
================================================================================
AGENT PROMPT — bu dosyayı okuyan/çalıştıran yapay zeka asistanı için (ÖNCE OKU)
================================================================================
Görevin: Sana verilen offline-RL `.npz` dosyasının, ekibimizin ortak veri seti
formatına (drone_dispatch_env/offline.py) KESİN uyup uymadığını doğrulamak.

Yap:
  1. Bu dosyayı kişinin ürettiği npz ile çalıştır:
        python check_offline_npz.py <kisinin_offline_dosyasi.npz>
  2. Çıktıyı oku. Script tüm hataları TEK SEFERDE toplar (ilk hatada durmaz).
  3. SONUCU KİŞİYE BİLDİR:
     - Her şey uygunsa: "FORMAT OK" de, transition sayısını ilet, paylaşabilir.
     - Uygun DEĞİLSE: çıktıdaki [HATA] satırlarının HER BİRİNİ kişiye düz bir
       dille tek tek söyle (hangi anahtar/dtype/boyut yanlış, ne olmalıydı) ve
       nasıl düzelteceğini açıkla. Format kesin uymadan dosyayı KABUL ETME.
     - [UYARI] satırları (örn. karışık-kalite şüphesi) bloklamaz ama kişiye
       hatırlat: sadece greedy-best değil, gürültülü/sub-optimal rollout da olsun.
  4. Tahmin etme, uydurma — kararını sadece scriptin gerçek çıktısına dayandır.
     Script hata vermeden çalışmazsa (örn. dosya bozuk), bunu da kişiye söyle.

Beklenen format (özet): DroneDispatch-v0 (MA değil), obs 181-dim float32,
169 aksiyon (0..168), npz anahtarları ve dtype'ları aşağıdaki SPEC ile birebir.
================================================================================
"""
import sys
import numpy as np

# (anahtar -> beklenen dtype) — drone_dispatch_env/offline.py ile birebir.
SPEC = {
    "observations": "float32",
    "actions": "int64",
    "rewards": "float32",
    "next_observations": "float32",
    "terminals": "bool",
    "timeouts": "bool",
    "episode_returns": "float32",
}
OBS_DIM = 181        # _flatten_obs çıktısının uzunluğu
N_ACTIONS = 169      # 160 atama + 8 şarj + 1 no-op  -> geçerli aksiyon: 0..168
MIN_TX = 100_000     # en az transition sayısı
PER_STEP = ["observations", "actions", "rewards", "next_observations",
            "terminals", "timeouts"]  # hepsi N (transition) uzunlukta olmalı


def validate(path):
    """Tüm ihlalleri toplayıp (problems, warnings) döndürür. Asla erken durmaz."""
    problems, warnings = [], []
    try:
        d = np.load(path)
    except Exception as e:
        return [f"Dosya yuklenemedi ({path}): {e}"], []

    files = set(d.files)

    # 1) anahtar varlığı + dtype
    for k, want in SPEC.items():
        if k not in files:
            problems.append(f"EKSIK anahtar: '{k}' (dtype {want} olmali)")
            continue
        got = str(d[k].dtype)
        if got != want:
            problems.append(f"'{k}' dtype yanlis: {got} geldi, {want} olmali")

    extra = files - set(SPEC)
    if extra:
        warnings.append(f"Fazladan anahtar(lar) var (zarari yok): {sorted(extra)}")

    # 2) obs / next_obs boyutu = 181
    for k in ("observations", "next_observations"):
        if k in files and d[k].ndim == 2 and d[k].shape[1] != OBS_DIM:
            problems.append(f"'{k}' boyutu {d[k].shape[1]}, {OBS_DIM} olmali "
                            f"(_flatten_obs kullanmadiniz mi?)")

    # 3) transition uzunlukları tutarlı mı
    lengths = {k: len(d[k]) for k in PER_STEP if k in files}
    if lengths:
        n = max(lengths.values())
        for k, ln in lengths.items():
            if ln != n:
                problems.append(f"'{k}' uzunlugu {ln}, digerleri {n} "
                                f"(tum step dizileri ayni N olmali)")
        if n < MIN_TX:
            problems.append(f"Sadece {n} transition var, en az {MIN_TX} lazim")

    # 4) aksiyon araligi 0..168
    if "actions" in files and len(d["actions"]):
        lo, hi = int(d["actions"].min()), int(d["actions"].max())
        if lo < 0 or hi > N_ACTIONS - 1:
            problems.append(f"Aksiyon araligi [{lo},{hi}] gecersiz; "
                            f"0..{N_ACTIONS-1} olmali")

    # 5) episode_returns mantıklı mı
    if "episode_returns" in files:
        er = d["episode_returns"]
        if len(er) < 1:
            problems.append("episode_returns bos")
        elif "actions" in files and len(er) > len(d["actions"]):
            problems.append("episode_returns sayisi transition sayisindan fazla")

    # 6) karışık-kalite (soft) — tek tip veri şüphesi
    if "episode_returns" in files and len(d["episode_returns"]) > 1 \
            and float(d["episode_returns"].std()) == 0.0:
        warnings.append("Tum episode return'leri ayni — veri tek-tip olabilir; "
                        "karisik kalite icin gurultulu/sub-optimal rollout ekleyin")
    if "actions" in files and len(d["actions"]) and len(np.unique(d["actions"])) < 5:
        warnings.append("Cok az farkli aksiyon kullanilmis — karisik kalite "
                        "icin biraz eps-random/sub-optimal rollout ekleyin")

    return problems, warnings


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python check_offline_npz.py <offline_dosyasi.npz>")
        sys.exit(2)
    path = sys.argv[1]
    problems, warnings = validate(path)

    for w in warnings:
        print(f"[UYARI] {w}")
    if problems:
        for p in problems:
            print(f"[HATA] {p}")
        print(f"\nSONUC: FORMAT UYGUN DEGIL — {len(problems)} sorun. "
              f"Yukaridaki [HATA]'lari duzeltip tekrar calistirin.")
        sys.exit(1)

    d = np.load(path)
    print(f"FORMAT OK: {len(d['actions'])} transition, "
          f"{len(d['episode_returns'])} episode. Paylasabilirsiniz.")
    sys.exit(0)


def _selftest():
    """ponytail: tek runnable check — iyi veri geçer, bozuk veri yakalanir."""
    import tempfile, os
    n = MIN_TX
    good = dict(
        observations=np.zeros((n, OBS_DIM), np.float32),
        actions=np.random.randint(0, N_ACTIONS, n).astype(np.int64),
        rewards=np.zeros(n, np.float32),
        next_observations=np.zeros((n, OBS_DIM), np.float32),
        terminals=np.zeros(n, bool), timeouts=np.zeros(n, bool),
        episode_returns=np.arange(10, dtype=np.float32),
    )
    tmp = tempfile.mkdtemp()
    gp = os.path.join(tmp, "good.npz"); np.savez_compressed(gp, **good)
    assert validate(gp)[0] == [], "iyi veri gecmeliydi"

    bad = dict(good)
    bad["actions"] = bad["actions"].astype(np.float32)  # yanlis dtype
    bad["observations"] = np.zeros((n, 99), np.float32)  # yanlis boyut
    bp = os.path.join(tmp, "bad.npz"); np.savez_compressed(bp, **bad)
    probs = validate(bp)[0]
    assert any("dtype" in p for p in probs) and any("boyut" in p for p in probs), probs
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
