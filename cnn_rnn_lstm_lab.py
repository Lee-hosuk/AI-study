"""
CNN · RNN · LSTM 체험 실습 코드
=================================

설명만 읽는 대신 "직접 돌려서 차이를 눈으로 확인"하는 3가지 실험입니다.

  실험 1. CNN vs MLP  : 이미지를 1~2픽셀 옮겨도 CNN은 덜 흔들린다 (지역성 + 가중치 공유)
                        + CNN이 무엇을 보고 있는지 필터/특징맵 시각화
  실험 2. RNN vs LSTM : "맨 앞 글자를 기억하라" 과제. 시퀀스가 길어지면 RNN은 무너지고 LSTM은 버틴다
  실험 3. 기울기 흐름  : 왜 그런가? 과거 시점으로 전달되는 기울기(gradient) 크기를 직접 측정

필요 패키지:  pip install torch scikit-learn matplotlib
실행:        python cnn_rnn_lstm_lab.py            (CPU 기준 약 1~2분)
             python cnn_rnn_lstm_lab.py --quick    (빠른 확인용, 약 30초)
코랩:        이 코드 전체를 셀 하나에 붙여넣고 실행 (그림이 셀 아래에 바로 표시됨)
             빠른 모드는 맨 위에 QUICK = True 한 줄 추가
결과 그림은 ./results 폴더에 저장됩니다.

데이터: sklearn에 내장된 손글씨 숫자(8x8) 데이터를 쓰므로 인터넷 다운로드가 필요 없습니다.
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로 저장
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ══════════════════════════════════════════════════════════════
# 실험 1. CNN vs MLP — 이미지를 옮겨도 알아볼까?
# ══════════════════════════════════════════════════════════════
class MLP(nn.Module):
    """이미지를 64개 숫자의 '줄'로 펴서 본다 → 픽셀 위치 관계를 모름"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """3x3 필터가 이미지 위를 미끄러지며 '모양'을 찾는다 → 위치가 바뀌어도 같은 필터가 반응"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 8x8 → 8x8 (16채널)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)  # 4x4 → 4x4 (64채널)
        self.fc = nn.Linear(64, 10)

    def forward(self, x, return_maps=False):
        m1 = F.relu(self.conv1(x))
        x = F.max_pool2d(m1, 2)               # 8x8 → 4x4
        m2 = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(m2, 1)      # 4x4 → 1x1 (위치 정보를 의도적으로 버림)
        out = self.fc(x.flatten(1))
        return (out, m1, m2) if return_maps else out


def shift_images(x, dx, dy):
    """이미지를 dx(가로), dy(세로) 픽셀만큼 이동. 빈 곳은 0으로 채움"""
    out = torch.zeros_like(x)
    H, W = x.shape[-2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd] = x[..., ys, xs]
    return out


def train_classifier(model, Xtr, ytr, epochs, lr=3e-3, bs=64):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return (model(X).argmax(1) == y).float().mean().item()


def experiment_cnn(epochs):
    banner("실험 1. CNN vs MLP : 이미지를 옮겨도 알아볼까?")
    digits = load_digits()
    X = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)  # (N,1,8,8), 0~1
    y = torch.tensor(digits.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xtr, Xte, ytr, yte = [t.to(DEVICE) for t in (Xtr, Xte, ytr, yte)]
    print(f"데이터: 학습 {len(Xtr)}장 / 테스트 {len(Xte)}장, 이미지 크기 8x8")

    mlp, cnn = MLP(), SmallCNN()
    print(f"파라미터 수 → MLP: {count_params(mlp):,}   CNN: {count_params(cnn):,}  (CNN이 더 적음!)")

    t = time.time(); train_classifier(mlp, Xtr, ytr, epochs); print(f"MLP 학습 완료 ({time.time()-t:.1f}s)")
    t = time.time(); train_classifier(cnn, Xtr, ytr, epochs); print(f"CNN 학습 완료 ({time.time()-t:.1f}s)")

    # 학습 때는 '원래 위치'만 봤다. 테스트 때 이미지를 옮겨보자.
    shifts = [(0, 0), (1, 0), (0, 1), (-1, 0), (1, 1), (2, 0), (-2, 1)]
    print("\n테스트 이미지를 이동시켰을 때 정확도 (학습 때는 이동한 이미지를 본 적 없음)")
    print(f"{'이동(dx,dy)':>12} | {'MLP':>7} | {'CNN':>7}")
    print("-" * 34)
    res = {"MLP": [], "CNN": []}
    for dx, dy in shifts:
        Xs = shift_images(Xte, dx, dy)
        a_m, a_c = accuracy(mlp, Xs, yte), accuracy(cnn, Xs, yte)
        res["MLP"].append(a_m); res["CNN"].append(a_c)
        print(f"{str((dx, dy)):>12} | {a_m*100:6.1f}% | {a_c*100:6.1f}%")

    # 그림 1: 이동 강건성 막대그래프
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(shifts)); w = 0.38
    ax.bar(xs - w/2, np.array(res["MLP"]) * 100, w, label="MLP", color="#9aa5b1")
    ax.bar(xs + w/2, np.array(res["CNN"]) * 100, w, label="CNN", color="#2b6cb0")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}" for s in shifts])
    ax.set_xlabel("shift (dx, dy) in pixels"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Exp 1. Accuracy when test digits are shifted"); ax.legend(); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(f"{OUT}/1_cnn_vs_mlp_shift.png", dpi=130); plt.close(fig)

    # 그림 2: CNN이 보는 것 — 1층 필터와 특징맵
    sample = Xte[0:1]
    with torch.no_grad():
        _, m1, _ = cnn(sample, return_maps=True)
    filt = cnn.conv1.weight.detach().cpu().squeeze(1)  # (16,3,3)
    fig, axes = plt.subplots(3, 9, figsize=(13, 4.6))
    axes[0, 0].imshow(sample.cpu()[0, 0], cmap="gray_r"); axes[0, 0].set_title(f"input ({yte[0].item()})")
    for i in range(8):
        axes[0, i + 1].imshow(filt[i], cmap="bwr"); axes[0, i + 1].set_title(f"filter {i}", fontsize=8)
    for r in (1, 2):
        axes[r, 0].axis("off")
        for i in range(8):
            k = (r - 1) * 8 + i
            axes[r, i + 1].imshow(m1.cpu()[0, k], cmap="magma"); axes[r, i + 1].set_title(f"map {k}", fontsize=8)
    for a in axes.flat: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Exp 1. Conv1 filters (top) and the feature maps they produce (bottom 2 rows)")
    fig.tight_layout(); fig.savefig(f"{OUT}/1_cnn_feature_maps.png", dpi=130); plt.close(fig)

    print("\n👉 관찰 포인트: 이동 폭이 커질수록 MLP는 크게 떨어지고, CNN은 상대적으로 덜 떨어집니다.")
    print("   CNN은 같은 3x3 필터를 이미지 전체에 재사용하므로 '어디에 있든 같은 모양'을 찾습니다.")
    print(f"   그림 저장: {OUT}/1_cnn_vs_mlp_shift.png, {OUT}/1_cnn_feature_maps.png")


# ══════════════════════════════════════════════════════════════
# 실험 2. RNN vs LSTM — 긴 시퀀스에서 처음 것을 기억할 수 있나?
# ══════════════════════════════════════════════════════════════
# 과제: 길이 T의 랜덤 숫자열 [7, 2, 9, 0, 4, ... ] 을 하나씩 읽은 뒤, 마지막에 "맨 첫 숫자(7)"를 맞힌다.
#       정답 정보는 오직 t=0 에만 있고, 나머지 T-1개는 전부 방해 잡음이다.
#       → T가 길수록 "먼 과거 정보를 끝까지 들고 가는 능력"이 필요하다. (찍으면 10%)

N_CLASSES = 10


def make_memory_batch(bs, T):
    seq = torch.randint(0, N_CLASSES, (bs, T))
    x = F.one_hot(seq, N_CLASSES).float()      # (bs, T, 10)
    y = seq[:, 0]                              # 첫 번째 숫자가 정답
    return x.to(DEVICE), y.to(DEVICE)


# LSTM 망각 게이트(forget gate) 바이어스 초기값.
# 양수로 두면 학습 초반부터 "기본값은 기억 유지"로 시작한다 (LSTM의 표준 관행).
# 👉 0으로 바꿔서 돌려보세요: LSTM도 긴 시퀀스에서 처음엔 잘 못 배우는 걸 볼 수 있습니다.
FORGET_BIAS = 3.0


class SeqClassifier(nn.Module):
    def __init__(self, cell="rnn", hidden=64):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(N_CLASSES, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_CLASSES)
        if cell == "lstm":
            # PyTorch LSTM 바이어스 순서: [입력 게이트 | 망각 게이트 | 후보값 | 출력 게이트]
            # bias_ih + bias_hh 두 개가 더해지므로 각각 절반씩 넣는다
            for name, p in self.rnn.named_parameters():
                if "bias" in name:
                    p.data.zero_()
                    p.data[hidden:2 * hidden] = FORGET_BIAS / 2

    def forward(self, x):
        out, _ = self.rnn(x)          # out: (bs, T, hidden) — 매 시점의 은닉 상태
        return self.head(out[:, -1])  # 마지막 시점의 은닉 상태만으로 답을 낸다


def train_memory(cell, T, steps, lr=3e-3, bs=128):
    torch.manual_seed(SEED)
    model = SeqClassifier(cell).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for s in range(steps):
        x, y = make_memory_batch(bs, T)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 기울기 폭발 방지 (공정 비교를 위해 둘 다 적용)
        opt.step()
        if s % 25 == 0 or s == steps - 1:
            curve.append((s, (logits.argmax(1) == y).float().mean().item()))
    model.eval()
    with torch.no_grad():
        x, y = make_memory_batch(2000, T)
        acc = (model(x).argmax(1) == y).float().mean().item()
    return model, acc, curve


def experiment_rnn_lstm(lengths, steps):
    banner("실험 2. RNN vs LSTM : 시퀀스 맨 앞 숫자를 끝까지 기억할 수 있나?")
    print("과제: 랜덤 숫자 T개를 하나씩 읽고, 마지막에 '첫 번째 숫자'를 맞히기 (찍으면 10%)")
    print(f"파라미터 수 → RNN: {count_params(SeqClassifier('rnn')):,}   "
          f"LSTM: {count_params(SeqClassifier('lstm')):,}  (LSTM은 게이트 4개라 약 4배)\n")
    print(f"{'길이 T':>7} | {'RNN':>7} | {'LSTM':>7} | 시간")
    print("-" * 40)
    results, curves, models = {}, {}, {}
    for T in lengths:
        t0 = time.time()
        m_r, a_r, c_r = train_memory("rnn", T, steps)
        m_l, a_l, c_l = train_memory("lstm", T, steps)
        results[T] = (a_r, a_l); curves[T] = (c_r, c_l); models[T] = (m_r, m_l)
        print(f"{T:>7} | {a_r*100:6.1f}% | {a_l*100:6.1f}% | {time.time()-t0:.0f}s")

    # 그림 3: 길이별 정확도 + 가장 긴 길이의 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    Ts = list(results)
    ax1.plot(Ts, [results[T][0] * 100 for T in Ts], "o-", color="#c05621", label="RNN")
    ax1.plot(Ts, [results[T][1] * 100 for T in Ts], "s-", color="#2b6cb0", label="LSTM")
    ax1.axhline(10, ls="--", color="gray", lw=1, label="random guess (10%)")
    ax1.set_xlabel("sequence length T"); ax1.set_ylabel("test accuracy (%)"); ax1.set_ylim(0, 105)
    ax1.set_title("Remember the FIRST token"); ax1.legend()
    Tmax = Ts[-1]
    for c, name, col in ((curves[Tmax][0], "RNN", "#c05621"), (curves[Tmax][1], "LSTM", "#2b6cb0")):
        ax2.plot([s for s, _ in c], [a * 100 for _, a in c], color=col, label=name)
    ax2.axhline(10, ls="--", color="gray", lw=1)
    ax2.set_xlabel("training step"); ax2.set_ylabel("batch accuracy (%)"); ax2.set_ylim(0, 105)
    ax2.set_title(f"Training curve at T={Tmax}"); ax2.legend()
    fig.suptitle("Exp 2. Vanilla RNN vs LSTM on a long-range memory task")
    fig.tight_layout(); fig.savefig(f"{OUT}/2_rnn_vs_lstm_memory.png", dpi=130); plt.close(fig)

    print("\n👉 관찰 포인트: T가 짧으면 둘 다 잘 하지만, 길어질수록 RNN은 10%(찍기) 근처로 추락합니다.")
    print("   LSTM은 '셀 상태(cell state)'라는 고속도로에 정보를 실어 게이트로 보호하기 때문에 오래 기억합니다.")
    print(f"   그림 저장: {OUT}/2_rnn_vs_lstm_memory.png")
    return models


# ══════════════════════════════════════════════════════════════
# 실험 3. 왜 RNN은 잊어버릴까? — 기울기 소실을 직접 측정
# ══════════════════════════════════════════════════════════════
def gradient_flow(model, T):
    """마지막 출력의 손실을 각 시점 입력 x_t 로 미분한 크기 ||dL/dx_t|| 를 측정.
       이 값이 작다 = 그 시점의 입력이 학습 신호를 거의 못 받는다 = 배울 수 없다"""
    model.train()  # cuDNN RNN은 backward에 train 모드가 필요
    x, y = make_memory_batch(256, T)
    x.requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    g = x.grad.norm(dim=2).mean(0).detach().cpu().numpy()  # (T,)
    return g


def experiment_gradients(T):
    banner(f"실험 3. 기울기는 과거로 얼마나 전달될까? (T={T}, 학습 전 초기 상태)")
    torch.manual_seed(SEED)
    rnn, lstm = SeqClassifier("rnn").to(DEVICE), SeqClassifier("lstm").to(DEVICE)
    g_r, g_l = gradient_flow(rnn, T), gradient_flow(lstm, T)

    print(f"{'시점 t':>7} | {'RNN ||dL/dx_t||':>16} | {'LSTM ||dL/dx_t||':>17}")
    print("-" * 48)
    for t in sorted(set([0, 1, T // 4, T // 2, 3 * T // 4, T - 2, T - 1])):
        print(f"{t:>7} | {g_r[t]:16.2e} | {g_l[t]:17.2e}")
    print(f"\n마지막 시점 대비 첫 시점의 기울기 비율 →  RNN: {g_r[0]/g_r[-1]:.1e}   LSTM: {g_l[0]/g_l[-1]:.1e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(T), g_r / g_r[-1], color="#c05621", label="RNN")
    ax.semilogy(range(T), g_l / g_l[-1], color="#2b6cb0", label="LSTM")
    ax.set_xlabel("time step t  (answer lives at t=0, loss is computed at t=T-1)")
    ax.set_ylabel("||dL/dx_t||  (relative to last step, log)")
    ax.set_title("Exp 3. How much learning signal reaches each past step?")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/3_gradient_flow.png", dpi=130); plt.close(fig)

    print("\n👉 관찰 포인트: RNN은 과거로 갈수록 기울기가 지수적으로 줄어듭니다(기울기 소실, vanishing gradient).")
    print("   t=0 의 정답 정보가 학습 신호를 거의 못 받으니, 실험 2에서 RNN이 배우지 못한 것입니다.")
    print("   LSTM은 셀 상태가 '덧셈'으로 이어져 있어 기울기가 훨씬 덜 줄어듭니다.")
    print(f"   그림 저장: {OUT}/3_gradient_flow.png")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="빠른 확인용 (짧게 학습)")
    args, _ = ap.parse_known_args()  # 코랩/주피터 셀에 붙여넣어도 오류 안 나도록
    QUICK = globals().get("QUICK", False)  # 코랩 셀에서는 위에 QUICK = True 라고 써두면 빠른 모드
    args.quick = args.quick or QUICK

    print(f"device: {DEVICE}, torch {torch.__version__}")
    if args.quick:
        experiment_cnn(epochs=10)
        experiment_rnn_lstm(lengths=[5, 20, 40], steps=600)
        experiment_gradients(T=40)
    else:
        experiment_cnn(epochs=40)
        experiment_rnn_lstm(lengths=[5, 10, 20, 40, 60], steps=1500)
        experiment_gradients(T=60)

    banner("끝! results/ 폴더의 그림을 확인해 보세요.")
    print("직접 바꿔보면 좋은 것들:")
    print(" - SmallCNN 의 adaptive_avg_pool2d 를 지우고 flatten 하면 이동에 더 약해질까?")
    print(" - SeqClassifier 의 hidden 크기, 학습 step 수, lr 을 바꾸면 RNN이 더 긴 T도 버틸까?")
    print(" - nn.GRU 를 추가해 LSTM과 비교해 보기 (rnn_cls 딕셔너리에 'gru': nn.GRU 추가)")

    # 코랩/주피터에서 실행 중이면 결과 그림을 셀 아래에 바로 보여준다
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            for f in sorted(os.listdir(OUT)):
                if f.endswith(".png"):
                    print(f"\n[{f}]")
                    display(Image(filename=os.path.join(OUT, f)))
    except ImportError:
        pass
