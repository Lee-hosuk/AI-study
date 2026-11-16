# AI-study

인공지능 공부용 실습 코드 모음입니다.

## 1. CNN · RNN · LSTM 체험 실습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lee-hosuk/AI-study/blob/main/cnn_rnn_lstm_lab.ipynb)

설명만 읽는 대신 직접 돌려서 차이를 눈으로 확인하는 실험 3가지입니다.

| 실험 | 내용 | 확인할 수 있는 것 |
|---|---|---|
| 1. CNN vs MLP | 테스트 이미지를 1~2픽셀 이동 | CNN은 같은 필터를 이미지 전체에 재사용해서 위치가 바뀌어도 덜 흔들림 + 필터/특징맵 시각화 |
| 2. RNN vs LSTM | 숫자 T개를 읽고 맨 첫 숫자 맞히기 | T=40, 60에서 RNN은 10%(찍기)로 추락, LSTM은 100% 유지 |
| 3. 기울기 흐름 | 과거 시점으로 전달되는 기울기 크기 측정 | RNN의 기울기 소실(vanishing gradient)을 숫자로 확인 |

### 실행 방법

- **코랩:** 위의 *Open in Colab* 배지를 누르고 셀을 실행하세요. 1~2분이면 끝나고 그림이 셀 아래에 표시됩니다.
- **로컬:**
  ```bash
  pip install torch scikit-learn matplotlib
  python cnn_rnn_lstm_lab.py          # 약 1~2분
  python cnn_rnn_lstm_lab.py --quick  # 약 30초
  ```

데이터는 scikit-learn에 내장된 8x8 손글씨 숫자를 쓰므로 다운로드가 필요 없습니다.

| 파일 | 설명 |
|---|---|
| `cnn_rnn_lstm_lab.ipynb` | 코랩용 노트북 |
| `cnn_rnn_lstm_lab.py` | 같은 내용의 파이썬 스크립트 |
