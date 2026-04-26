"""
LoL 승률 예측 LSTM 모델
- 입력: csv 폴더의 lstm_features_KR_*.csv 파일들
- 출력: 분 단위 블루팀 승률 예측 (0.0 ~ 1.0)
- 설치: pip install torch scikit-learn pandas numpy matplotlib glob2
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== 설정 =====
CSV_DIR       = "csv"          # CSV 파일이 있는 폴더
DROP_MINUTES  = 2              # 앞 N분 제거 (초반 노이즈)
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.1
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False   

print(f"🖥  학습 디바이스: {DEVICE}")


# ============================================================
# 1. 데이터 로드
# ============================================================
def load_all_csv(csv_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(csv_dir, "lstm_features_KR_*.csv")) # glob: csv_dir 폴더의 csv 파일 찾기
    if not files:
        raise FileNotFoundError(f"'{csv_dir}' 폴더에 CSV 파일이 없습니다.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) # 
    print(f"✅ CSV 로드 완료: {len(files)}개 파일, {df['match_id'].nunique()}경기, {len(df)}행")
    return df


# ============================================================
# 2. 전처리
# ============================================================
def preprocess(df: pd.DataFrame):
    # 2-1. 앞 N분 제거
    df = df[df["minute"] >= DROP_MINUTES].copy()

    # 2-2. 피처 / 타겟 분리
    META_COLS   = ["match_id", "minute", "blue_win"]
    FEATURE_COLS = [c for c in df.columns if c not in META_COLS]

    # 2-3. MAX_MINUTES 결정 (95% 분위수)
    match_lengths = df.groupby("match_id")["minute"].max()
    MAX_MINUTES   = int(match_lengths.quantile(0.95))
    print(f"\n📊 경기 시간 분포:")
    print(match_lengths.describe().to_string())
    print(f"\n⏱  MAX_MINUTES (95% 분위수): {MAX_MINUTES}분")

    # 2-4. 정규화 (훈련 데이터 기준 fit)
    scaler = StandardScaler()

    # 경기 단위로 train/test 분리 (데이터 누수 방지)
    match_ids = df["match_id"].unique() # match_id 겹치는거 없이 추출
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42) 
    val_ids,  test_ids  = train_test_split(test_ids,  test_size=0.5, random_state=42)

    train_df = df[df["match_id"].isin(train_ids)]
    val_df   = df[df["match_id"].isin(val_ids)]
    test_df  = df[df["match_id"].isin(test_ids)]

    # 훈련 데이터로만 scaler fit
    scaler.fit(train_df[FEATURE_COLS])

    print(f"\n📂 데이터 분할:")
    print(f"  훈련: {len(train_ids)}경기")
    print(f"  검증: {len(val_ids)}경기")
    print(f"  테스트: {len(test_ids)}경기")

    return train_df, val_df, test_df, FEATURE_COLS, MAX_MINUTES, scaler


# ============================================================
# 3. 경기별 시퀀스 변환 (패딩 포함)
# ============================================================
def make_sequences(df: pd.DataFrame,
                   feature_cols: list,
                   max_minutes: int,
                   scaler: StandardScaler):
    X_list, y_list = [], []

    for match_id, group in df.groupby("match_id"):
        group = group.sort_values("minute")

        # 피처 정규화
        features = scaler.transform(group[feature_cols])  #FEATURE_COLS만 골라서 정규화
        label    = int(group["blue_win"].iloc[0]) # 정답 레이블, blue_win은 모든 행이 같은값이라 iloc[0]

        T = features.shape[0] # row의 개수 = 경기 시간 몇 분인지

        if T >= max_minutes: # 경기가 MAX_MINUTES 보다 길면 
            # 앞을 잘라서 뒤 max_minutes 분만 사용
            features = features[-max_minutes:]
        else: # 경기가 MAX_MINUTES보다 짧으면
            # 뒤를 0으로 패딩
            pad = np.zeros((max_minutes - T, features.shape[1]))
            features = np.vstack([features, pad])

        X_list.append(features)  # 1000경기 처리 후: X_list = [(45, 80), (45, 80), (45, 80), ...]  ← 1000개
        y_list.append(label)  # 1000경기 처리 후: y_list = [1, 0, 1, 1, 0, ...]                 ← 1000개

    X = np.array(X_list, dtype=np.float32)   # LSTM에 사용할 3차원 배열로 변환 (경기 수, max_minutes, 피처들)
    y = np.array(y_list, dtype=np.float32)   # 1차원 배열로 변환 (경기수, )
    return X, y


# ============================================================
# 4. Dataset / DataLoader
# ============================================================
class MatchDataset(Dataset): # numpy 배열 X,y를 파이토치가 배치 단위로 꺼내 쓸 수 있는 형태로 바꿔줌
    def __init__(self, X, y):
        self.X = torch.tensor(X) # 파이토치 텐서로 변환
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y) # 데이터셋 전체 크기 반환

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # 인덱스 받아서 해당 경기의 X,y를 반환


# ============================================================
# 5. LSTM 모델
# ============================================================
class WinPredictorLSTM(nn.Module): # 파이토치의 nn.Module 상속받아서 학습, 저장, GPU 이동 등의 기능 자동으로 사용 가능
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__() # 부모 클래스 nn.Module 초기화
        self.lstm = nn.LSTM(
            input_size  = input_size, # input_size: 매 분마다 들어오는 피처 수 (80-3)개
            hidden_size = hidden_size, # 77개 피처 -> 64개 hidden_state -> 피처 압축 64개로
            num_layers  = num_layers, # LSTM을 2층으로 쌓음 
            batch_first = True,  # batch, seq, feature순 
            dropout     = dropout if num_layers > 1 else 0.0, # LSTM 층 사이 dropout 적용 
        )
        self.dropout = nn.Dropout(dropout) # 0.1 무작위로 10% 뉴런을 꺼서 과적합 방지
        self.fc      = nn.Linear(hidden_size, 1) # 64개의 hidden state를 1개의 값으로 압축 - 승률 원시값
        self.sigmoid = nn.Sigmoid() # sigmoid로 0~1 사이로 변환

    def forward(self, x):
        # x: (batch, seq_len, input_size) - (1000경기, 45분, 77개 피처)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_size) - (1000경기, 45분, 64hidden state)
        out     = out[:, -1, :]        # 마지막 타임스텝만 사용 - (1000경기, 64) 마지막 타임스텝의 hidden state는 이전 모든 타임스텝의 정보를 압축해서 갖고있음
        out     = self.dropout(out)    # dropout 적용
        out     = self.fc(out)         # (batch, 1) - 승률 원시값 1개로 압축 
        out     = self.sigmoid(out)    # 0~1 변환
        return out.squeeze(1)          # (batch,)로 변환, y.shape와 맞춰주기 위해


# ============================================================
# 6. 학습 루프
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train() # dropout 켜짐
    total_loss = 0 # 손실 누적 변수 초기화
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad() # 이전 배치 기울기 초기화
        pred = model(X_batch) # forward 함수 실행 - 승률 출력
        loss = criterion(pred, y_batch) # 예측값과 정답을 비교해서 손실 계산, driterion - BCELoss, loss = 1000경기의 평균 손실값
        loss.backward() # 손실값을 기준으로 각 가중치가 얼마나 영향을 줬는지 계산
        optimizer.step() # backwrard()에서 계산된 기울기로 실제 가중치 업데이트
        total_loss += loss.item() # 누적 손실
    return total_loss / len(loader) # 평균 손실 반환


def eval_epoch(model, loader, criterion):
    model.eval() # dropout 꺼짐
    total_loss = 0
    all_preds, all_labels = [], [] # 모든 배치의 예측값, 정답을 리스트에 모음
    with torch.no_grad(): 
        for X_batch, y_batch in loader: 
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)  # 평균 손실 구함
    auc      = roc_auc_score(all_labels, all_preds) # AUC-ROC 점수
    acc      = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds]) # 0.5기준 1/0 변환 후 정확도 계산
    return avg_loss, auc, acc


# ============================================================
# 7. 학습 결과 시각화
# ============================================================
def plot_history(train_losses, val_losses, val_aucs, val_accs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses,   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(val_aucs, color="orange", label="Val AUC")
    axes[1].set_title("AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()

    axes[2].plot(val_accs, color="green", label="Val Accuracy")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.5, 1.0)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("📊 학습 그래프 저장: training_history.png")


# ============================================================
# 8. 분 단위 승률 예측 (단일 경기)
# ============================================================
def predict_winrate_by_minute(model, df, match_id, feature_cols, scaler):
    """
    특정 경기의 분 단위 승률을 예측해서 그래프로 출력
    """
    group    = df[df["match_id"] == match_id].sort_values("minute") # 분별로 정렬
    features = scaler.transform(group[feature_cols].values) # 77개 피처 정규화
    label    = int(group["blue_win"].iloc[0]) # 실제 결과
    minutes  = group["minute"].values 

    model.eval() #dropout 꺼짐
    win_rates = [] # 승률 리스트
    with torch.no_grad(): # 기울기 계산 x 
        # 1분씩 늘려가며 예측
        for t in range(1, len(features) + 1):
            seq   = torch.tensor(features[:t], dtype=torch.float32).unsqueeze(0).to(DEVICE) #unsqueeze(0) : 배치 차원 추가
            # t를 1씩 늘려가며 
            # t=1: features[:1]  → 3분 데이터만         
            # t=2: features[:2]  → 3분, 4분 데이터      
            # t=3: features[:3]  → 3분, 4분, 5분 데이터 
            # ...
            # t=36: features[:36] → 3분~38분 전체       
            pred  = model(seq).item() # 승률 원시값 1개 텐서에서 파이썬 숫자로
            win_rates.append(pred) # 분당 승률 리스트에 저장 

    # 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(minutes, win_rates, color="royalblue", linewidth=2, label="블루팀 승률")
    plt.axhline(0.5, color="gray",  linestyle="--", alpha=0.5, label="50%")
    plt.axhline(1.0, color="blue",  linestyle=":",  alpha=0.3)
    plt.axhline(0.0, color="red",   linestyle=":",  alpha=0.3)
    plt.fill_between(minutes, win_rates, 0.5,
                     where=[w >= 0.5 for w in win_rates],
                     alpha=0.15, color="blue")
    plt.fill_between(minutes, win_rates, 0.5,
                     where=[w < 0.5 for w in win_rates],
                     alpha=0.15, color="red")
    result = "블루팀 승리" if label == 1 else "레드팀 승리"
    plt.title(f"{match_id} | 실제 결과: {result}")
    plt.xlabel("경기 시간 (분)")
    plt.ylabel("블루팀 승률")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"winrate_{match_id}.png", dpi=150)
    plt.show()
    print(f"📊 승률 그래프 저장: winrate_{match_id}.png")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":

    # ── 1. 데이터 로드 ──
    df = load_all_csv(CSV_DIR) # 폴더의 모든 csv파일을 하나의 DataFrame으로 합침

    # ── 2. 전처리 ──
    train_df, val_df, test_df, feature_cols, MAX_MINUTES, scaler = preprocess(df) # 앞 2분 제거, 정규화, train/val/test분리
    INPUT_SIZE = len(feature_cols) # 77개
    print(f"\n🔢 입력 피처 수: {INPUT_SIZE}개") 

    # ── 3. 시퀀스 변환 ──
    print("\n⚙️  시퀀스 변환 중...")
    X_train, y_train = make_sequences(train_df, feature_cols, MAX_MINUTES, scaler) # train을 LSTM 입력 형태로 변환(800, 45, 77)
    X_val,   y_val   = make_sequences(val_df,   feature_cols, MAX_MINUTES, scaler)
    X_test,  y_test  = make_sequences(test_df,  feature_cols, MAX_MINUTES, scaler)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # ── 4. DataLoader ──
    train_loader = DataLoader(MatchDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True) # 32경기씩 배치로 묶음
    val_loader   = DataLoader(MatchDataset(X_val,   y_val),   batch_size=BATCH_SIZE) 
    test_loader  = DataLoader(MatchDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    # ── 5. 모델 초기화 ──
    model     = WinPredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE) # LSTM모델 생성
    criterion = nn.BCELoss() # 이진 분류 손실함수 BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam 옵티마이저. 모델의 모든 가중치를 학습률 0.001로 업데이트
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) # 손실이 5 epoch동안 줄지 않으면 학습률을 절반으로 줄임

    print(f"\n🧠 모델 구조:")
    print(model)

    # ── 6. 학습 ──
    print(f"\n🚀 학습 시작 (총 {EPOCHS} 에폭)")
    train_losses, val_losses, val_aucs, val_accs = [], [], [], [] # 에폭마다 기록할 리스트 초기화
    best_auc  = 0 # 최고 AUC 변수 초기화
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1): # 1~50까지 에폭 반복
        train_loss            = train_epoch(model, train_loader, criterion, optimizer)  # 한 에폭에서 훈련 한번,
        val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion) # 검증 한번 실행
        scheduler.step(val_loss) # 현재 손실 스케줄러에 전달

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)

        # 최고 모델 저장
        if val_auc > best_auc: # 검증 AUC가 이전보다 높을 때만 모델을 저장
            best_auc   = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pt") # 모델의 모든 가중치를 딕셔너리 형태로 꺼내서 파일로 저장

        if epoch % 5 == 0 or epoch == 1: # 1번째 에폭과 5의 배수 에폭마다 출력
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    print(f"\n🏆 최고 Val AUC: {best_auc:.4f} (Epoch {best_epoch})")

    # ── 7. 테스트 평가 ──
    model.load_state_dict(torch.load("best_model.pt")) # 저장해둔 최고 성능 모델 불러옴
    test_loss, test_auc, test_acc = eval_epoch(model, test_loader, criterion) # 학습에 사용하지 않은 테스트 데이터로 최종 성능 측정
    print(f"\n📋 테스트 결과:")
    print(f"  AUC-ROC : {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    # ── 8. 학습 그래프 ──
    plot_history(train_losses, val_losses, val_aucs, val_accs)

    # ── 9. 분 단위 승률 예측 예시 ──
    sample_id = test_df["match_id"].iloc[0] # 테스트 데이터에서 첫번째 경기를 골라서
    predict_winrate_by_minute(model, test_df, sample_id, feature_cols, scaler) # 분 단위 승률 그래프 그림
