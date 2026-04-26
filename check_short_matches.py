"""
3분 이전에 끝나는 (리메이크) 경기 CSV 파일 검사 및 삭제
"""

import os
import glob
import csv

CSV_DIR   = "csv"
MIN_LIMIT = 3.0   # 이 분수 미만으로 끝나는 경기를 리메이크로 판단

# ===== 파일 목록 로드 =====
files = glob.glob(os.path.join(CSV_DIR, "lstm_features_KR_*.csv"))
if not files:
    print(f"❌ '{CSV_DIR}' 폴더에 CSV 파일이 없습니다.")
    exit()

print(f"📂 총 {len(files)}개 파일 검사 중...\n")

# ===== 각 파일의 최대 minute 확인 =====
short_files = []   # (파일경로, 최대 분수)

for filepath in files:
    max_minute = 0.0
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            minute = float(row.get("minute", 0))
            if minute > max_minute:
                max_minute = minute

    if max_minute < MIN_LIMIT:
        short_files.append((filepath, max_minute))

# ===== 결과 출력 =====
print(f"📊 검사 결과:")
print(f"  전체 파일 수        : {len(files)}개")
print(f"  {MIN_LIMIT}분 미만 경기 수 : {len(short_files)}개")
print(f"  정상 경기 수        : {len(files) - len(short_files)}개")

if short_files:
    print(f"\n⚠️  {MIN_LIMIT}분 미만 경기 파일 목록:")
    for filepath, max_minute in short_files:
        filename = os.path.basename(filepath)
        print(f"  {filename}  (최대 {max_minute}분)")

    # ===== 삭제 여부 확인 =====
    answer = input(f"\n위 {len(short_files)}개 파일을 삭제할까요? (y/n): ").strip().lower()
    if answer == "y":
        for filepath, _ in short_files:
            os.remove(filepath)
            print(f"  🗑️  삭제: {os.path.basename(filepath)}")
        print(f"\n✅ {len(short_files)}개 파일 삭제 완료")
    else:
        print("삭제를 건너뜁니다.")
else:
    print(f"\n✅ {MIN_LIMIT}분 미만 경기 없음 — 모든 파일이 정상입니다.")
