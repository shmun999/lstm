"""
csv 폴더의 lstm_features_*.csv 파일에서 중복 경기 ID 검사
"""

import os
import glob
from collections import defaultdict

CSV_DIR = "csv"

# ===== 파일 목록 로드 =====
files = glob.glob(os.path.join(CSV_DIR, "lstm_features_KR_*.csv"))
if not files:
    print(f"❌ '{CSV_DIR}' 폴더에 CSV 파일이 없습니다.")
    exit()

print(f"📂 총 {len(files)}개 파일 검사 중...\n")

# ===== 파일명에서 경기 ID 추출 =====
# 파일명 형식: lstm_features_KR_XXXXXXXXXX.csv
match_id_to_files = defaultdict(list)

for filepath in files:
    filename = os.path.basename(filepath)
    # lstm_features_ 제거, .csv 제거 → KR_XXXXXXXXXX
    match_id = filename.replace("lstm_features_", "").replace(".csv", "")
    match_id_to_files[match_id].append(filename)

# ===== 중복 검사 =====
total       = len(files)
unique      = len(match_id_to_files)
duplicates  = {mid: fnames for mid, fnames in match_id_to_files.items() if len(fnames) > 1}

print(f"📊 검사 결과:")
print(f"  전체 파일 수  : {total}개")
print(f"  고유 경기 수  : {unique}개")
print(f"  중복 파일 수  : {total - unique}개")

if duplicates:
    print(f"\n⚠️  중복 발견된 경기 ID ({len(duplicates)}개):")
    for match_id, fnames in duplicates.items():
        print(f"  {match_id}")
        for fname in fnames:
            print(f"    └── {fname}")

    # ===== 중복 파일 삭제 여부 확인 =====
    answer = input(f"\n중복 파일 {total - unique}개를 삭제할까요? (y/n): ").strip().lower()
    if answer == "y":
        deleted = 0
        for match_id, fnames in duplicates.items():
            for fname in fnames[1:]:   # 첫 번째 파일만 남기고 나머지 삭제
                filepath = os.path.join(CSV_DIR, fname)
                os.remove(filepath)
                print(f"  🗑️  삭제: {fname}")
                deleted += 1
        print(f"\n✅ {deleted}개 파일 삭제 완료")
    else:
        print("삭제를 건너뜁니다.")
else:
    print(f"\n✅ 중복 없음 — 모든 파일이 고유한 경기 ID를 가지고 있습니다.")
