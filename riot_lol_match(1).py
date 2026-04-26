"""
Riot API - LoL 경기 정보 수집기
필요한 패키지: pip install requests
"""

import requests
import csv
import os
from typing import Optional

# =============================
# 설정
# =============================
API_KEY = "RGAPI-7663b9fb-8ff5-437e-89fb-45d7f5da9a08"  # 본인 API 키 입력

# 지역 설정 (한국 서버)
REGION = "kr"                    # 계정 서버 (kr, na1, euw1 등)
ROUTING = "asia"                 # 매치 라우팅 (asia, americas, europe)

HEADERS = {"X-Riot-Token": API_KEY}


# =============================
# 1단계: PUUID 가져오기
# =============================
def get_puuid(game_name: str, tag_line: str) -> Optional[str]:
    """
    Riot ID(게임 이름 + 태그)로 PUUID를 가져옵니다.
    예: get_puuid("Hide on bush", "KR1")
    """
    url = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ PUUID 조회 성공: {data['puuid'][:20]}...")
        return data["puuid"]
    else:
        print(f"❌ PUUID 조회 실패: {response.status_code} - {response.json()}")
        return None


# =============================
# 2단계: 최근 경기 ID 목록 가져오기
# =============================
def get_match_ids(puuid: str, count: int = 5, queue: Optional[int] = None) -> list:
    """
    PUUID로 최근 경기 ID 목록을 가져옵니다.

    queue 종류:
        420 = 솔로 랭크
        440 = 자유 랭크
        450 = 칼바람
        400 = 일반 게임
        None = 모든 게임 모드
    """
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"count": count}
    if queue:
        params["queue"] = queue

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        match_ids = response.json()
        print(f"✅ 경기 ID {len(match_ids)}개 조회 성공")
        return match_ids
    else:
        print(f"❌ 경기 ID 조회 실패: {response.status_code} - {response.json()}")
        return []


# =============================
# 3단계: CSV로 저장
# =============================
def save_match_ids_to_csv(match_ids: list, game_name: str, tag_line: str, output_dir: str = "csv"):
    """
    경기 ID 목록을 CSV 파일로 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename  = f"{game_name}_{tag_line}_match_ids.csv".replace(" ", "_")
    filepath  = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id"])          # 헤더
        for match_id in match_ids:
            writer.writerow([match_id])

    print(f"✅ 저장 완료: {filepath}  ({len(match_ids)}개)")
    return filepath


# =============================
# 전체 실행
# =============================
def main():
    GAME_NAME  = input("소환사 이름 입력 (예: Hide on bush): ").strip()
    TAG_LINE   = input("태그 입력 (예: KR1): ").strip()
    QUEUE_TYPE = 420   # 솔로 랭크
    COUNT      = 100   # 가져올 경기 수

    print(f"\n🔍 '{GAME_NAME}#{TAG_LINE}' 솔로랭크 경기 ID 수집 시작...\n")

    # 1단계: PUUID 조회
    puuid = get_puuid(GAME_NAME, TAG_LINE)
    if not puuid:
        return

    # 2단계: 경기 ID 목록 조회
    match_ids = get_match_ids(puuid, count=COUNT, queue=QUEUE_TYPE)
    if not match_ids:
        return

    # 3단계: CSV 저장
    save_match_ids_to_csv(match_ids, GAME_NAME, TAG_LINE)

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
