"""
Riot API - LoL 경기 정보 수집기
필요한 패키지: pip install requests
"""

import requests
import json
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
# 3단계: 경기 상세 정보 가져오기
# =============================
def get_match_detail(match_id: str) -> Optional[dict]:
    """
    경기 ID로 상세 정보를 가져옵니다.
    """
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        print(f"✅ 경기 상세 정보 조회 성공: {match_id}")
        return response.json()
    else:
        print(f"❌ 경기 상세 조회 실패: {response.status_code} - {response.json()}")
        return None


# =============================
# 경기 정보 요약 출력
# =============================
def print_match_summary(match_data: dict, target_puuid: str):
    """
    경기 데이터에서 핵심 정보를 출력합니다.
    """
    info = match_data["info"]
    metadata = match_data["metadata"]

    game_duration = info["gameDuration"] // 60  # 초 → 분
    game_mode = info["gameMode"]

    print(f"\n{'='*50}")
    print(f"📋 경기 ID   : {metadata['matchId']}")
    print(f"🎮 게임 모드 : {game_mode}")
    print(f"⏱ 경기 시간 : {game_duration}분")
    print(f"{'='*50}")

    # 내 참가자 정보 찾기
    for participant in info["participants"]:
        if participant["puuid"] == target_puuid:
            result = "🏆 승리" if participant["win"] else "💀 패배"
            print(f"[내 기록]")
            print(f"  소환사명  : {participant['summonerName']}")
            print(f"  챔피언    : {participant['championName']}")
            print(f"  결과      : {result}")
            print(f"  KDA       : {participant['kills']} / {participant['deaths']} / {participant['assists']}")
            print(f"  CS        : {participant['totalMinionsKilled'] + participant['neutralMinionsKilled']}")
            print(f"  가한 피해 : {participant['totalDamageDealtToChampions']:,}")
            print(f"  비전 점수 : {participant['visionScore']}")
            break


# =============================
# 전체 실행 예시
# =============================
def main():
    # ▼ 여기에 조회할 소환사 정보 입력
    GAME_NAME = "텅빈설레임"   # Riot ID 이름
    TAG_LINE = "KR1"             # 태그 (#뒤 부분)
    MATCH_COUNT = 16              # 가져올 경기 수
    QUEUE_TYPE =  420            # 솔로 랭크 (None이면 전체)

    print(f"🔍 '{GAME_NAME}#{TAG_LINE}' 경기 정보 수집 시작...\n")

    # 1단계: PUUID 조회
    puuid = get_puuid(GAME_NAME, TAG_LINE)
    if not puuid:
        return

    # 2단계: 경기 ID 목록 조회
    match_ids = get_match_ids(puuid, count=MATCH_COUNT, queue=QUEUE_TYPE)
    if not match_ids:
        return

    # 3단계: 각 경기 상세 정보 조회 및 출력
    print(f"\n📊 최근 {len(match_ids)}경기 분석 중...")
    for match_id in match_ids:
        match_data = get_match_detail(match_id)
        if match_data:
            print_match_summary(match_data, puuid)

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
