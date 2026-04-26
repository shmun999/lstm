"""
LoL LSTM 학습 데이터 추출 스크립트
- 경기 ID를 입력받아 분 단위 피처를 CSV로 저장
- 사용법: python extract_lstm_features.py
"""

import requests
import json
import csv
import os
import sys
from collections import defaultdict

# ===== 설정 =====
API_KEY = "RGAPI-7663b9fb-8ff5-437e-89fb-45d7f5da9a08"   # ← 본인 API 키 입력
ROUTING = "asia"

HEADERS = {"X-Riot-Token": API_KEY}

# 포지션 레이블 (Riot API 기준)
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
POSITION_SHORT = {
    "TOP":     "top",
    "JUNGLE":  "jg",
    "MIDDLE":  "mid",
    "BOTTOM":  "bot",
    "UTILITY": "sup",
}


# ===== API 호출 =====
def api_get(url: str) -> dict | None:
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"❌ API 오류 {response.status_code}: {response.text[:200]}")
    return None


# ===== 포지션 매핑 만들기 =====
def build_position_map(participants: list) -> dict:
    """
    participantId → {position, teamId, puuid} 매핑 반환
    블루팀(100): participantId 1~5
    레드팀(200): participantId 6~10
    """
    pos_map = {}
    for p in participants:
        pid = p["participantId"]
        position = p.get("teamPosition") or p.get("individualPosition", "UNKNOWN")
        pos_map[pid] = {
            "position": position,
            "teamId":   p["teamId"],   # 100=블루, 200=레드
            "puuid":    p["puuid"],
        }
    return pos_map


# ===== 메인 추출 함수 =====
def extract_features(match_id: str) -> list[dict] | None:
    """
    한 경기의 분 단위 피처를 추출하여 리스트로 반환.
    각 원소는 1개 프레임(1분)의 피처 dict.
    """

    # ── 1. 매치 기본 정보 ──
    match_data = api_get(f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}")
    if not match_data:
        return None

    info         = match_data["info"]
    participants = info["participants"]
    pos_map      = build_position_map(participants)

    # 최종 승패 (블루팀 기준)
    blue_win = next(p["win"] for p in participants if p["teamId"] == 100)

    # 최종 visionScore (라인별) — 타임라인에 분 단위 없으므로 최종값만 보유
    final_vision = {}
    for p in participants:
        pid      = p["participantId"]
        position = pos_map[pid]["position"]
        team     = "blue" if p["teamId"] == 100 else "red"
        key      = f"{team}_{POSITION_SHORT.get(position, position.lower())}_vision_final"
        final_vision[key] = p.get("visionScore", 0)

    # ── 2. 타임라인 ──
    timeline = api_get(f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline")
    if not timeline:
        return None

    frames = timeline["info"]["frames"]

    # ── 3. 이벤트 누적 카운터 (프레임 순회 전 초기화) ──
    blue_kills   = 0;  red_kills   = 0
    blue_towers  = 0;  red_towers  = 0
    blue_dragons = 0;  red_dragons = 0
    blue_barons  = 0;  red_barons  = 0
    blue_heralds = 0;  red_heralds = 0

    # 와드 설치 누적 (분 단위 시야 대리 지표)
    blue_wards = defaultdict(int)   # position_short → 누적 설치 수
    red_wards  = defaultdict(int)

    rows = []

    for frame in frames:
        minute = round(frame["timestamp"] / 60000, 1)

        # ── 3-1. 이 프레임의 이벤트 처리 (누적) ──
        for event in frame.get("events", []):
            etype = event.get("type", "")

            if etype == "CHAMPION_KILL":
                killer = event.get("killerId", 0)
                if 1 <= killer <= 5:
                    blue_kills += 1
                elif 6 <= killer <= 10:
                    red_kills += 1

            elif etype == "BUILDING_KILL":
                # teamId = 파괴된 팀 → 상대팀이 획득
                destroyed_team = event.get("teamId", 0)
                if destroyed_team == 200:   # 레드팀 건물 파괴 = 블루팀 득점
                    blue_towers += 1
                elif destroyed_team == 100:
                    red_towers += 1

            elif etype == "ELITE_MONSTER_KILL":
                killer_team = event.get("killerTeamId", event.get("teamId", 0))
                monster     = event.get("monsterType", "")
                if monster == "DRAGON":
                    if killer_team == 100: blue_dragons += 1
                    else:                  red_dragons  += 1
                elif monster == "BARON_NASHOR":
                    if killer_team == 100: blue_barons += 1
                    else:                  red_barons  += 1
                elif monster == "RIFTHERALD":
                    if killer_team == 100: blue_heralds += 1
                    else:                  red_heralds  += 1

            elif etype == "WARD_PLACED":
                creator = event.get("creatorId", 0)
                if creator in pos_map:
                    team     = "blue" if pos_map[creator]["teamId"] == 100 else "red"
                    pos_s    = POSITION_SHORT.get(pos_map[creator]["position"], "unk")
                    if team == "blue":
                        blue_wards[pos_s] += 1
                    else:
                        red_wards[pos_s] += 1

        # ── 3-2. participantFrames 파싱 ──
        pframes = frame.get("participantFrames", {})

        # 포지션별 데이터 수집
        lane_data = {
            "blue": {s: {} for s in POSITION_SHORT.values()},
            "red":  {s: {} for s in POSITION_SHORT.values()},
        }

        for pid_str, pdata in pframes.items():
            pid      = int(pid_str)
            meta     = pos_map.get(pid, {})
            team     = "blue" if meta.get("teamId") == 100 else "red"
            pos_s    = POSITION_SHORT.get(meta.get("position", ""), None)
            if not pos_s:
                continue

            lane_data[team][pos_s] = {
                "current_gold": pdata.get("currentGold", 0),
                "total_gold":   pdata.get("totalGold", 0),
                "cs":           pdata.get("minionsKilled", 0) + pdata.get("jungleMinionsKilled", 0),
                "level":        pdata.get("level", 0),
            }

        # ── 3-3. 팀 전체 골드 합산 ──
        blue_total_gold = sum(d.get("total_gold", 0) for d in lane_data["blue"].values())
        red_total_gold  = sum(d.get("total_gold", 0) for d in lane_data["red"].values())

        # ── 3-4. 피처 dict 구성 ──
        row = {
            "match_id":    match_id,
            "minute":      minute,
            "blue_win":    int(blue_win),   # 타겟 레이블

            # 팀 킬
            "blue_kills":  blue_kills,
            "red_kills":   red_kills,
            "diff_kills":  blue_kills - red_kills,

            # 팀 타워
            "blue_towers": blue_towers,
            "red_towers":  red_towers,
            "diff_towers": blue_towers - red_towers,

            # 오브젝트
            "blue_dragons": blue_dragons,
            "red_dragons":  red_dragons,
            "blue_barons":  blue_barons,
            "red_barons":   red_barons,
            "blue_heralds": blue_heralds,
            "red_heralds":  red_heralds,
            "diff_dragons": blue_dragons - red_dragons,
            "diff_barons":  blue_barons  - red_barons,

            # 팀 전체 골드
            "blue_total_gold": blue_total_gold,
            "red_total_gold":  red_total_gold,
            "diff_total_gold": blue_total_gold - red_total_gold,
        }

        # 라인별 피처 (블루/레드 각각 + diff)
        for pos_s in POSITION_SHORT.values():
            b = lane_data["blue"].get(pos_s, {})
            r = lane_data["red"].get(pos_s, {})

            # 획득 골드 (currentGold = 현재 보유 골드)
            row[f"blue_{pos_s}_current_gold"] = b.get("current_gold", 0)
            row[f"red_{pos_s}_current_gold"]  = r.get("current_gold", 0)
            row[f"diff_{pos_s}_current_gold"] = b.get("current_gold", 0) - r.get("current_gold", 0)

            # 총 골드 (totalGold = 게임 시작부터 획득한 누적 골드)
            row[f"blue_{pos_s}_total_gold"] = b.get("total_gold", 0)
            row[f"red_{pos_s}_total_gold"]  = r.get("total_gold", 0)
            row[f"diff_{pos_s}_total_gold"] = b.get("total_gold", 0) - r.get("total_gold", 0)

            # CS
            row[f"blue_{pos_s}_cs"] = b.get("cs", 0)
            row[f"red_{pos_s}_cs"]  = r.get("cs", 0)
            row[f"diff_{pos_s}_cs"] = b.get("cs", 0) - r.get("cs", 0)

            # 레벨
            row[f"blue_{pos_s}_level"] = b.get("level", 0)
            row[f"red_{pos_s}_level"]  = r.get("level", 0)
            row[f"diff_{pos_s}_level"] = b.get("level", 0) - r.get("level", 0)

            # 시야 (와드 설치 누적 — 타임라인 분 단위 visionScore 미제공으로 대체)
            row[f"blue_{pos_s}_wards"] = blue_wards.get(pos_s, 0)
            row[f"red_{pos_s}_wards"]  = red_wards.get(pos_s, 0)
            row[f"diff_{pos_s}_wards"] = blue_wards.get(pos_s, 0) - red_wards.get(pos_s, 0)

        rows.append(row)

    # 최종 visionScore를 모든 행에 추가 (게임 끝 기준 고정값)
    for row in rows:
        row.update(final_vision)

    print(f"  ✅ {match_id}: {len(rows)}개 프레임, {len(rows[0])}개 피처 추출 완료")
    return rows


# ===== 경기별 개별 CSV 저장 =====
def save_match_csv(rows: list[dict], match_id: str, output_dir: str = "csv"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"lstm_features_{match_id}.csv")

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  💾 저장 완료: {output_path}  ({len(rows)}행 × {len(fieldnames)}열)")


# ===== 실행 =====
if __name__ == "__main__":
    # ▼ 분석할 경기 ID 목록 입력
    MATCH_IDS = [
        "KR_8188272844",
        "KR_8156471214",
        # "KR_xxxxxxxxxx",  # 추가 경기
    ]

    success, fail = 0, 0
    for i, match_id in enumerate(MATCH_IDS):
        print(f"\n[{i+1}/{len(MATCH_IDS)}] {match_id} 처리 중...")
        rows = extract_features(match_id)
        if rows:
            save_match_csv(rows, match_id)
            success += 1
        else:
            print(f"  ⚠️ {match_id} 데이터 추출 실패, 건너뜀")
            fail += 1

    print(f"\n✅ 완료: 성공 {success}개 / 실패 {fail}개")
