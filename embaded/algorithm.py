from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


# -----------------------------------------------------------------------------
# AI-tunable baseline configuration
# These weights can later be adjusted by Gemini or another optimizer without
# changing the overall recommendation architecture.
# -----------------------------------------------------------------------------
AI_TUNABLE_WEIGHTS = {
    "text_similarity": 0.45,
    "category_match": 0.25,
    "time_match": 0.20,
    "recency_bonus": 0.10,
}

CATEGORY_KEYWORDS = {
    "meeting": ["미팅", "회의", "meeting", "call", "브리핑", "협의"],
    "consultation": ["상담", "consult", "문의", "상담예약"],
    "lunch": ["점심", "lunch", "식사"],
    "dinner": ["저녁", "dinner", "회식"],
    "schedule": ["일정", "스케줄", "약속", "reservation", "예약"],
    "general": ["일반", "기타", "todo", "메모"],
}

TIME_BUCKETS = {
    "morning": (5, 11),
    "afternoon": (12, 17),
    "evening": (18, 21),
    "night": (22, 23),
    "late_night": (0, 4),
}


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"[^0-9가-힣a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [token for token in text.split(" ") if token]


# -----------------------------------------------------------------------------
# Category and time extraction
# -----------------------------------------------------------------------------
def infer_category(title: Any, note: Any = "") -> str:
    haystack = normalize_text(f"{title} {note}")
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if normalize_text(keyword) in haystack:
                return category
    return "general"


def get_time_bucket(dt: Any) -> str:
    if dt is None:
        return "unknown"

    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return "unknown"

    hour = dt.hour
    for bucket, (start, end) in TIME_BUCKETS.items():
        if start <= hour <= end:
            return bucket
    return "unknown"


def get_weekday(dt: Any) -> str:
    if dt is None:
        return "unknown"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return "unknown"
    weekday_names = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    return weekday_names[dt.weekday()]


# -----------------------------------------------------------------------------
# Preference profiles
# -----------------------------------------------------------------------------
def reservation_to_features(reservation: Dict[str, Any]) -> Dict[str, Any]:
    title = reservation.get("title") or ""
    note = reservation.get("note") or ""
    category = reservation.get("category") or infer_category(title, note)
    dt = reservation.get("reservation_datetime")

    token_list = tokenize(f"{title} {note}")
    return {
        "category": category,
        "time_bucket": get_time_bucket(dt),
        "weekday": get_weekday(dt),
        "tokens": token_list,
        "raw_text": normalize_text(f"{title} {note}"),
    }


def build_user_preference_vector(liked_reservations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    category_counts: Counter[str] = Counter()
    time_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()

    total = 0
    for reservation in liked_reservations:
        features = reservation_to_features(reservation)
        category_counts[features["category"]] += 1
        if features["time_bucket"] != "unknown":
            time_counts[features["time_bucket"]] += 1
        for token in features["tokens"]:
            if len(token) > 1:
                keyword_counts[token] += 1
        total += 1

    return {
        "category_counts": category_counts,
        "time_counts": time_counts,
        "keyword_counts": keyword_counts,
        "total_likes": max(total, 1),
    }


# -----------------------------------------------------------------------------
# Similarity helpers
# -----------------------------------------------------------------------------
def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    keys = set(vec_a) | set(vec_b)
    if not keys:
        return 0.0

    dot = sum(vec_a.get(key, 0.0) * vec_b.get(key, 0.0) for key in keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


# -----------------------------------------------------------------------------
# Score a single candidate reservation against a user's preference profile
# -----------------------------------------------------------------------------
def score_candidate_against_profile(candidate: Dict[str, Any], profile: Dict[str, Any]) -> float:
    features = reservation_to_features(candidate)

    category_counts = profile["category_counts"]
    category_total = sum(category_counts.values())
    category_score = 0.0
    if category_total:
        category_score = category_counts.get(features["category"], 0) / category_total

    time_counts = profile["time_counts"]
    time_total = sum(time_counts.values())
    time_score = 0.0
    if time_total:
        time_score = time_counts.get(features["time_bucket"], 0) / time_total

    candidate_tokens = set(features["tokens"])
    profile_tokens = set(profile["keyword_counts"].keys())
    text_score = jaccard_similarity(candidate_tokens, profile_tokens)

    # A light recency bonus for very recent likes, useful when a user has a fresh
    # preference pattern. This does not replace the real ranking model but helps.
    recency_bonus = 0.0
    liked_at_values = [reservation.get("liked_at") for reservation in profile.get("recent_likes", []) if reservation.get("liked_at")]
    if liked_at_values:
        recency_bonus = 0.05

    final_score = (
        AI_TUNABLE_WEIGHTS["text_similarity"] * text_score
        + AI_TUNABLE_WEIGHTS["category_match"] * category_score
        + AI_TUNABLE_WEIGHTS["time_match"] * time_score
        + AI_TUNABLE_WEIGHTS["recency_bonus"] * recency_bonus
    )
    return round(final_score, 4)


# -----------------------------------------------------------------------------
# Ranking API
# -----------------------------------------------------------------------------
def rank_reservations_for_user(
    liked_reservations: Iterable[Dict[str, Any]],
    candidate_reservations: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    profile = build_user_preference_vector(liked_reservations)
    profile["recent_likes"] = list(liked_reservations)

    scored = []
    for reservation in candidate_reservations:
        score = score_candidate_against_profile(reservation, profile)
        scored.append({
            "reservation": reservation,
            "score": score,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


# -----------------------------------------------------------------------------
# Example usage style for later integration with a DB or API layer.
# -----------------------------------------------------------------------------
def recommend_for_user(liked_reservations: Iterable[Dict[str, Any]], candidate_reservations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return rank_reservations_for_user(liked_reservations, candidate_reservations)


def build_ai_weight_prompt() -> str:
    return """
너는 추천 시스템 엔지니어다.

아래는 사용자 예약 좋아요 데이터와 추천 로직의 설명이다.
- 데이터는 사용자가 좋아요 누른 예약 목록이다.
- 추천 알고리즘은 content-based filtering 기반이다.
- 각 예약은 title, note, reservation_datetime, category, time_bucket를 가진다.
- 계산 점수는 text_similarity, category_match, time_match, recency_bonus를 사용한다.

현재 기본 가중치는:
{
  "text_similarity": 0.45,
  "category_match": 0.25,
  "time_match": 0.20,
  "recency_bonus": 0.10
}

작업:
1. 좋아요 데이터의 의미를 분석해라.
2. 사용자 선호 키워드와 시간대 패턴을 정리해라.
3. 추천 점수 가중치를 조정할 수 있는 개선안 3개를 제안해라.
4. 전처리 규칙에서 누락된 동의어/카테고리 정리를 제안해라.
5. 최종 결과는 JSON으로만 반환해라.
6. 수정은 embaded/algorithm.py 안에서 가능한 함수에 한정해라.
7. 코드는 다른 파일을 건드리지 마라.

출력 format:
{
  "suggested_weights": {...},
  "preprocessing_rules": [...],
  "reasoning": "..."
}
""".strip()
