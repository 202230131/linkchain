import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from embaded.algorithm import rank_reservations_for_user
from embaded.main import generate_chat_response, normalize_gemini_reply, search_text
from embaded.mcp import (
    build_mcp_response,
    gmail_get_message,
    gmail_list_recent,
    google_drive_list,
    google_drive_search,
    google_search,
    google_tools_status,
)

def get_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


DB_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:8888@localhost:5432/gemini_docs_3072"
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI(title="LinkChain Personal Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Reservation(Base):
    __tablename__ = "reservations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    note = Column(Text, default="")
    reservation_datetime = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ReservationLike(Base):
    __tablename__ = "user_reservation_likes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    reservation_id = Column(Integer, ForeignKey("reservations.id"), nullable=False, index=True)
    liked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    source = Column(String(30), default="button", nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "reservation_id", name="uq_user_reservation_like"),
    )


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_current_user_id(request: Request) -> int:
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    try:
        return int(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="유효하지 않은 로그인 정보입니다.") from exc


def parse_korean_date_time(message: str):
    text = message.strip()
    if not text:
        return None

    lowered = text.lower()
    if not any(keyword in lowered for keyword in ["예약", "약속", "일정", "시간", "날짜", "미팅", "회의", "상담", "점심", "저녁"]):
        return None

    now = datetime.now()
    date_value = now
    time_value = "09:00"

    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[T\s]\d{1,2}:\d{2}", text):
        candidate = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[T\s]\d{1,2}:\d{2}", text).group(0)
        normalized = candidate.replace(" ", "T")
        try:
            return datetime.fromisoformat(normalized), text
        except ValueError:
            pass

    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        candidate = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text).group(0)
        try:
            date_value = datetime.fromisoformat(candidate)
        except ValueError:
            try:
                date_value = datetime.strptime(candidate, "%Y/%m/%d")
            except ValueError:
                date_value = now

    elif "오늘" in text:
        date_value = now
    elif "내일" in text:
        date_value = now + timedelta(days=1)
    elif "모레" in text:
        date_value = now + timedelta(days=2)
    elif "다음주" in text or "다음 주" in text:
        date_value = now + timedelta(days=7)
    else:
        weekday_map = {
            "월요일": 0,
            "화요일": 1,
            "수요일": 2,
            "목요일": 3,
            "금요일": 4,
            "토요일": 5,
            "일요일": 6,
        }
        for name, offset in weekday_map.items():
            if name in text:
                days_until = (offset - now.weekday()) % 7
                if days_until == 0:
                    days_until = 7
                date_value = now + timedelta(days=days_until)
                break

    am_pm = None
    if "오전" in text:
        am_pm = "am"
    elif "오후" in text:
        am_pm = "pm"

    if re.search(r"\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?", text):
        time_match = re.search(r"\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?", text)
        clock_text = time_match.group(0)
        hour = int(re.search(r"\d{1,2}", clock_text).group(0))
        minute_match = re.search(r"(\d{1,2})\s*분", clock_text)
        minute = int(minute_match.group(1)) if minute_match else 0
        if am_pm == "pm" and 1 <= hour <= 11:
            hour += 12
        elif am_pm == "am" and hour == 12:
            hour = 0
        time_value = f"{hour:02d}:{minute:02d}"
    elif re.search(r"\d{1,2}:\d{2}", text):
        time_value = re.search(r"\d{1,2}:\d{2}", text).group(0)

    if re.search(r"\d{1,2}\s*시\s*\d{1,2}\s*분", text):
        time_match = re.search(r"\d{1,2}\s*시\s*\d{1,2}\s*분", text)
        hour = int(re.search(r"\d{1,2}", time_match.group(0)).group(0))
        minute = int(re.search(r"(\d{1,2})\s*분", time_match.group(0)).group(1))
        if am_pm == "pm" and 1 <= hour <= 11:
            hour += 12
        elif am_pm == "am" and hour == 12:
            hour = 0
        time_value = f"{hour:02d}:{minute:02d}"

    try:
        dt = datetime.combine(date_value.date(), datetime.strptime(time_value, "%H:%M").time())
    except ValueError:
        dt = datetime.combine(date_value.date(), datetime.strptime("09:00", "%H:%M").time())

    return dt, text


def looks_like_booking_intent(message: str) -> bool:
    text = message.strip()
    if not text:
        return False

    normalized = text.lower()
    booking_patterns = [
        r"예약\s*(해|해줘|해줄래|해도|해도돼|부탁|할게|등록|잡|잡아|잡을래)",
        r"약속\s*(잡|잡아|잡을래|등록|해줘)",
        r"(미팅|회의|상담|점심|저녁)\s*(예약|잡|잡아|등록)",
        r"(일정|시간|날짜)\s*(잡|잡아|정해|정해줘|예약)",
        r"(오늘|내일|모레|다음주|금요일|월요일|화요일|수요일|목요일|토요일|일요일).*\s*(예약|약속|일정)",
    ]

    for pattern in booking_patterns:
        if re.search(pattern, normalized):
            return True

    return False


def maybe_create_reservation_from_message(user_id: int, message: str):
    parsed = parse_korean_date_time(message)
    if parsed is None:
        return None

    target_datetime, raw_text = parsed
    title = "예약"
    if "미팅" in message:
        title = "미팅"
    elif "상담" in message:
        title = "상담"
    elif "회의" in message:
        title = "회의"
    elif "점심" in message:
        title = "점심 약속"
    elif "저녁" in message:
        title = "저녁 약속"

    db = SessionLocal()
    try:
        reservation = Reservation(
            user_id=user_id,
            title=title,
            note=raw_text,
            reservation_datetime=target_datetime,
            status="pending",
        )
        db.add(reservation)
        db.commit()
        db.refresh(reservation)
        return {
            "id": reservation.id,
            "title": reservation.title,
            "reservation_datetime": reservation.reservation_datetime.isoformat(),
            "status": reservation.status,
        }
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).with_name("service.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/google/status")
async def google_status():
    return google_tools_status()


@app.get("/google/connect")
async def google_connect():
    config = google_tools_status()
    if not config["ready"]["oauth"]:
        return {
            "status": "not_configured",
            "message": "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI를 설정해야 OAuth 연결을 시작할 수 있습니다.",
        }

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    scope = "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/customsearch"
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        + f"?client_id={client_id}"
        + f"&redirect_uri={redirect_uri}"
        + "&response_type=code"
        + "&scope=" + "%20".join(scope.split())
        + "&access_type=offline"
        + "&prompt=consent"
    )
    return {"status": "ok", "oauth_url": auth_url}


@app.get("/google/callback")
async def google_callback(request: Request, code: str | None = None):
    if not code:
        raise HTTPException(status_code=400, detail="Google OAuth code가 없습니다.")

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=500, detail="Google OAuth 환경 변수가 설정되지 않았습니다.")

    payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data=payload,
            timeout=20,
        )
        response.raise_for_status()
        token_json = response.json()
        access_token = token_json.get("access_token")
        if access_token:
            redirect = RedirectResponse(url="/", status_code=302)
            redirect.set_cookie("google_access_token", access_token, httponly=True, samesite="lax")
            return redirect
        return {"status": "error", "detail": "Google 토큰을 받지 못했습니다.", "raw": token_json}
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Google OAuth 토큰 교환 실패: {exc}") from exc


@app.post("/signup")
async def signup(payload: dict):
    name = (payload or {}).get("name", "").strip()
    email = (payload or {}).get("email", "").strip().lower()
    password = (payload or {}).get("password", "")

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="name, email, password는 모두 필요합니다.")

    db = SessionLocal()
    try:
        exists = db.query(User).filter(User.email == email).first()
        if exists:
            raise HTTPException(status_code=409, detail="이미 가입된 이메일입니다.")

        user = User(name=name, email=email, password_hash=hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        }
    finally:
        db.close()


@app.post("/login")
async def login(payload: dict, response: Response):
    email = (payload or {}).get("email", "").strip().lower()
    password = (payload or {}).get("password", "")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or user.password_hash != hash_password(password):
            raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")

        response.set_cookie(key="user_id", value=str(user.id), httponly=True, samesite="lax")
        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        }
    finally:
        db.close()


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="user_id")
    return {"ok": True}


@app.get("/me")
async def me(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            raise HTTPException(status_code=401, detail="유효하지 않은 사용자입니다.")
        return {"id": user.id, "name": user.name, "email": user.email}
    finally:
        db.close()


@app.get("/reservations")
async def get_reservations(request: Request):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        rows = db.query(Reservation).filter(Reservation.user_id == user_id).order_by(Reservation.reservation_datetime.asc()).all()
        return {
            "reservations": [
                {
                    "id": r.id,
                    "title": r.title,
                    "note": r.note,
                    "reservation_datetime": r.reservation_datetime.isoformat() if r.reservation_datetime else None,
                    "status": r.status,
                }
                for r in rows
            ]
        }
    finally:
        db.close()


@app.get("/reservations/recommendations")
async def get_recommendations(request: Request):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        liked_rows = db.query(ReservationLike, Reservation).join(Reservation, Reservation.id == ReservationLike.reservation_id).filter(
            ReservationLike.user_id == user_id
        ).order_by(ReservationLike.liked_at.desc()).all()

        liked_reservations = [
            {
                "id": row.Reservation.id,
                "title": row.Reservation.title,
                "note": row.Reservation.note,
                "reservation_datetime": row.Reservation.reservation_datetime.isoformat() if row.Reservation.reservation_datetime else None,
                "status": row.Reservation.status,
                "liked_at": row.ReservationLike.liked_at.isoformat() if row.ReservationLike.liked_at else None,
            }
            for row in liked_rows
        ]

        candidate_rows = db.query(Reservation).filter(Reservation.user_id == user_id).order_by(Reservation.reservation_datetime.asc()).all()
        candidate_reservations = [
            {
                "id": r.id,
                "title": r.title,
                "note": r.note,
                "reservation_datetime": r.reservation_datetime.isoformat() if r.reservation_datetime else None,
                "status": r.status,
            }
            for r in candidate_rows
        ]

        liked_ids = {reservation["id"] for reservation in liked_reservations}
        scored = rank_reservations_for_user(liked_reservations, candidate_reservations)
        recommendations = []
        for item in scored:
            reservation = item["reservation"]
            reasons = []
            if item["score"] >= 0.6:
                reasons.append("높은 선호도")
            if reservation.get("reservation_datetime"):
                reasons.append("시간대 선호")
            if reasons == []:
                reasons.append("일반 추천")

            recommendations.append({
                "id": reservation["id"],
                "title": reservation["title"],
                "note": reservation["note"],
                "reservation_datetime": reservation.get("reservation_datetime"),
                "status": reservation.get("status"),
                "score": float(item["score"]),
                "liked": reservation["id"] in liked_ids,
                "reasons": reasons,
            })

        return {"reservations": recommendations}
    finally:
        db.close()


@app.get("/recommendations/weights/prompt")
async def get_ai_weight_prompt():
    return {
        "prompt": """
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
    }


@app.post("/reservations/{reservation_id}/like")
async def like_reservation(request: Request, reservation_id: int):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        reservation = db.query(Reservation).filter(Reservation.id == reservation_id, Reservation.user_id == user_id).first()
        if not reservation:
            raise HTTPException(status_code=404, detail="예약을 찾을 수 없습니다.")

        like = db.query(ReservationLike).filter(ReservationLike.user_id == user_id, ReservationLike.reservation_id == reservation_id).first()
        if like:
            return {"ok": True, "liked": True, "reservation_id": reservation_id}

        db.add(ReservationLike(user_id=user_id, reservation_id=reservation_id, source="button"))
        db.commit()
        return {"ok": True, "liked": True, "reservation_id": reservation_id}
    finally:
        db.close()


@app.delete("/reservations/{reservation_id}/like")
async def unlike_reservation(request: Request, reservation_id: int):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        like = db.query(ReservationLike).filter(ReservationLike.user_id == user_id, ReservationLike.reservation_id == reservation_id).first()
        if like:
            db.delete(like)
            db.commit()
        return {"ok": True, "liked": False, "reservation_id": reservation_id}
    finally:
        db.close()


@app.put("/reservations/{reservation_id}")
async def update_reservation(request: Request, reservation_id: int, payload: dict):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        reservation = db.query(Reservation).filter(Reservation.id == reservation_id, Reservation.user_id == user_id).first()
        if not reservation:
            raise HTTPException(status_code=404, detail="예약을 찾을 수 없습니다.")

        title = (payload or {}).get("title", reservation.title).strip() or reservation.title
        note = (payload or {}).get("note", reservation.note) or reservation.note
        reservation_datetime_raw = (payload or {}).get("reservation_datetime")
        status = (payload or {}).get("status", reservation.status)

        if reservation_datetime_raw:
            try:
                reservation.reservation_datetime = datetime.fromisoformat(reservation_datetime_raw)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="reservation_datetime 형식이 잘못되었습니다.") from exc

        reservation.title = title
        reservation.note = note
        reservation.status = status
        db.commit()
        db.refresh(reservation)
        return {
            "id": reservation.id,
            "title": reservation.title,
            "note": reservation.note,
            "reservation_datetime": reservation.reservation_datetime.isoformat(),
            "status": reservation.status,
        }
    finally:
        db.close()


@app.delete("/reservations/{reservation_id}")
async def delete_reservation(request: Request, reservation_id: int):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        reservation = db.query(Reservation).filter(Reservation.id == reservation_id, Reservation.user_id == user_id).first()
        if not reservation:
            raise HTTPException(status_code=404, detail="예약을 찾을 수 없습니다.")
        db.delete(reservation)
        db.commit()
        return {"ok": True, "deleted_id": reservation_id}
    finally:
        db.close()


@app.post("/reservations")
async def create_reservation(request: Request, payload: dict):
    user_id = get_current_user_id(request)
    title = (payload or {}).get("title", "").strip() or "새 예약"
    note = (payload or {}).get("note", "")
    reservation_datetime_raw = (payload or {}).get("reservation_datetime")

    if not reservation_datetime_raw:
        raise HTTPException(status_code=400, detail="reservation_datetime이 필요합니다.")

    try:
        reservation_datetime = datetime.fromisoformat(reservation_datetime_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="reservation_datetime 형식이 잘못되었습니다.") from exc

    db = SessionLocal()
    try:
        reservation = Reservation(
            user_id=user_id,
            title=title,
            note=note,
            reservation_datetime=reservation_datetime,
            status="pending",
        )
        db.add(reservation)
        db.commit()
        db.refresh(reservation)
        return {
            "id": reservation.id,
            "title": reservation.title,
            "note": reservation.note,
            "reservation_datetime": reservation.reservation_datetime.isoformat(),
            "status": reservation.status,
        }
    finally:
        db.close()


@app.get("/chat/history")
async def get_chat_history(request: Request):
    user_id = get_current_user_id(request)
    db = SessionLocal()
    try:
        rows = db.query(ChatMessage).filter(ChatMessage.user_id == user_id).order_by(ChatMessage.created_at.asc()).all()
        return {
            "messages": [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }
    finally:
        db.close()


@app.post("/ask")
async def ask(question: str = ""):
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")
    results = search_text(question, k=3)
    return {
        "question": question,
        "results": [
            {"text": doc.page_content, "score": float(score)}
            for doc, score in results
        ],
    }


@app.post("/chat")
async def chat(request: Request, payload: dict):
    user_id = get_current_user_id(request)
    message = None
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("question") or payload.get("text")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    if looks_like_booking_intent(message):
        reply = "예약은 예약 탭에서만 등록할 수 있어요. 대화창에서는 일반 질문이나 상담만 가능합니다."
        db = SessionLocal()
        try:
            db.add(ChatMessage(user_id=user_id, role="user", content=message))
            db.add(ChatMessage(user_id=user_id, role="assistant", content=reply))
            db.commit()
        finally:
            db.close()
        return {"message": message, "reply": reply}

    raw_reply = generate_chat_response(message)
    reply_text = normalize_gemini_reply(raw_reply)

    db = SessionLocal()
    try:
        db.add(ChatMessage(user_id=user_id, role="user", content=message))
        db.add(ChatMessage(user_id=user_id, role="assistant", content=reply_text))
        db.commit()
    finally:
        db.close()

    return {"message": message, "reply": reply_text}


def serialize_reservation_row(row):
    return {
        "id": row.id,
        "title": row.title,
        "note": row.note,
        "reservation_datetime": row.reservation_datetime.isoformat() if row.reservation_datetime else None,
        "status": row.status,
    }


@app.post("/mcp")
async def mcp(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="MCP payload must be an object")

    method = payload.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id", "initialize"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "linkchain-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        from embaded.mcp import list_mcp_tools
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id", "tools/list"),
            "result": list_mcp_tools(),
        }

    if method == "tools/call":
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be an object")

        tool_name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=400, detail="arguments must be an object")

        user_id = get_current_user_id(request)
        db = SessionLocal()
        try:
            if tool_name == "search_knowledge":
                question = arguments.get("question") or arguments.get("text") or arguments.get("query")
                if not question:
                    raise HTTPException(status_code=400, detail="question text is required")
                return build_mcp_response(question, k=int(arguments.get("k", 3)))

            if tool_name == "google_search":
                query = arguments.get("query") or arguments.get("text")
                if not query:
                    raise HTTPException(status_code=400, detail="query is required")
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(google_search(query, num=int(arguments.get("num", 5))), ensure_ascii=False)}],
                        "structuredContent": google_search(query, num=int(arguments.get("num", 5))),
                        "isError": False,
                    },
                }

            if tool_name == "google_drive_list":
                access_token = arguments.get("access_token") or request.cookies.get("google_access_token")
                result = google_drive_list(access_token=access_token, folder_id=arguments.get("folder_id"), page_size=int(arguments.get("page_size", 10)))
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                        "structuredContent": result,
                        "isError": False,
                    },
                }

            if tool_name == "google_drive_search":
                access_token = arguments.get("access_token") or request.cookies.get("google_access_token")
                query = arguments.get("query")
                if not query:
                    raise HTTPException(status_code=400, detail="query is required")
                result = google_drive_search(access_token=access_token, query=query, page_size=int(arguments.get("page_size", 10)))
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                        "structuredContent": result,
                        "isError": False,
                    },
                }

            if tool_name == "gmail_list_recent":
                access_token = arguments.get("access_token") or request.cookies.get("google_access_token")
                result = gmail_list_recent(access_token=access_token, max_results=int(arguments.get("max_results", 5)))
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                        "structuredContent": result,
                        "isError": False,
                    },
                }

            if tool_name == "gmail_get_message":
                access_token = arguments.get("access_token") or request.cookies.get("google_access_token")
                message_id = arguments.get("message_id")
                if not message_id:
                    raise HTTPException(status_code=400, detail="message_id is required")
                result = gmail_get_message(access_token=access_token, message_id=message_id)
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                        "structuredContent": result,
                        "isError": False,
                    },
                }

            if tool_name == "get_my_reservations":
                rows = db.query(Reservation).filter(Reservation.user_id == user_id).order_by(Reservation.reservation_datetime.asc()).all()
                reservations = [serialize_reservation_row(row) for row in rows]
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"reservations": reservations}, ensure_ascii=False)}],
                        "structuredContent": {"reservations": reservations},
                        "isError": False,
                    },
                }

            if tool_name == "create_reservation":
                title = str(arguments.get("title", "")).strip() or "새 예약"
                note = arguments.get("note") or ""
                reservation_datetime_raw = arguments.get("reservation_datetime")
                if not reservation_datetime_raw:
                    raise HTTPException(status_code=400, detail="reservation_datetime is required")
                try:
                    reservation_datetime = datetime.fromisoformat(str(reservation_datetime_raw))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="reservation_datetime 형식이 잘못되었습니다.") from exc

                reservation = Reservation(user_id=user_id, title=title, note=str(note), reservation_datetime=reservation_datetime, status="pending")
                db.add(reservation)
                db.commit()
                db.refresh(reservation)
                payload_data = {"reservation": serialize_reservation_row(reservation)}
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload_data, ensure_ascii=False)}],
                        "structuredContent": payload_data,
                        "isError": False,
                    },
                }

            if tool_name == "delete_reservation":
                reservation_id = int(arguments.get("reservation_id"))
                reservation = db.query(Reservation).filter(Reservation.id == reservation_id, Reservation.user_id == user_id).first()
                if not reservation:
                    raise HTTPException(status_code=404, detail="예약을 찾을 수 없습니다.")
                db.delete(reservation)
                db.commit()
                return {
                    "jsonrpc": "2.0",
                    "id": payload.get("id", tool_name),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"deleted_id": reservation_id}, ensure_ascii=False)}],
                        "structuredContent": {"deleted_id": reservation_id},
                        "isError": False,
                    },
                }

            raise HTTPException(status_code=404, detail=f"Unsupported MCP tool: {tool_name}")
        finally:
            db.close()

    if "question" in payload:
        return build_mcp_response(payload["question"])

    raise HTTPException(status_code=400, detail="Unsupported MCP payload")
