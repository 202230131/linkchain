import hashlib
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from embaded.main import generate_chat_response, normalize_gemini_reply, search_text
from embaded.mcp import build_mcp_response

DB_URL = "postgresql://postgres:8888@localhost:5432/gemini_docs_3072"
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


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


Base.metadata.create_all(bind=engine)

app = FastAPI(title="LinkChain Personal Assistant")


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


@app.post("/mcp")
async def mcp(payload: dict):
    method = payload.get("method")
    if method == "tools/call":
        params = payload.get("params", {})
        arguments = params.get("arguments", {})
        question = arguments.get("question") or arguments.get("text") or arguments.get("query")
        if not question:
            raise HTTPException(status_code=400, detail="question text is required")
        return build_mcp_response(question)

    if "question" in payload:
        return build_mcp_response(payload["question"])

    raise HTTPException(status_code=400, detail="Unsupported MCP payload")
