import os
from functools import lru_cache
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

def ensure_database_exists(db_name: str) -> None:
    admin_connection = "postgresql://postgres:8888@localhost:5432/postgres"
    with psycopg.connect(admin_connection) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{db_name}"')
                print(f"Created Postgres database: {db_name}")
            else:
                print(f"Database already exists: {db_name}")


def reset_stale_vector_tables(db_name: str) -> None:
    conn_str = f"postgresql://postgres:8888@localhost:5432/{db_name}"
    with psycopg.connect(conn_str) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding")
            cur.execute("DROP TABLE IF EXISTS langchain_pg_collection")
            print(f"Reset stale pgvector tables in database: {db_name}")
 
@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        api_key=api_key,
    )

    db_name = os.getenv("PGVECTOR_DB", "gemini_docs_3072")
    ensure_database_exists(db_name)
    if os.getenv("RESET_VECTOR_STORE", "true").lower() == "true":
        reset_stale_vector_tables(db_name)
    connection_string = f"postgresql://postgres:8888@localhost:5432/{db_name}"

    return PGVector(
        embeddings=embeddings,
        connection=connection_string,
        collection_name=os.getenv("PGVECTOR_COLLECTION", "gemini_documents_v2_3072"),
        embedding_length=3072,
        use_jsonb=True,
        create_extension=True,
    )

def add_texts(texts: list[str]) -> None:
    vector_store = get_vector_store()
    vector_store.add_texts(texts)


def store_chat_turn(user_message: str, assistant_message: str) -> None:
    vector_store = get_vector_store()
    vector_store.add_texts([
        f"[user] {user_message}",
        f"[assistant] {assistant_message}",
    ])


def get_chat_context(question: str, k: int = 5):
    try:
        docs = search_text(question, k=k)
    except Exception:
        return []
    return [doc.page_content for doc, _ in docs]


def get_chat_model():
    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
    return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0.7)


def normalize_gemini_reply(raw):
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(raw, dict):
        if "text" in raw:
            return str(raw["text"])
        if "content" in raw:
            return str(raw["content"])
        return str(raw)
    return str(raw)


def generate_chat_response(user_message: str) -> str:
    try:
        context = get_chat_context(user_message, k=3)
        llm = get_chat_model()

        context_block = ""
        if context:
            context_block = "이전 대화 기억:\n" + "\n".join(f"- {c}" for c in context)

        prompt = "\n\n".join(part for part in [context_block, f"사용자: {user_message}", "대답은 자연스럽고 친절하게 해줘."] if part)
        reply_obj = llm.invoke(prompt)
        reply = normalize_gemini_reply(reply_obj.content if hasattr(reply_obj, "content") else reply_obj)
        store_chat_turn(user_message, reply)
        return reply
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "429" in message or "resource_exhausted" in lowered or "quota" in lowered or "rate limit" in lowered:
            return "Gemini API 쿼터가 초과되었습니다. 잠시 후 다시 시도해 주세요."
        return f"Gemini 호출 중 오류가 발생했습니다: {message}"


def search_text(question: str, k: int = 3):
    vector_store = get_vector_store()
    return vector_store.similarity_search_with_score(question, k=k)