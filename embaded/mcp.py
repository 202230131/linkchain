import json
import os

import requests


def get_google_client_config():
    search_provider = (os.getenv("SEARCH_PROVIDER") or "brave").strip().lower()
    return {
        "search_provider": search_provider,
        "google_client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "google_client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "google_redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
        "google_cse_api_key": os.getenv("GOOGLE_CSE_API_KEY"),
        "google_cse_engine_id": os.getenv("GOOGLE_CSE_ENGINE_ID"),
        "brave_api_key": os.getenv("BRAVE_API_KEY"),
        "google_oauth_enabled": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET") and os.getenv("GOOGLE_REDIRECT_URI")),
    }


def google_tools_status():
    config = get_google_client_config()
    brave_ready = bool(config["brave_api_key"])
    cse_ready = bool(config["google_cse_api_key"] and config["google_cse_engine_id"])
    ready = {
        "search": brave_ready or cse_ready,
        "oauth": config["google_oauth_enabled"],
        "drive": config["google_oauth_enabled"],
        "gmail": config["google_oauth_enabled"],
    }
    return {
        "status": "ok" if any(ready.values()) else "not_configured",
        "ready": ready,
        "config": config,
        "message": "Google OAuth는 유지하고, 웹 검색은 Brave Search 또는 Google CSE 중 하나를 사용합니다.",
    }


def mcp_result(payload, request_id="search_knowledge"):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ],
            "structuredContent": payload,
            "isError": False,
        },
    }


def list_mcp_tools():
    return {
        "tools": [
            {
                "name": "search_knowledge",
                "description": "검색 문맥을 조회해 관련 지식을 찾습니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "get_my_reservations",
                "description": "현재 로그인한 사용자의 예약 목록을 조회합니다.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "create_reservation",
                "description": "현재 사용자의 예약을 생성합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "note": {"type": "string"},
                        "reservation_datetime": {"type": "string"},
                    },
                    "required": ["title", "reservation_datetime"],
                },
            },
            {
                "name": "delete_reservation",
                "description": "현재 사용자의 예약을 삭제합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"reservation_id": {"type": "integer"}},
                    "required": ["reservation_id"],
                },
            },
            {
                "name": "google_search",
                "description": "Google Custom Search를 사용해 웹 검색 결과를 조회합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "google_drive_list",
                "description": "Google Drive 파일 목록을 조회합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "folder_id": {"type": "string"},
                        "page_size": {"type": "integer", "default": 10},
                        "access_token": {"type": "string"},
                    },
                },
            },
            {
                "name": "google_drive_search",
                "description": "Google Drive에서 파일을 검색합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "page_size": {"type": "integer", "default": 10},
                        "access_token": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "gmail_list_recent",
                "description": "최근 Gmail 메시지를 조회합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_results": {"type": "integer", "default": 5},
                        "access_token": {"type": "string"},
                    },
                },
            },
            {
                "name": "gmail_get_message",
                "description": "특정 Gmail 메시지 내용을 조회합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"},
                        "access_token": {"type": "string"},
                    },
                    "required": ["message_id"],
                },
            },
        ]
    }


def google_search(query: str, num: int = 5):
    provider = (os.getenv("SEARCH_PROVIDER") or "brave").strip().lower()

    if provider == "google" or provider == "cse":
        api_key = os.getenv("GOOGLE_CSE_API_KEY")
        engine_id = os.getenv("GOOGLE_CSE_ENGINE_ID")
        if not api_key or not engine_id:
            return {
                "status": "not_configured",
                "message": "GOOGLE_CSE_API_KEY와 GOOGLE_CSE_ENGINE_ID를 설정해야 Google Search를 사용할 수 있습니다.",
            }

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": min(max(num, 1), 10),
        }
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items", [])
            return {
                "status": "ok",
                "provider": "google_cse",
                "query": query,
                "results": [
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                    }
                    for item in items
                ],
            }
        except requests.RequestException as exc:
            return {"status": "error", "provider": "google_cse", "message": f"Google Search 요청 실패: {exc}"}

    brave_api_key = os.getenv("BRAVE_API_KEY")
    if not brave_api_key:
        return {
            "status": "not_configured",
            "provider": "brave",
            "message": "BRAVE_API_KEY를 설정해야 웹 검색을 사용할 수 있습니다. 추천 설정: SEARCH_PROVIDER=brave",
        }

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key,
    }
    params = {"q": query, "count": min(max(num, 1), 10)}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("web", {}).get("results", [])
        return {
            "status": "ok",
            "provider": "brave",
            "query": query,
            "results": [
                {
                    "title": item.get("title"),
                    "link": item.get("url"),
                    "snippet": item.get("description"),
                }
                for item in items
            ],
        }
    except requests.RequestException as exc:
        return {"status": "error", "provider": "brave", "message": f"Brave Search 요청 실패: {exc}"}


def google_drive_list(access_token: str | None = None, folder_id: str | None = None, page_size: int = 10):
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if not token:
        return {"status": "not_configured", "message": "Google Drive를 사용하려면 OAuth access_token이 필요합니다."}

    params = {"pageSize": min(max(page_size, 1), 20), "fields": "files(id,name,mimeType,webViewLink)"}
    if folder_id:
        params["q"] = f"' {folder_id} ' in parents"

    try:
        response = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return {"status": "ok", "files": data.get("files", [])}
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Google Drive 요청 실패: {exc}"}


def google_drive_search(access_token: str | None = None, query: str = "", page_size: int = 10):
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if not token:
        return {"status": "not_configured", "message": "Google Drive를 사용하려면 OAuth access_token이 필요합니다."}
    if not query:
        return {"status": "error", "message": "query is required"}

    try:
        response = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params={
                "q": f"name contains '{query}'",
                "pageSize": min(max(page_size, 1), 20),
                "fields": "files(id,name,mimeType,webViewLink)",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return {"status": "ok", "query": query, "files": data.get("files", [])}
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Google Drive 검색 실패: {exc}"}


def gmail_list_recent(access_token: str | None = None, max_results: int = 5):
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if not token:
        return {"status": "not_configured", "message": "Gmail을 사용하려면 OAuth access_token이 필요합니다."}

    try:
        response = requests.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            params={"maxResults": min(max(max_results, 1), 20)},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return {"status": "ok", "messages": data.get("messages", [])}
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Gmail 목록 조회 실패: {exc}"}


def gmail_get_message(access_token: str | None = None, message_id: str = ""):
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if not token:
        return {"status": "not_configured", "message": "Gmail을 사용하려면 OAuth access_token이 필요합니다."}
    if not message_id:
        return {"status": "error", "message": "message_id is required"}

    try:
        response = requests.get(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={"format": "full"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return {"status": "ok", "message": data}
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Gmail 메시지 조회 실패: {exc}"}


def build_mcp_response(question: str, k: int = 3):
    from embaded.main import search_text

    docs = search_text(question, k=k)
    results = []
    for doc, score in docs:
        results.append({
            "text": doc.page_content,
            "score": float(score),
        })

    return mcp_result({
        "question": question,
        "results": results,
    }, request_id="search_knowledge")
