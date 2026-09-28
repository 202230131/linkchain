import json


def build_mcp_response(question: str, k: int = 3):
    from embaded.main import search_text

    docs = search_text(question, k=k)
    results = []
    for doc, score in docs:
        results.append({
            "text": doc.page_content,
            "score": float(score),
        })

    payload = {
        "jsonrpc": "2.0",
        "id": "search_knowledge",
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "question": question,
                        "results": results,
                    }, ensure_ascii=False),
                }
            ],
            "structuredContent": {
                "question": question,
                "results": results,
            },
            "isError": False,
        },
    }
    return payload
