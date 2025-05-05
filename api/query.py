# api/query.py
import os
import json
from supabase import create_client
import openai


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
GPT_MODEL = "text-embedding-3-small"

def get_query_embedding(query):
    response = openai.embeddings.create(
        model=GPT_MODEL,
        input=query
    )
    return response.data[0].embedding

def handler(request):
    try:
        body = json.loads(request.body.decode("utf-8"))
        query_text = body.get("query", "")
        top_k = body.get("top_k", 5)

        embedding = get_query_embedding(query_text)
        response = supabase.rpc("match_documents", {
            "query_embedding": embedding,
            "match_count": top_k
        }).execute()

        return {
            "statusCode": 200,
            "body": json.dumps({"results": [r["chunk_text"] for r in response.data]}),
            "headers": {"Content-Type": "application/json"}
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
