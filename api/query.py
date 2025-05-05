from fastapi import FastAPI, Request
import json
from supabase import create_client
import openai
import os

app = FastAPI()

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

@app.post("/query")
async def query_handler(request: Request):
    try:
        body = await request.json()
        query_text = body.get("query", "")
        top_k = body.get("top_k", 5)

        embedding = get_query_embedding(query_text)
        response = supabase.rpc("match_documents", {
            "query_embedding": embedding,
            "match_count": top_k
        }).execute()

        return {"results": [r["chunk_text"] for r in response.data]}
    except Exception as e:
        return {"error": str(e)}

