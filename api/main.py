from fastapi import FastAPI, Request
from supabase import create_client
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
GPT_MODEL = "gpt-3.5-turbo"
EMBED_MODEL = "text-embedding-3-small"

def get_query_embedding(query: str):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    return response.data[0].embedding

@app.post("/api/main")
async def query_handler(request: Request):
    try:
        body = await request.json()
        query_text = body.get("query", "")
        session_id = body.get("session_id", "default_session")
        top_k = body.get("top_k", 5)

        # Get history from Supabase
        history_response = supabase \
            .from_("chat_history") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .limit(10) \
            .execute()

        history = history_response.data if history_response.data else []

        # Convert history to OpenAI messages format
        message_history = [{"role": h["role"], "content": h["content"]} for h in history]

        # Embedding + document match
        embedding = get_query_embedding(query_text)
        match_response = supabase.rpc("match_documents", {
            "query_embedding": embedding,
            "match_count": top_k
        }).execute()

        matches = match_response.data
        context_chunks = [r["content"] for r in matches]
        context = "\n\n".join(context_chunks)

        # Append current system and user prompt
        system_prompt = "You are a helpful assistant. Use the context to answer."
        message_history.insert(0, {"role": "system", "content": system_prompt})
        message_history.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"})

        # Call OpenAI
        completion = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=message_history
        )
        answer = completion.choices[0].message.content.strip()

        # Save user + assistant messages to Supabase
        supabase.table("chat_history").insert([
            {"session_id": session_id, "role": "user", "content": query_text},
            {"session_id": session_id, "role": "assistant", "content": answer}
        ]).execute()

        return {
            "answer": answer,
            "sources": [
                {"id": r["id"], "content": r["content"], "similarity": r["similarity"]} for r in matches
            ]
        }

    except Exception as e:
        return {"error": str(e)}
