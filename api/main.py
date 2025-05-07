from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY

GPT_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

def get_query_embedding(query: str):
    response = openai.embeddings.create(
        model=GPT_MODEL,
        input=query
    )
    return response.data[0].embedding

def contextualize_question(chat_history, current_question):
    """Use chat history and user question to create a standalone question."""
    system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "rewrite it as a standalone question. Do NOT answer the question."
    )
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [
        {"role": "user", "content": current_question}
    ]
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content.strip()

def fetch_chat_history(session_id, limit=10):
    """Fetch latest N chat messages from Supabase for given session_id."""
    result = supabase.table("chat_history") \
        .select("role, content") \
        .eq("session_id", session_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    if result.data:
        return list(reversed(result.data))
    return []

def insert_chat_message(session_id, role, content):
    """Insert a new message into Supabase chat history."""
    supabase.table("chat_history").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

@app.post("/api/main")
async def query_handler(request: Request):
    try:
        body = await request.json()
        query_text = body.get("query", "")
        session_id = body.get("session_id", "default")
        top_k = body.get("top_k", 5)

        raw_history = fetch_chat_history(session_id)
        chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in raw_history]

        standalone_question = contextualize_question(chat_history, query_text)

        embedding = get_query_embedding(standalone_question)
        response = supabase.rpc("match_documents", {
            "query_embedding": embedding,
            "match_count": top_k
        }).execute()

        matches = response.data
        context_chunks = [r["content"] for r in matches]
        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a helpful assistant. Use the context below to answer the user's question. "
            "If the answer isn't in the context, say you don't know."
        )
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {standalone_question}"}]

        completion = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )
        final_answer = completion.choices[0].message.content.strip()

        insert_chat_message(session_id, "user", query_text)
        insert_chat_message(session_id, "assistant", final_answer)

        return {
            "answer": final_answer,
            "sources": [
                {"id": r["id"], "content": r["content"], "similarity": r["similarity"]}
                for r in matches
            ]
        }

    except Exception as e:
        return {"error": str(e)}
