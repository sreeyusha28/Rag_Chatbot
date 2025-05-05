from fastapi import FastAPI, Request
from supabase import create_client
import openai
import os
from dotenv import load_dotenv  
from mangum import Mangum

load_dotenv()  


app = FastAPI()
handler = Mangum(app)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
GPT_MODEL = "text-embedding-3-small"

def get_query_embedding(query: str):
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

        matches = response.data
        context_chunks = [r["content"] for r in matches]
        context = "\n\n".join(context_chunks)

        system_prompt = "You are a helpful assistant. Use the context below to answer the user's question."
        user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        final_answer = completion.choices[0].message.content.strip()

        return {
            "answer": final_answer,
            "sources": [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "similarity": r["similarity"]
                } for r in matches
            ]
        }
    except Exception as e:
        return {"error": str(e)}
