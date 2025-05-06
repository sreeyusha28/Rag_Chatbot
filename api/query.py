from supabase import create_client
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
GPT_MODEL = "text-embedding-3-small"

def handler(request):
    try:
        body = json.loads(request.body.decode())
        query_text = body.get("query", "")
        top_k = body.get("top_k", 5)

        embedding = openai.embeddings.create(
            model=GPT_MODEL,
            input=query_text
        ).data[0].embedding

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
            "statusCode": 200,
            "body": json.dumps({
                "answer": final_answer,
                "sources": [
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "similarity": r["similarity"]
                    } for r in matches
                ]
            }),
            "headers": {"Content-Type": "application/json"}
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
