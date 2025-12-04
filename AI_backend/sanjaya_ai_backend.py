import os
import sys
import json
import numpy as np
import traceback
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---

# 🚨 YOUR API KEY IS HERE 🚨
API_KEY = "API-KEY-HERE"

try:
    # Initialize the Gemini Client
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(json.dumps({"error": f"Failed to initialize Gemini Client: {str(e)}"}))
    sys.exit(1)

GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
EMBEDDING_MODEL = 'text-embedding-004' 
CHUNK_SEPARATOR = '\n---CHUNK_SEPARATOR---\n'

# Keep at 15 for good context, but we will limit the output style
K_NEAREST_NEIGHBORS = 15

# --- 2. ASSET LOADING ---

def load_assets():
    """Loads scripture chunks and FAISS index embeddings from files."""
    try:
        if not os.path.exists("scripture_chunks.txt"):
            return None, None

        with open("scripture_chunks.txt", 'r', encoding='utf-8') as f:
            chunks = f.read().split(CHUNK_SEPARATOR)
        
        import faiss
        if not os.path.exists("scripture_index.faiss"):
            return None, None
            
        index = faiss.read_index("scripture_index.faiss")
        
        return chunks, index
    except Exception:
        return None, None

# --- 3. CORE RAG FUNCTIONS ---

def get_embedding(text):
    """Generates an embedding vector using the Gemini API."""
    try:
        resp = client.models.embed_content(
            model=EMBEDDING_MODEL, 
            content=text, 
            config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
        )
        try:
            vec = resp.embeddings[0].values
        except (AttributeError, KeyError):
            vec = resp['embedding']
            
        return np.array(vec, dtype='float32').reshape(1, -1)
    except Exception:
        return None

def generate_answer(prompt, system_instruction):
    """Generates the final response using the Gemini API."""
    try:
        # Construct content correctly
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
        
        sys_instruction_content = types.Content(
            role="system",
            parts=[types.Part.from_text(text=system_instruction)]
        )

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_content],
            config=types.GenerateContentConfig(
                temperature=0.3,  # Lowered for more direct, less "flowery" answers
                max_output_tokens=2048, # Sufficient for a good answer, but not a novel
                system_instruction=sys_instruction_content
            )
        )
        return resp.text.strip()
    except Exception as e:
        return f"I am having trouble connecting to the AI Guru. Error: {str(e)}"

# --- 4. MAIN LOGIC (DIRECT MODE) ---

def process_query(user_query):
    # UPDATED: Persona strictly enforces directness and chat-style formatting
    system_persona = (
        "You are Sanjaya, a wise and concise AI guide on the Gita, Ramayana, and Mahabharata. "
        "Your goal is to answer the user's question directly and clearly. "
        "1. Start with a simple, respectful greeting (e.g., 'Namaste, seeker'). "
        "2. Immediately provide the direct answer to the question based on the scriptures. "
        "3. Do NOT use markdown headers (like ## or ###). Do NOT use hashtags. "
        "4. You may use bolding for key terms. "
        "5. Keep the tone helpful and conversational, not like a long sermon or essay. "
        "6. If you use the provided context, integrate it naturally without saying 'According to the context below'."
    )

    # 2. Try Retrieval (Local Search)
    chunks, index = load_assets()
    context_string = ""
    
    if chunks and index:
        emb = get_embedding(user_query)
        if emb is not None:
            D, I = index.search(emb, K_NEAREST_NEIGHBORS)
            found_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            if found_chunks:
                # We add a note to the prompt so the AI knows this is reference material
                context_string = "\n\n--- REFERENCE MATERIAL ---\n" + "\n---\n".join(found_chunks)

    # 3. Build Prompt
    final_prompt = (
        f"USER QUESTION: {user_query}\n\n"
        f"INSTRUCTION: Answer the user directly using your knowledge and the reference material below. "
        f"Avoid long introductions. Get straight to the point after a brief greeting."
        f"{context_string}"
    )

    # 4. Generate
    return generate_answer(final_prompt, system_persona)

# --- 5. EXECUTION ENTRY POINT ---

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"answer": "System Ready. Please provide a query argument."}))
        else:
            query = sys.argv[1]
            answer = process_query(query)
            print(json.dumps({"answer": answer}))
            
    except Exception as e:
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))
        sys.exit(1) 