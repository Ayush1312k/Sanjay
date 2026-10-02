import os
import sys
import json
import re
import numpy as np
from dotenv import load_dotenv

# Load environment variables: check local script dir first, then default
script_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(script_dir, '.env')
if os.path.exists(env_file):
    load_dotenv(env_file)
else:
    load_dotenv()

# --- 1. CONFIGURATION & CLIENT INITIALIZATION ---

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def sanitize_secret(text):
    """Sanitizes potential API keys and secrets from text before logging."""
    if not text:
        return ""
    text_str = str(text)
    # Mask Gemini API keys (starts with AIza...)
    text_str = re.sub(r'AIza[0-9A-Za-z-_]{35}', '[REDACTED_GEMINI_KEY]', text_str)
    # Mask Anthropic API keys (starts with sk-ant-...)
    text_str = re.sub(r'sk-ant-[0-9A-Za-z-_]+', '[REDACTED_ANTHROPIC_KEY]', text_str)
    return text_str

# Initialize Gemini Client
gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to initialize Gemini Client: {sanitize_secret(str(e))}\n")

# Initialize Anthropic Client
claude_client = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to initialize Anthropic Client: {sanitize_secret(str(e))}\n")

GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/gemini-embedding-001')
CHUNK_SEPARATOR = '\n---CHUNK_SEPARATOR---\n'
K_NEAREST_NEIGHBORS = 15

# --- 2. ASSET LOADING WITH IN-MEMORY CACHE ---

_CACHED_CHUNKS = None
_CACHED_INDEX = None

def load_assets():
    """Loads scripture chunks and FAISS index embeddings from files with in-memory caching."""
    global _CACHED_CHUNKS, _CACHED_INDEX
    if _CACHED_CHUNKS is not None and _CACHED_INDEX is not None:
        return _CACHED_CHUNKS, _CACHED_INDEX

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        chunks_path = os.path.join(current_dir, "scripture_chunks.txt")
        index_path = os.path.join(current_dir, "scripture_index.faiss")

        if not os.path.exists(chunks_path):
            sys.stderr.write("Notice: scripture_chunks.txt not found.\n")
            return None, None

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = f.read().split(CHUNK_SEPARATOR)
        
        if not os.path.exists(index_path):
            sys.stderr.write("Notice: scripture_index.faiss not found.\n")
            return None, None

        import faiss
        index = faiss.read_index(index_path)
        
        _CACHED_CHUNKS = chunks
        _CACHED_INDEX = index
        return chunks, index
    except Exception as e:
        sys.stderr.write(f"Asset loading error: {sanitize_secret(str(e))}\n")
        return None, None

# --- 3. CORE RAG & MODEL GENERATION FUNCTIONS ---

def get_embedding(text):
    """Generates an embedding vector using the Gemini API."""
    if not gemini_client:
        return None
    try:
        from google.genai import types
        resp = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL, 
            contents=text, 
            config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
        )
        try:
            vec = resp.embeddings[0].values
        except (AttributeError, KeyError):
            vec = resp['embedding']
            
        return np.array(vec, dtype='float32').reshape(1, -1)
    except Exception as e:
        sys.stderr.write(f"Embedding error: {sanitize_secret(str(e))}\n")
        return None

def generate_gemini_answer(prompt, system_instruction):
    """Generates response using Gemini API."""
    if not gemini_client:
        return None
    try:
        from google.genai import types
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
        
        sys_instruction_content = types.Content(
            role="system",
            parts=[types.Part.from_text(text=system_instruction)]
        )

        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_content],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                system_instruction=sys_instruction_content
            )
        )
        return resp.text.strip() if resp.text else None
    except Exception as e:
        sys.stderr.write(f"Gemini Generation Error: {sanitize_secret(str(e))}\n")
        return None

def generate_claude_answer(prompt, system_instruction):
    """Generates response using Anthropic Claude API."""
    if not claude_client:
        return None
    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=system_instruction,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        sys.stderr.write(f"Claude Generation Error: {sanitize_secret(str(e))}\n")
        return None

def merge_answers(user_query, gemini_answer, claude_answer, system_persona):
    """Merges outputs from both Gemini and Claude into a single optimal answer."""
    synth_prompt = (
        f"USER QUESTION: {user_query}\n\n"
        f"CANDIDATE RESPONSE A:\n{gemini_answer}\n\n"
        f"CANDIDATE RESPONSE B:\n{claude_answer}\n\n"
        f"INSTRUCTION:\n"
        f"Synthesize CANDIDATE RESPONSE A and CANDIDATE RESPONSE B into a single, cohesive, optimal answer.\n"
        f"Combine the best insights, scripture citations, and wisdom from both candidates.\n"
        f"Do NOT reference Candidate A, Candidate B, Gemini, or Claude. Output a single seamless answer.\n"
        f"Ensure there are no repeated paragraphs or redundant phrasing.\n"
        f"Start with a simple, respectful greeting."
    )
    
    # Attempt synthesis using Gemini first
    merged = generate_gemini_answer(synth_prompt, system_persona)
    if merged:
        return merged
        
    # Fallback to Claude for synthesis
    merged = generate_claude_answer(synth_prompt, system_persona)
    if merged:
        return merged
        
    # If both synthesis calls fail, return the clearer/longer answer or both
    return gemini_answer if len(gemini_answer) >= len(claude_answer) else claude_answer

# --- 4. MAIN LOGIC (DIRECT MODE) ---

def process_query(user_query):
    """Processes user query through RAG retrieval and dual-model generation."""
    if not gemini_client and not claude_client:
        return "Service unavailable: API credentials are not configured on the server."

    system_persona = (
        "You are Sajay, a wise and concise AI guide on the Gita, Ramayana, and Mahabharata. "
        "Your goal is to answer the user's question directly and clearly. "
        "1. Start with a simple, respectful greeting (e.g., 'Namaste, seeker'). "
        "2. Immediately provide the direct answer to the question based on the scriptures. "
        "3. Do NOT use markdown headers (like ## or ###). Do NOT use hashtags. "
        "4. You may use bolding for key terms. "
        "5. Keep the tone helpful and conversational, not like a long sermon or essay. "
        "6. If you use the provided context, integrate it naturally without saying 'According to the context below'."
    )

    # 1. Try Retrieval (Local FAISS Search)
    chunks, index = load_assets()
    context_string = ""
    
    if chunks and index:
        emb = get_embedding(user_query)
        if emb is not None and emb.shape[1] == index.d:
            D, I = index.search(emb, K_NEAREST_NEIGHBORS)
            found_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            if found_chunks:
                context_string = "\n\n--- REFERENCE MATERIAL ---\n" + "\n---\n".join(found_chunks)

    # 2. Build Prompt
    final_prompt = (
        f"USER QUESTION: {user_query}\n\n"
        f"INSTRUCTION: Answer the user directly using your knowledge and the reference material below. "
        f"Avoid long introductions. Get straight to the point after a brief greeting."
        f"{context_string}"
    )

    # 3. Generate answers from both models
    gemini_resp = generate_gemini_answer(final_prompt, system_persona)
    claude_resp = generate_claude_answer(final_prompt, system_persona)

    # 4. Merge or return available answer
    if gemini_resp and claude_resp:
        return merge_answers(user_query, gemini_resp, claude_resp, system_persona)
    elif gemini_resp:
        return gemini_resp
    elif claude_resp:
        return claude_resp
    else:
        return "I am having trouble connecting to the AI Guru services. Please check your network and API configuration."

# --- 5. STANDALONE HTTP SERVER MODE ---

def run_standalone_server(port=5000):
    """Starts a standalone HTTP server for Python hosting platforms (Render, Railway, Fly.io, etc.)."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    # Pre-cache assets at startup
    chunks, index = load_assets()
    if chunks and index:
        sys.stdout.write(f"Loaded {len(chunks)} scripture chunks and FAISS index into memory.\n")

    frontend_origin = os.getenv("FRONTEND_URL", "*")

    class SanjayaHandler(BaseHTTPRequestHandler):
        def _set_cors(self):
            origin = self.headers.get("Origin", "")
            allowed = "*"
            if frontend_origin != "*":
                origins_list = [o.strip() for o in frontend_origin.split(",")]
                if origin in origins_list or "localhost" in origin or "127.0.0.1" in origin:
                    allowed = origin
            self.send_header("Access-Control-Allow-Origin", allowed)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def do_OPTIONS(self):
            self.send_response(204)
            self._set_cors()
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path).path
            if parsed == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._set_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy", "service": "Sanjaya AI Backend"}).encode("utf-8"))
            elif parsed in ["/", "/index.html"]:
                html_path = os.path.join(script_dir, "index.html")
                if os.path.exists(html_path):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    with open(html_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
            elif parsed in ["/sanjaya_logo.jpg", "/ancient_parchment.jpg", "/lord_ram_archer.jpg"]:
                asset_file = os.path.join(script_dir, parsed.lstrip("/"))
                if os.path.exists(asset_file):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    with open(asset_file, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path).path
            if parsed != "/api/query":
                self.send_response(404)
                self.end_headers()
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 50000:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self._set_cors()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid request size."}).encode("utf-8"))
                    return

                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                payload = json.loads(raw_body)
                query = payload.get("query")

                if not query or not isinstance(query, str) or len(query.strip()) == 0:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self._set_cors()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Query parameter is required and cannot be empty."}).encode("utf-8"))
                    return

                if len(query) > 2000:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self._set_cors()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Query exceeds maximum allowed length (2000 characters)."}).encode("utf-8"))
                    return

                answer = process_query(query.strip())
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._set_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"answer": answer}).encode("utf-8"))

            except Exception as e:
                sys.stderr.write(f"Error handling request: {sanitize_secret(str(e))}\n")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._set_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Unable to process the request at this time."}).encode("utf-8"))

        def log_message(self, format, *args):
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    httpd = HTTPServer(("", port), SanjayaHandler)
    sys.stdout.write(f"Sanjaya AI Backend HTTP server listening on port {port}...\n")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("Server shutting down.\n")
        httpd.server_close()

# --- 6. CLI / SCRIPT EXECUTION ENTRY POINT ---

if __name__ == "__main__":
    if "--server" in sys.argv:
        port = int(os.getenv("PORT", 5000))
        run_standalone_server(port)
    else:
        try:
            # Query can be passed via command line argument or stdin
            query = None
            if len(sys.argv) > 1 and sys.argv[1] != "":
                query = sys.argv[1]
            elif not sys.stdin.isatty():
                query = sys.stdin.read().strip()

            if not query:
                print(json.dumps({"answer": "System Ready. Please provide a query."}))
                sys.exit(0)

            answer = process_query(query)
            print(json.dumps({"answer": answer}))

        except Exception as e:
            sys.stderr.write(f"Fatal error: {sanitize_secret(str(e))}\n")
            print(json.dumps({"error": "Unable to process the request at this time."}))
            sys.exit(1)