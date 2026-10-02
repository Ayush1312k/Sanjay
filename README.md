# Sanjaya AI Guru

Sanjaya AI Guru is a production-ready, AI-powered spiritual assistant providing grounded, authentic wisdom from the **Bhagavad Gita**, **Ramayana**, and **Mahabharata**. Named after Sanjaya, the visionary narrator blessed with divine vision in the Mahabharata epic, this application uses a hybrid Retrieval-Augmented Generation (RAG) architecture combining local FAISS vector search with Google Gemini and Anthropic Claude for deep, scripture-anchored spiritual guidance.

---

## Features

* **Bhagavad Gita Knowledge**: Fast semantic retrieval across all 18 chapters and 700 verses.
* **Ramayana Knowledge**: Narrative context, moral guidance, and character philosophy from Valmiki Ramayana.
* **Mahabharata Knowledge**: Contextual insight into Dharma, duty, and ethics from the great epic.
* **RAG Architecture**: Scripture chunks mapped into high-dimensional vector embeddings for grounded answers.
* **FAISS Vector Index**: Fast local vector similarity search over embedded scripture corpus.
* **Gemini Integration**: High-speed contextual reasoning with Google's Gemini models.
* **Claude Integration**: Rigorous theological and philosophical nuance with Anthropic's Claude 3.5 Sonnet.
* **Dual-Model Synthesis**: Automatic intelligent merging of Gemini and Claude outputs for optimal answers.
* **AI Spiritual Guidance**: Courteous, respectful tone designed for sincere seekers without sermons or hallucinations.
* **Production UI**: Clean, dark-mode spiritual aesthetic with typewriter response animation and speech synthesis.

---

## Architecture

The project is built with a decoupled, secure architecture designed for safe public GitHub hosting and scalable cloud deployment:

```text
               +----------------------------------+
               |        User Browser / UI         |
               +----------------------------------+
                                |
                                | HTTPS POST /api/query
                                v
               +----------------------------------+
               |   Frontend / Vercel Edge Proxy   |
               | (Vercel Serverless Function)     |
               +----------------------------------+
                                |
                                | Internal HTTPS POST /api/query
                                v
               +----------------------------------+
               |      Sanjaya AI Guru Backend     |
               |  (Node.js Express / Python HTTP) |
               +----------------------------------+
                     |                     |
                     v                     v
         +--------------------+   +--------------------+
         |   FAISS Vector DB  |   | Scripture Chunks   |
         | (15MB Index Cache) |   | (4MB Text Chunks)  |
         +--------------------+   +--------------------+
                     |
                     +---------------------+
                     |                     |
                     v                     v
          +--------------------+ +--------------------+
          |  Google Gemini API | | Anthropic Claude   |
          |  (Embeddings / LLM)| | (Philosophic LLM)  |
          +--------------------+ +--------------------+
```

### Why Decoupled Deployment?
The Python backend utilizes `faiss-cpu` (a C++ OpenMP shared library) and a pre-built 15MB vector index with 4MB text chunks. Because Vercel serverless functions enforce strict bundle size, memory, and runtime constraints where C++ binary dependencies like `libgomp.so` and dual-LLM cold starts can time out, Sanjaya clearly separates:
1. **Frontend**: Static Web Application deployed to **Vercel** with a serverless proxy at `/api/query`.
2. **Backend**: Dedicated Python/Node RAG server running locally or on a Python-capable platform (Render, Railway, Fly.io, Cloud Run, Hugging Face Spaces).

---

## Requirements

### Runtime Prerequisites
* **Node.js**: v18.0.0 or higher (v20+ recommended)
* **Python**: v3.10, 3.11, 3.12, or 3.13
* **npm**: v9.0.0 or higher

### Node Dependencies (`package.json`)
* `express` (^4.18.2)

### Python Dependencies (`requirements.txt`)
* `google-genai`
* `anthropic`
* `python-dotenv`
* `numpy`
* `faiss-cpu`

---

## Environment Variables

The backend requires credentials for Google Gemini and/or Anthropic Claude.

Create an `.env` file in `AI_backend/.env` (or set environment variables on your server):

```env
# AI Model Credentials (At least one key is required)
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Server Configuration (Optional)
PORT=5000
FRONTEND_URL=http://localhost:5000
```

> **NEVER hardcode actual API keys in source code or commit `.env` files to git.**

---

## Local Setup

### 1. Clone Repository
```bash
git clone https://github.com/Ayush1312k/Sanjay.git
cd Sanjay
```

### 2. Configure Environment Variables
Copy the template file to `.env`:

**On Windows (PowerShell):**
```powershell
Copy-Item AI_backend\.env.example AI_backend\.env
```

**On Linux / macOS:**
```bash
cp AI_backend/.env.example AI_backend/.env
```

Open `AI_backend/.env` in your text editor and fill in your actual credentials:
```env
GEMINI_API_KEY=your_actual_gemini_key
ANTHROPIC_API_KEY=your_actual_anthropic_key
```

### 3. Install Python Dependencies
```bash
cd AI_backend
pip install -r requirements.txt
```

### 4. Install Node Dependencies
```bash
npm install
```

### 5. Run Application

#### Option A: Node.js Express Server (Default)
```bash
node server.js
```
The server starts at `http://localhost:5000`. Open your browser and navigate to `http://localhost:5000` to interact with Sanjaya.

#### Option B: Standalone Python HTTP Server
```bash
python sanjaya_ai_backend.py --server
```
Runs a native Python HTTP server on port 5000 that pre-caches scripture embeddings into RAM for maximum query performance.

---

## GitHub Setup & Security

The repository is configured with a comprehensive `.gitignore` preventing:
* Local `.env` and secret files
* `node_modules/` dependencies
* Python `__pycache__`, virtual environments, and temporary files

Before pushing to GitHub:
```bash
# Verify no secret or untracked sensitive files are present
git status

# Push to your remote
git branch -M main
git push -u origin main
```

---

## Vercel Deployment

You can deploy the Sanjaya AI Guru frontend directly to Vercel:

### 1. Import Project to Vercel
1. Log into your [Vercel Dashboard](https://vercel.com).
2. Click **Add New Project** -> **Import Git Repository**.
3. Select your `Sanjay` repository.

### 2. Configure Project Settings
* **Framework Preset**: Other
* **Root Directory**: `./` (Default root directory contains `vercel.json` and `/api/query.js`)
* **Build Command**: None (Static assets)
* **Output Directory**: None

### 3. Configure Vercel Environment Variables
In the Vercel project settings under **Environment Variables**, add:
* `BACKEND_URL`: The HTTPS URL of your deployed Python/Node backend (e.g. `https://sanjaya-backend.onrender.com`).
* `FRONTEND_URL`: (Optional) Your Vercel frontend URL for strict CORS validation (e.g. `https://sanjaya.vercel.app`).

### 4. Deploy Dedicated Backend
Deploy `AI_backend` to a platform supporting Python and FAISS:
* **Render**: Create a Web Service -> connect repo -> Root Dir: `AI_backend` -> Build Command: `pip install -r requirements.txt` -> Start Command: `python sanjaya_ai_backend.py --server`
* **Railway**: Create project -> deploy from repo -> set start command to `python AI_backend/sanjaya_ai_backend.py --server`
* Set `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` in the hosting platform's environment variables.
* Copy the resulting backend URL and set it as `BACKEND_URL` in your Vercel frontend settings.

---

## Security

> **CRITICAL SECURITY REQUIREMENT**:
> Never commit `.env`, API keys, authentication tokens, or other secrets to GitHub.
> API credentials must be supplied exclusively through environment variables.
> All client queries are routed through a validated backend API boundary. Browser code never contains or calls Gemini or Anthropic API keys directly.
