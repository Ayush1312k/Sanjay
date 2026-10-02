# Sanjaya AI Guru — Backend Service

This directory contains the Python RAG engine, FAISS vector index, scripture corpus, and Express.js API server for Sanjaya AI Guru.

For full project documentation, setup guides, and deployment instructions, please refer to the root [README.md](../README.md).

## Quick Local Run
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your `GEMINI_API_KEY` and/or `ANTHROPIC_API_KEY`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
4. Start server:
   ```bash
   node server.js
   # or
   python sanjaya_ai_backend.py --server
   ```
