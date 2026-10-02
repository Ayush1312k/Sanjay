// Vercel Serverless Function Proxy for Sanjaya AI Guru
// This function acts as the secure API boundary between the frontend on Vercel
// and the dedicated Python / FAISS RAG backend.

module.exports = async (req, res) => {
    // 1. CORS headers
    const frontendUrl = process.env.FRONTEND_URL || '*';
    const origin = req.headers.origin || '*';
    const allowedOrigin = frontendUrl === '*' || frontendUrl.split(',').map(s => s.trim()).includes(origin) ? origin : frontendUrl;

    res.setHeader('Access-Control-Allow-Origin', allowedOrigin);
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
        return res.status(204).end();
    }

    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed. Use POST.' });
    }

    // 2. Validate request input
    const body = req.body || {};
    const query = body.query;

    if (!query || typeof query !== 'string' || query.trim().length === 0) {
        return res.status(400).json({ error: 'Query parameter is required and cannot be empty.' });
    }

    if (query.length > 2000) {
        return res.status(400).json({ error: 'Query exceeds maximum allowed length of 2000 characters.' });
    }

    // 3. Check for configured backend URL
    const backendUrl = process.env.BACKEND_URL;
    if (!backendUrl) {
        return res.status(503).json({
            error: 'Backend service is not configured. Please set BACKEND_URL in Vercel environment variables.'
        });
    }

    // 4. Securely forward query to backend
    try {
        const endpoint = backendUrl.replace(/\/+$/, '') + '/api/query';
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 55000); // 55s timeout

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query.trim() }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            console.error(`[Vercel Proxy] Backend returned status code ${response.status}`);
            return res.status(502).json({ error: 'Unable to process the request at this time.' });
        }

        const data = await response.json();
        return res.status(200).json({ answer: data.answer || 'No response returned.' });
    } catch (err) {
        console.error('[Vercel Proxy Error]: Request forwarding failed.');
        return res.status(500).json({ error: 'Unable to process the request at this time.' });
    }
};
