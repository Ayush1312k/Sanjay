const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Load environment variables from .env if present (for local development)
const envPath = path.join(__dirname, '.env');
if (fs.existsSync(envPath)) {
    try {
        const envContent = fs.readFileSync(envPath, 'utf8');
        envContent.split(/\r?\n/).forEach(line => {
            const trimmed = line.trim();
            if (trimmed && !trimmed.startsWith('#')) {
                const eqIdx = trimmed.indexOf('=');
                if (eqIdx !== -1) {
                    const key = trimmed.slice(0, eqIdx).trim();
                    const val = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, '');
                    if (!process.env[key]) {
                        process.env[key] = val;
                    }
                }
            }
        });
    } catch (err) {
        console.error('Notice: Could not parse local .env file.');
    }
}

const app = express();
const PORT = process.env.PORT || 5000;
const FRONTEND_URL = process.env.FRONTEND_URL;
const MAX_QUERY_LENGTH = 2000;

// Middleware to parse JSON bodies with size limit
app.use(express.json({ limit: '64kb' }));

// Helper to sanitize logs against credential leakage
function sanitizeLog(text) {
    if (!text) return '';
    return String(text)
        .replace(/AIza[0-9A-Za-z-_]{35}/g, '[REDACTED_API_KEY]')
        .replace(/sk-ant-[0-9A-Za-z-_]+/g, '[REDACTED_API_KEY]');
}

// Configurable CORS handling
app.use((req, res, next) => {
    const origin = req.headers.origin;

    let allowedOrigin = '*';
    if (FRONTEND_URL && FRONTEND_URL !== '*') {
        const allowedOrigins = FRONTEND_URL.split(',').map(s => s.trim());
        if (origin && (allowedOrigins.includes(origin) || origin.includes('localhost') || origin.includes('127.0.0.1'))) {
            allowedOrigin = origin;
        } else {
            allowedOrigin = allowedOrigins[0] || '*';
        }
    } else if (origin) {
        allowedOrigin = origin;
    }

    res.header('Access-Control-Allow-Origin', allowedOrigin);
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
        return res.sendStatus(204);
    }
    next();
});

// Serve static assets (images, html, etc.)
app.use(express.static(__dirname));

// Health check endpoint for uptime monitoring and deployment platforms
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', service: 'Sanjaya Express Server' });
});

// Root endpoint serves index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Main API endpoint for RAG query
app.post('/api/query', (req, res) => {
    // 1. Validate request structure
    if (!req.body || typeof req.body !== 'object') {
        return res.status(400).json({ error: 'Invalid request body. Expected JSON object.' });
    }

    const userQuery = req.body.query;

    // 2. Validate query input
    if (typeof userQuery !== 'string' || userQuery.trim().length === 0) {
        return res.status(400).json({ error: 'Query parameter is required and cannot be empty.' });
    }

    if (userQuery.length > MAX_QUERY_LENGTH) {
        return res.status(400).json({
            error: `Query exceeds maximum allowed length of ${MAX_QUERY_LENGTH} characters.`
        });
    }

    const cleanQuery = userQuery.trim();

    // 3. Execute Python RAG backend
    const pythonScript = path.join(__dirname, 'sanjaya_ai_backend.py');
    const pythonCmd = process.env.PYTHON_PATH || (process.platform === 'win32' ? 'python' : 'python3');

    const pythonProcess = spawn(pythonCmd, [pythonScript], {
        cwd: __dirname,
        env: { ...process.env }
    });

    // Pass the query via stdin to avoid shell argument length limits or injection
    pythonProcess.stdin.write(cleanQuery);
    pythonProcess.stdin.end();

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
    });

    // Set 60-second execution timeout
    const timeout = setTimeout(() => {
        pythonProcess.kill();
        if (!res.headersSent) {
            console.error('[Timeout] Python process timed out after 60s.');
            res.status(504).json({ error: 'Request timed out while consulting scriptures.' });
        }
    }, 60000);

    // Handle process completion
    pythonProcess.on('close', (code) => {
        clearTimeout(timeout);
        if (res.headersSent) return;

        if (code === 0) {
            try {
                const result = JSON.parse(pythonOutput.trim());
                if (result.error) {
                    console.error('[Python Application Error]:', sanitizeLog(result.error));
                    return res.status(500).json({
                        error: 'Unable to process the request at this time.'
                    });
                }
                return res.json({ answer: result.answer || 'I could not retrieve an answer at this time.' });
            } catch (err) {
                console.error('[Parsing Error] Malformed Python output JSON:', sanitizeLog(pythonOutput.slice(0, 150)));
                return res.status(500).json({
                    error: 'Unable to process the request at this time.'
                });
            }
        } else {
            console.error(`[Process Error] Python exited with code ${code}.`);
            if (pythonError) {
                console.error('[Python Stderr]:', sanitizeLog(pythonError));
            }
            return res.status(500).json({
                error: 'Unable to process the request at this time.'
            });
        }
    });

    pythonProcess.on('error', (err) => {
        clearTimeout(timeout);
        if (res.headersSent) return;
        console.error('[Execution Error] Failed to spawn Python process:', sanitizeLog(err.message));
        return res.status(500).json({
            error: 'Unable to process the request at this time.'
        });
    });
});

app.listen(PORT, () => {
    console.log(`Sanjaya Express Server listening on http://localhost:${PORT}`);
});