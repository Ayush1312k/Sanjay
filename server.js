const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const app = express();
const PORT = 5000;

// Middleware to parse JSON bodies and handle CORS
app.use(express.json());
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*'); // Allow access from frontend HTML file
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

// Serve the static frontend file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Main API endpoint for RAG query
app.post('/api/query', (req, res) => {
    const userQuery = req.body.query;

    if (!userQuery) {
        return res.status(400).json({ error: 'Missing query parameter.' });
    }

    // --- Execute Python RAG Script ---
    // We execute the sanjaya_ai_backend.py script and pass the query as a command-line argument.
    const pythonProcess = spawn('python', ['sanjaya_ai_backend.py', userQuery]);
    
    let pythonOutput = '';
    let pythonError = '';

    // Capture output from the Python script
    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
    });

    // Capture errors from the Python script
    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
        if (code === 0) {
            // Success: Python script finished and returned a result (should be JSON)
            try {
                // Assuming Python outputs a single JSON object (the answer)
                const result = JSON.parse(pythonOutput.trim());
                res.json({ answer: result.answer });
            } catch (e) {
                // If Python output wasn't valid JSON (e.g., printed a raw error message)
                console.error("Python Output Parsing Error:", e);
                console.error("Raw Python Output:", pythonOutput);
                res.status(500).json({ error: 'Internal RAG error: Malformed Python output.', details: pythonOutput.substring(0, 200) });
            }
        } else {
            // Failure: Python script exited with a non-zero code (error)
            console.error(`Python RAG script exited with code ${code}.`);
            console.error("Python Error Stream:", pythonError);
            res.status(500).json({ 
                error: 'RAG Process Failed on Server.',
                details: pythonError || 'Check environment variables (GEMINI_API_KEY) and Python dependencies (faiss-cpu, google-genai).'
            });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Express.js Backend Server is running on http://localhost:${PORT}`);
    console.log('Ensure sanjaya_ai_backend.py is executable and dependencies are installed.');
});