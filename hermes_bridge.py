#!/usr/bin/env python3
"""
Hermes Bridge: HTTP endpoint that forwards requests to Hermes CLI and returns responses.
Run this alongside the Pipecat bot on your Mac.
"""
import subprocess
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os

app = FastAPI(title="Hermes Bridge")

# Path to hermes binary - adjust if needed
HERMES_BIN = os.getenv("HERMES_BIN", "/Users/openclaw/.hermes/hermes-agent/bin/hermes")

@app.post("/chat")
async def chat(request: Request):
    """Send text to Hermes and return response."""
    try:
        data = await request.json()
        text = data.get("text", "")
        session_id = data.get("session_id", "voice-agent")
        
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' field")
        
        # Call hermes chat -q with the text
        # Using --resume to maintain context across calls to the same session
        result = subprocess.run(
            [HERMES_BIN, "chat", "-q", text, "--resume", session_id],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Hermes error: {result.stderr}")
            return JSONResponse(
                content={"error": result.stderr, "response": "Sorry, I encountered an error."},
                status_code=500
            )
        
        response_text = result.stdout.strip()
        return {"response": response_text}
        
    except subprocess.TimeoutExpired:
        return JSONResponse(
            content={"error": "timeout", "response": "Sorry, that took too long."},
            status_code=504
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "response": "Sorry, something went wrong."},
            status_code=500
        )

@app.get("/health")
async def health():
    return {"status": "ok", "hermes_bin": HERMES_BIN}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"Starting Hermes Bridge on port {port}")
    print(f"Hermes binary: {HERMES_BIN}")
    uvicorn.run(app, host="0.0.0.0", port=port)
