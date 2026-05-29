import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from twilio.twiml.voice_response import VoiceResponse
from loguru import logger
from bot import run_bot
import asyncio

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

app = FastAPI(title="Twilio WhatsApp Voice Bot")

@app.post("/twilio")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    logger.info(f"Incoming call: {call_sid}")

    response = VoiceResponse()
    # Connect call to bot via media stream
    stream_url = f"wss://{request.url.hostname}/twilio/stream?call_sid={call_sid}"
    response.connect().stream(url=stream_url)
    return str(response)

@app.websocket("/twilio/stream")
async def twilio_stream(websocket):
    """Handle Twilio Media Stream WebSocket."""
    await websocket.accept()
    logger.info("WebSocket connected for Twilio Media Stream")
    try:
        await run_bot()
    except Exception as e:
        logger.error(f"Bot failed: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host=host, port=port)
