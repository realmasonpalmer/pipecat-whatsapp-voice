import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, Response
from twilio.twiml.voice_response import VoiceResponse, Stream
from loguru import logger

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

app = FastAPI(title="Twilio WhatsApp Voice Bot")

@app.post("/TWILIO")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls."""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        logger.info(f"Incoming call: {call_sid}")

        response = VoiceResponse()
        connect = response.connect()
        stream = connect.stream(url=f"wss://{request.url.hostname}/TWILIO/stream/{call_sid}")
        # Return XML with proper content type
        return Response(content=str(response), media_type="application/xml")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        response = VoiceResponse()
        return Response(content=str(response), media_type="application/xml")

@app.websocket("/TWILIO/stream/{call_sid}")
async def twilio_stream(websocket: WebSocket, call_sid: str):
    """Handle Twilio Media Stream WebSocket."""
    await websocket.accept()
    logger.info(f"WebSocket connected for call: {call_sid}")

    # Extract stream SID from first Twilio message
    stream_sid = None
    try:
        data = await websocket.receive_json()
        if data.get("event") == "connected":
            logger.info(f"Stream connected: {data}")
        elif data.get("event") == "start":
            stream_sid = data.get("start", {}).get("streamSid")
            logger.info(f"Stream started: {stream_sid}")
    except Exception as e:
        logger.error(f"Failed to get stream SID: {e}")
        await websocket.close()
        return

    if not stream_sid:
        logger.error("No stream SID received")
        await websocket.close()
        return

    # Run bot with this WebSocket and stream SID
    from bot import run_bot
    try:
        await run_bot(websocket, stream_sid)
    except Exception as e:
        logger.error(f"Bot failed: {e}")
    finally:
        await websocket.close()
        logger.info(f"WebSocket closed for call: {call_sid}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host=host, port=port)
