import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from loguru import logger
import asyncio

load_dotenv()

app = FastAPI(title="Twilio WhatsApp Voice Bot")

@app.post("/TWILIO")
async def twilio_voice_webhook(request: Request):
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    logger.info(f"Incoming call: {call_sid}")
    hostname = request.url.hostname
    stream_url = f"wss://{hostname}/TWILIO/stream/{call_sid}"
    response = VoiceResponse()
    connect = response.connect()
    connect.stream(url=stream_url)
    xml = str(response)
    logger.info(f"TwiML: {xml}")
    return Response(content=xml, media_type="application/xml")

@app.websocket("/TWILIO/stream/{call_sid}")
async def twilio_stream(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for call: {call_sid}")

    stream_sid = None
    try:
        # Wait for start event with timeout
        for _ in range(50):  # ~5 seconds timeout
            data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
            event = data.get("event")
            logger.info(f"Twilio event received: {event}, full data: {data}")

            if event == "connected":
                logger.info(f"Stream connected event for {call_sid}")
            elif event == "start":
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                logger.info(f"Start event: stream_sid={stream_sid}, start_data={start_data}")
                if stream_sid:
                    break
                else:
                    logger.warning("Start event missing streamSid")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for start event")
    except Exception as e:
        logger.error(f"Error getting stream SID: {e}")
        await websocket.close()
        return

    if not stream_sid:
        logger.error("No stream SID received")
        await websocket.close()
        return

    logger.info(f"Starting bot for stream: {stream_sid}")
    from bot import run_bot
    try:
        await run_bot(websocket, stream_sid, call_sid)
    except Exception as e:
        logger.error(f"Bot failed: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # Transport already closed it
        logger.info(f"WebSocket closed for call: {call_sid}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host=host, port=port)
