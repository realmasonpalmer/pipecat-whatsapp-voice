import os
import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TextFrame
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.processors.frame_processor import FrameProcessor

HERMES_BRIDGE_URL = os.getenv("HERMES_BRIDGE_URL", "http://localhost:8001")


class HermesProcessor(FrameProcessor):
    """Custom processor that calls Hermes bridge for LLM responses."""
    def __init__(self, session_id="voice-agent"):
        super().__init__()
        self.session_id = session_id
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_frame(self, frame):
        if isinstance(frame, TextFrame):
            logger.info(f"HermesProcessor: got text: {frame.text[:50]}...")
            try:
                response = await self.client.post(
                    f"{HERMES_BRIDGE_URL}/chat",
                    json={"text": frame.text, "session_id": self.session_id}
                )
                response.raise_for_status()
                data = response.json()
                reply = data.get("response", "Sorry, I didn't get that.")
                logger.info(f"Hermes responded: {reply[:50]}...")
                return TextFrame(text=reply)
            except Exception as e:
                logger.error(f"Hermes bridge error: {e}")
                return TextFrame(text="Sorry, I encountered an error.")
        return frame


async def run_bot(websocket, stream_sid, call_sid):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                call_sid=call_sid,
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
    )

    # STT via OpenAI
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"), model="whisper-1")
    
    # TTS via OpenAI
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), model="tts-1", sample_rate=24000)
    
    # Custom Hermes processor instead of OpenAI LLM
    hermes_processor = HermesProcessor(session_id="voice-agent")

    context = LLMContext([
        {"role": "user", "content": "Start by greeting Mason warmly as Archie."}
    ])

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    from pipecat.pipeline.pipeline import Pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        hermes_processor,
        tts,
        transport.output(),
        assistant_aggregator,
    ])
    
    worker = PipelineWorker(pipeline, params=PipelineParams(
        audio_out_sample_rate=8000,
        enable_metrics=True,
        enable_usage_metrics=True,
    ))
    
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()
