import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

# Optional STT/TTS - import if keys exist
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

if DEEPGRAM_API_KEY:
    from pipecat.services.deepgram import DeepgramSTTService
if CARTESIA_API_KEY:
    from pipecat.services.cartesia import CartesiaTTSService

SYSTEM_INSTRUCTION = """You are Archie — Mason's autonomous operations controller.
Keep responses SHORT — max 2 sentences. Voice-first. No markdown.
Always end with: DONE, BLOCKED, or NEEDS APPROVAL."""

async def run_bot(websocket, stream_sid):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=TwilioFrameSerializer(stream_sid=stream_sid),
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        system_prompt=SYSTEM_INSTRUCTION,
    )

    # Build pipeline parts
    pipeline_parts = [transport.input()]

    # STT
    if DEEPGRAM_API_KEY:
        stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
        pipeline_parts.append(stt)
        logger.info("Deepgram STT enabled")
    else:
        logger.warning("No DEEPGRAM_API_KEY, STT disabled")

    # User aggregator
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        LLMContext([{"role": "user", "content": "Start by greeting Mason warmly as Archie."}]),
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    pipeline_parts.extend([user_aggregator, llm])

    # TTS
    if CARTESIA_API_KEY:
        tts = CartesiaTTSService(api_key=CARTESIA_API_KEY, voice_id="71a7ad14-091c-4e8e-a314-022ece01c121")
        pipeline_parts.append(tts)
        logger.info("Cartesia TTS enabled")
    else:
        logger.warning("No CARTESIA_API_KEY, TTS disabled")

    # Output and assistant aggregator
    pipeline_parts.extend([transport.output(), assistant_aggregator])

    from pipecat.pipeline.pipeline import Pipeline
    pipeline = Pipeline(pipeline_parts)

    worker = PipelineWorker(pipeline, params={"enable_metrics": True, "enable_usage_metrics": True})

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
