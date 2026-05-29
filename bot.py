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
from pipecat.services.openai import OpenAILLMService, OpenAISTTService, OpenAITTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
    )

    stt = OpenAISTTService(api_key=OPENAI_API_KEY, model="whisper-1")
    tts = OpenAITTSService(api_key=OPENAI_API_KEY, model="tts-1")

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
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(pipeline, params={"enable_metrics": True, "enable_usage_metrics": True})

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
