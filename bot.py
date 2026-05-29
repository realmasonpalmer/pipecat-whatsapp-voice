import os
from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv()

SYSTEM_INSTRUCTION = """You are Archie — Mason's autonomous operations controller.
Keep responses SHORT — max 2 sentences. Voice-first. No markdown.
Always end with: DONE, BLOCKED, or NEEDS APPROVAL."""

async def run_bot(webrtc_connection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        system_prompt=SYSTEM_INSTRUCTION,
    )

    context = LLMContext([
        {"role": "user", "content": "Start by greeting Mason warmly as Archie."}
    ])

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(), user_aggregator, llm,
        transport.output(), assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(
        enable_metrics=True, enable_usage_metrics=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
