"""FastAPI-based MCP server exposing speech and chat tools."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .asr_module import ASRModule
from .conversation import ConversationManager
from .nlp_module import DeepSeekClient, KnowledgeBase, NLPModule
from .setting import PROJECT_ROOT, configure_logging, load_config
from .tts_module import HiggsAudioTTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .langchain_agent import AgentRunResult, LangChainVoiceAgent


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    text: str
    audio_path: Optional[str]
    citations: list[Any]


class AgentChatRequest(BaseModel):
    text: Optional[str] = None
    audio_path: Optional[str] = None
    auto_tts: Optional[bool] = None


class AgentChatResponse(BaseModel):
    text: str
    audio_path: Optional[str]
    citations: list[Any]
    tools: list[str]


def build_conversation_manager(config: Dict[str, Any]) -> ConversationManager:
    asr_cfg = config.get("asr", {})
    asr_module = ASRModule(**asr_cfg)

    nlp_cfg = config.get("nlp", {})
    ds_cfg = nlp_cfg.get("deepseek", {})
    if not ds_cfg.get("api_key"):
        raise RuntimeError("DeepSeek API key missing in configuration")
    deepseek = DeepSeekClient(
        api_key=ds_cfg["api_key"],
        model=ds_cfg.get("model", "deepseek-r1"),
        base_url=ds_cfg.get("base_url", "https://api.deepseek.com"),
    )

    knowledge_cfg = nlp_cfg.get("knowledge_base", {})
    knowledge_base: Optional[KnowledgeBase] = None
    if knowledge_cfg.get("enabled", True):
        kb_path = knowledge_cfg.get("index_path")
        if not kb_path:
            kb_path = PROJECT_ROOT / "models" / "embeddings" / "knowledge_base.json"
        knowledge_base = KnowledgeBase(
            embedding_model=knowledge_cfg.get(
                "embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ),
            index_path=kb_path,
        )

    nlp_module = NLPModule(
        llm_client=deepseek,
        knowledge_base=knowledge_base,
        system_prompt=nlp_cfg.get("system_prompt"),
    )

    tts_cfg = config.get("tts", {})
    if not tts_cfg.get("api_base"):
        raise RuntimeError("HiggsAudio api_base missing in configuration")
    tts_module = HiggsAudioTTS(
        api_base=tts_cfg["api_base"],
        api_key=tts_cfg.get("api_key"),
        default_voice=tts_cfg.get("default_voice", "default"),
        sample_rate=tts_cfg.get("sample_rate", 22050),
        audio_format=tts_cfg.get("format", "wav"),
        timeout=tts_cfg.get("timeout", 60),
    )

    return ConversationManager(
        asr=asr_module,
        nlp=nlp_module,
        tts=tts_module,
        tts_output_dir=config.get("runtime", {}).get("tts_output_dir", PROJECT_ROOT / "logs"),
    )


def build_agent(manager: ConversationManager, config: Dict[str, Any]):
    agent_cfg = config.get("agent", {})
    if not agent_cfg.get("enabled", False):
        return None

    from .langchain_agent import LangChainVoiceAgent  # Imported lazily for optional dependency

    return LangChainVoiceAgent(
        manager,
        system_prompt=agent_cfg.get("system_prompt"),
        retrieval_top_k=agent_cfg.get("retrieval_top_k", 4),
        auto_tts=agent_cfg.get("auto_tts", True),
        llm_temperature=agent_cfg.get("temperature", 0.6),
        llm_top_p=agent_cfg.get("top_p", 0.9),
    )


def create_app(config_path: Optional[str | Path] = None) -> FastAPI:
    configure_logging()
    config = load_config(config_path)
    manager = build_conversation_manager(config)
    agent = build_agent(manager, config)

    app = FastAPI(title="Voice Assistant MCP Server")
    app.state.manager = manager
    app.state.agent = agent

    @app.post("/speech_to_text")
    async def speech_to_text(file: UploadFile = File(...)) -> Dict[str, Any]:
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            result = app.state.manager.asr.transcribe(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": [segment.__dict__ for segment in result.segments],
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")
        turn = app.state.manager.handle_text(payload.text)
        citations = [citation.__dict__ for citation in turn.citations]
        return ChatResponse(
            text=turn.assistant_text,
            audio_path=str(turn.audio) if turn.audio else None,
            citations=citations,
        )

    @app.post("/agent_chat", response_model=AgentChatResponse)
    async def agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
        agent_instance = app.state.agent
        if agent_instance is None:
            raise HTTPException(status_code=503, detail="Agent mode is disabled in configuration")

        if not (payload.text and payload.text.strip()) and not payload.audio_path:
            raise HTTPException(status_code=400, detail="Provide text or audio_path for the agent")

        result = agent_instance.run(
            text=payload.text if payload.text and payload.text.strip() else None,
            audio_path=payload.audio_path,
            auto_tts=payload.auto_tts,
        )
        return AgentChatResponse(
            text=result.text,
            audio_path=str(result.audio_path) if result.audio_path else None,
            citations=[citation.__dict__ for citation in result.citations],
            tools=result.tool_messages,
        )

    @app.post("/text_to_speech")
    async def text_to_speech(payload: ChatRequest) -> Dict[str, Any]:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")
        tts_response = app.state.manager.tts.synthesize(payload.text, app.state.manager.tts_output_dir)
        return {
            "audio_path": str(tts_response.audio_path),
            "format": tts_response.format,
            "sample_rate": tts_response.sample_rate,
        }

    return app
