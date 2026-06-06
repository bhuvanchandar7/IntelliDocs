import os
import time
from typing import Any, List, Optional, Iterator
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from backend.config import settings

class DummyLLM(LLM):
    """
    A developer-friendly mock LLM that simulates streaming generation.
    It prints out information about the retrieved context, making it extremely
    useful for testing the RAG pipeline end-to-end without API keys or heavy models.
    """
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "Dummy LLM response (use streaming for full details)."

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        # Try parsing context out of the prompt
        context = ""
        if "Context:" in prompt:
            try:
                parts = prompt.split("Context:\n")
                if len(parts) > 1:
                    context = parts[1].split("\n\nQuestion:")[0]
            except Exception:
                pass

        response_text = (
            "👋 Hello! You are running in IntelliDocs developer **'dummy'** mode.\n\n"
            "This mode lets you test the full RAG pipeline (ingestion, parsing, chunking, Chroma vector storage, "
            "BM25 keyword search, and CrossEncoder reranking) without needing external API keys or heavy local models.\n\n"
            "Here is the context retrieved from your documents:\n"
        )

        if context.strip():
            # Format the retrieved context nicely
            trimmed_context = context.strip()
            if len(trimmed_context) > 600:
                trimmed_context = trimmed_context[:600] + "\n... [truncated] ..."
            response_text += f"\n```text\n{trimmed_context}\n```\n\n"
        else:
            response_text += "\n*(No document context was found. Try uploading a PDF in the Knowledge Base!)*\n\n"

        response_text += (
            "To use a real model, update the `LLM_PROVIDER` setting in your `.env` file to "
            "`openai`, `gemini`, `ollama`, or `local` (and provide any required keys or model files)."
        )

        # Stream words with minor delays to simulate realistic LLM streaming
        words = response_text.split(" ")
        for word in words:
            chunk = GenerationChunk(text=word + " ")
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)
            yield chunk
            time.sleep(0.015)

def get_llm():
    """
    LLM Factory. Returns an instance of the configured LLM client.
    """
    provider = settings.llm_provider.lower()
    
    print(f"Initializing LLM provider: {provider}")
    
    if provider == "dummy":
        return DummyLLM()
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided to use the OpenAI LLM provider.")
        return ChatOpenAI(
            openai_api_key=api_key,
            model=settings.llm_model or "gpt-4o-mini",
            temperature=0.7,
            streaming=True
        )
        
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenAI
        api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided to use the Gemini LLM provider.")
        return ChatGoogleGenAI(
            google_api_key=api_key,
            model=settings.llm_model or "gemini-1.5-flash",
            temperature=0.7,
            streaming=True
        )
        
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model or "mistral",
            temperature=0.7
        )
        
    elif provider == "local":
        model_path = settings.llm_model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local GGUF model file not found at '{model_path}'. "
                "Please download the model or switch LLM_PROVIDER to 'dummy' or 'openai'."
            )
            
        from langchain_community.llms import LlamaCpp
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=4096,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=True,
            temperature=0.7,
        )
        
    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            "Supported providers: local, openai, gemini, ollama, dummy"
        )
