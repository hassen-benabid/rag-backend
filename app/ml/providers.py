"""Build chat and embedding models for any supported provider.

Same idea as rag-project/config.py, but the settings are passed in as arguments
instead of read from the environment, so this module stays framework-independent.
"""
DEFAULT_CHAT_MODELS = {
    "google": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}
DEFAULT_EMBEDDING_MODELS = {
    "google": "models/gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def build_chat_model(provider: str, model: str | None, api_key: str | None, timeout: int):
    model = model or DEFAULT_CHAT_MODELS[provider]
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0,
                                      timeout=timeout, max_retries=2)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=0,
                          timeout=timeout, max_retries=2)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0)
    raise ValueError(f"Unknown LLM provider: {provider}")


def build_embeddings(provider: str, model: str | None, api_key: str | None):
    model = model or DEFAULT_EMBEDDING_MODELS[provider]
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=api_key)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown embedding provider: {provider}")
