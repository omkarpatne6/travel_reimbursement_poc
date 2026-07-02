"""Multi-provider LLM factory.

Reads provider credentials from the environment. The caller supplies a
provider name (from the request payload) and an optional model override;
everything else -- API keys, endpoints, default model names -- comes from
`.env`.
"""

import os

from langchain_core.language_models import BaseChatModel

PROVIDER_DEFAULTS = {
    "openai": "gpt-4o",
    "azure_openai": "gpt-4o",
    "gemini": "gemini-1.5-pro",
    "anthropic": "claude-3-5-sonnet-20241022",
}

PROVIDER_MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "azure_openai": "AZURE_OPENAI_DEPLOYMENT",
    "gemini": "GEMINI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}


def resolve_model_name(provider: str, model_name: str | None = None) -> str:
    """Return the model/deployment name `get_llm` would actually use, for display in
    observability traces -- without needing to introspect provider-specific client
    attributes (ChatOpenAI uses `.model_name`, AzureChatOpenAI `.azure_deployment`,
    ChatGoogleGenerativeAI `.model`, etc.)."""
    provider = provider.lower()
    if model_name:
        return model_name
    env_var = PROVIDER_MODEL_ENV.get(provider)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    return PROVIDER_DEFAULTS.get(provider, "unknown")


def get_llm(provider: str, model_name: str | None = None, temperature: float = 0) -> BaseChatModel:
    """Build a chat LLM for the given provider.

    `model_name` overrides the env default if supplied. Raises ValueError if
    required env vars are missing for the provider.
    """
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or os.environ.get("OPENAI_MODEL", PROVIDER_DEFAULTS["openai"]),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature,
        )

    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=model_name or os.environ["AZURE_OPENAI_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or os.environ.get("GEMINI_MODEL", PROVIDER_DEFAULTS["gemini"]),
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name or os.environ.get("ANTHROPIC_MODEL", PROVIDER_DEFAULTS["anthropic"]),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: openai, azure_openai, gemini, anthropic"
        )


def get_available_providers() -> list[str]:
    """Return the list of providers whose required env vars are present."""
    checks = {
        "openai": ["OPENAI_API_KEY"],
        "azure_openai": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"],
        "gemini": ["GOOGLE_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }
    return [p for p, keys in checks.items() if all(os.environ.get(k) for k in keys)]
