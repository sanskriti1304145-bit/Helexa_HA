"""
llm.py - one place that creates the LLM (Claude) object.

Every part of the project that needs the LLM calls get_llm(), so the model
name, key and limits are set in exactly one spot (values come from config.py).
Tests pass in a fake LLM instead, so they run offline and cost nothing.
"""

from langchain_anthropic import ChatAnthropic

import config


def get_llm(model=None):
    """Return a LangChain chat model connected to Anthropic Claude.

    model: leave as None for the agent's model (config.LLM_MODEL); the
    evaluation passes config.JUDGE_MODEL.

    Note: we do not set "temperature". The current Claude models choose
    their own sampling settings and reject custom temperature values.
    """
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    return ChatAnthropic(
        model=model or config.LLM_MODEL,
        anthropic_api_key=config.ANTHROPIC_API_KEY,
        max_tokens=config.LLM_MAX_TOKENS,
        max_retries=2,                 # retry briefly on network hiccups / rate limits
        default_request_timeout=60,    # seconds
    )
