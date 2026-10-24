"""OpenRouter chat client with live model catalog checks.

Usage:

    client = OpenRouterLLMClient(
        openrouter_api_key=...,
        model="openai/gpt-4o-mini",
    )
    client.chat(user_prompt="Tell me a fun fact")
    client.chat(user_prompt="Tell me a fun fact", system_prompt="You are a helpful assistant")
    client.chat(messages=[{"role": "user", "content": "Tell me a fun fact"}])

    client = OpenRouterLLMClient(
        model="meta-llama/llama-3.3-70b-instruct",
        mode="floor",
    )
    client.chat(user_prompt="Hello")
    client.chat(user_prompt="Hello", mode="nitro")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

from dotenv import load_dotenv
from openrouter import OpenRouter
from openrouter.types import UNSET

logger = logging.getLogger("openrouter_client")

_PROVIDER = "openrouter"
_DEFAULT_URL = "https://openrouter.ai/api/v1"
_MODES = ("balanced", "floor", "nitro")
_ROUTING_SUFFIXES = (":floor", ":nitro")

Mode = Literal["balanced", "floor", "nitro"]


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _catalog_aliases(model_id: str) -> list[str]:
    aliases = [model_id]
    if model_id.startswith("models/"):
        aliases.append(model_id.removeprefix("models/"))
    return aliases


def _check_mode(mode: str) -> Mode:
    if mode == "balanced" or mode == "floor" or mode == "nitro":
        return mode
    allowed = ", ".join(_MODES)
    raise ValueError(f"mode must be one of {allowed}; got {mode!r}")


def _base_model_id(model: str) -> str:
    for suffix in _ROUTING_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def _apply_mode(model: str, mode: Mode) -> str:
    if mode == "balanced":
        return model
    return f"{model}:{mode}"


def _message_text(content: Any) -> str:
    if content is None or content is UNSET:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class OpenRouterLLMClient:
    """Send chat completions through the official OpenRouter Python SDK.

    The client is enabled only when **both** an API key and a base URL are
    present. Values are resolved in this order: constructor argument, then
    environment variable (after ``load_dotenv(override=True)``), then the
    default URL ``https://openrouter.ai/api/v1``. The key has no implicit
    default. A missing key or URL is logged and the client is left disabled.

    Model ids must be full OpenRouter ids, for example ``openai/gpt-4o-mini``.
    A requested id is confirmed against the live ``/models`` catalog. Confirmed
    ids and the catalog are cached for ``cache_ttl`` seconds (default 1 hour).

    ``mode`` controls OpenRouter provider routing. ``balanced`` sends the base
    model id and keeps OpenRouter's default price-weighted load balancing.
    ``floor`` appends ``:floor`` (cheapest providers, including flex tier).
    ``nitro`` appends ``:nitro`` (highest throughput, including priority tier).
    ``:floor`` and ``:nitro`` are routing variants, not catalog model ids, so
    the catalog check always uses the base id. A flex or priority endpoint can
    bill at that tier's rate.
    """

    def __init__(
        self,
        *,
        openrouter_api_key: str | None = None,
        openrouter_base_url: str | None = None,
        model: str | None = None,
        mode: Mode = "balanced",
        cache_ttl: int = 3600,
        timeout: float = 90.0,
    ) -> None:
        """Create a client when an OpenRouter key and base URL are available.

        Args:
            openrouter_api_key: OpenRouter secret. Falls back to
                ``OPENROUTER_API_KEY``.
            openrouter_base_url: OpenRouter OpenAI-compatible base URL. Falls
                back to ``OPENROUTER_BASE_URL``, then
                ``https://openrouter.ai/api/v1``.
            model: Default model id used by ``chat()`` when none is passed.
                Must be a full OpenRouter id. A trailing ``:floor`` or
                ``:nitro`` is ignored; routing comes from ``mode``.
            mode: Default routing mode for ``chat()``. One of ``balanced``,
                ``floor``, or ``nitro``. Defaults to ``balanced``.
            cache_ttl: Seconds to cache confirmed model ids and the catalog.
                Defaults to 3600.
            timeout: Request timeout in seconds for chat and catalog calls.
                Defaults to 90. Passed to the SDK as ``timeout_ms``.
        """
        load_dotenv(override=True)
        self.model = model
        self.mode = _check_mode(mode)
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._client: OpenRouter | None = None
        self._base_url: str | None = None
        self._skipped: dict[str, str] = {}
        self._catalog: tuple[float, set[str]] | None = None
        self._routes: dict[str, tuple[float, str]] = {}

        api_key = _first_nonempty(openrouter_api_key, _env("OPENROUTER_API_KEY"))
        base_url = _first_nonempty(openrouter_base_url, _env("OPENROUTER_BASE_URL"), _DEFAULT_URL)
        missing = []
        if not api_key:
            missing.append("no api key (set openrouter_api_key or OPENROUTER_API_KEY)")
        if not base_url:
            missing.append("no base url (set openrouter_base_url or OPENROUTER_BASE_URL)")
        if missing:
            reason = "; ".join(missing)
            self._skipped[_PROVIDER] = reason
            logger.warning("skipping %s: %s", _PROVIDER, reason)
            return

        self._client = OpenRouter(
            api_key=api_key,
            server_url=base_url,
            timeout_ms=int(self.timeout * 1000),
        )
        self._base_url = base_url
        logger.info("enabled %s (%s)", _PROVIDER, base_url)

    def list_models(self) -> dict[str, list[str]]:
        """Return currently advertised OpenRouter model ids.

        Refreshes the catalog. Returns an empty dict when the client was
        skipped at init (missing key or URL) or the catalog request fails.
        """
        catalog = self._provider_catalog(force=True)
        if catalog is None:
            return {}
        return {_PROVIDER: sorted(catalog)}

    def chat(
        self,
        user_prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        mode: Mode | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion for a full OpenRouter model id.

        Provide either ``user_prompt`` (optionally with ``system_prompt``) or a
        raw ``messages`` list, not both. Extra ``kwargs`` are forwarded to
        ``chat.send``. ``stream`` is always false so the return value is text.

        Args:
            user_prompt: User message content. Required unless ``messages``
                is given.
            system_prompt: Optional system message, prepended when using
                ``user_prompt``.
            model: Full OpenRouter model id. Defaults to the constructor
                ``model``. A trailing ``:floor`` or ``:nitro`` is stripped
                before the catalog check.
            mode: Routing mode for this call. Overrides the constructor
                ``mode``. One of ``balanced``, ``floor``, or ``nitro``.
            messages: OpenAI-style message list. Mutually exclusive with the
                prompt arguments.
            **kwargs: Extra Chat Completions parameters (for example
                ``temperature``).

        Returns:
            The assistant message content from the first choice.
        """
        messages = self._build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
        )

        chosen_model = model or self.model
        if not chosen_model:
            raise ValueError("No model specified; pass model= to chat() or to OpenRouterLLMClient()")

        chosen_mode = self.mode if mode is None else _check_mode(mode)
        base_model = _base_model_id(chosen_model)
        resolved_model = self._resolve(base_model)
        request_model = _apply_mode(resolved_model, chosen_mode)
        client = self._client
        if client is None:
            raise ValueError(self._missing_model_message(base_model, tried="none"))

        logger.info(
            "resolved %r -> %s (%s, mode=%s)",
            request_model,
            _PROVIDER,
            self._base_url,
            chosen_mode,
        )
        started = time.monotonic()
        response = client.chat.send(
            model=request_model,
            messages=messages,
            **{**kwargs, "stream": False},
        )
        elapsed_s = time.monotonic() - started
        logger.info(
            "response from %s (%s) in %.2fs",
            request_model,
            _PROVIDER,
            elapsed_s,
        )
        return _message_text(response.choices[0].message.content)

    def _build_messages(
        self,
        *,
        user_prompt: str | None,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        using_prompt_args = user_prompt is not None or system_prompt is not None
        if messages is not None and using_prompt_args:
            raise ValueError("Pass messages or prompt arguments, not both")
        if messages is not None:
            return messages
        if user_prompt is None:
            raise ValueError("Provide user_prompt or messages")

        built: list[dict[str, Any]] = []
        if system_prompt is not None:
            built.append({"role": "system", "content": system_prompt})
        built.append({"role": "user", "content": user_prompt})
        return built

    def _resolve(self, model: str) -> str:
        cached = self._cached_route(model)
        if cached is not None:
            logger.info("cache hit: %r -> %s (%s)", cached, _PROVIDER, self._base_url)
            return cached

        if self._client is None:
            raise ValueError(self._missing_model_message(model, tried="none"))

        catalog = self._provider_catalog()
        if catalog is None:
            raise ValueError(self._missing_model_message(model, tried=f"{_PROVIDER} (catalog unavailable)"))
        if model not in catalog:
            logger.info("%s catalog: not found %r", _PROVIDER, model)
            raise ValueError(self._missing_model_message(model, tried=_PROVIDER))

        logger.info("%s catalog: found %r", _PROVIDER, model)
        return self._store_route(model)

    def _missing_model_message(self, model: str, *, tried: str) -> str:
        skipped = ", ".join(f"{name} ({reason})" for name, reason in self._skipped.items()) or "none"
        return (
            f"No configured provider listed model {model!r}. "
            f"Tried: {tried}. Skipped: {skipped}."
        )

    def _cached_route(self, model: str) -> str | None:
        entry = self._routes.get(model)
        if entry is None:
            return None
        cached_at, resolved_model = entry
        if time.monotonic() - cached_at > self.cache_ttl:
            logger.info("cache expired for %r", model)
            self._routes.pop(model, None)
            return None
        if self._client is None:
            self._routes.pop(model, None)
            return None
        return resolved_model

    def _store_route(self, model: str) -> str:
        self._routes[model] = (time.monotonic(), model)
        return model

    def _provider_catalog(self, *, force: bool = False) -> set[str] | None:
        if self._catalog is not None and not force:
            fetched_at, ids = self._catalog
            if time.monotonic() - fetched_at <= self.cache_ttl:
                return ids
            logger.info("%s catalog cache expired; refreshing", _PROVIDER)

        client = self._client
        if client is None:
            return None
        try:
            ids: set[str] = set()
            page = client.models.list(output_modalities="all")
            while page is not None:
                for item in page.result.data:
                    for alias in _catalog_aliases(item.id):
                        ids.add(alias)
                page = page.next()
        except Exception as exc:
            logger.warning("skipping %s catalog: %s", _PROVIDER, exc)
            self._catalog = None
            return None

        self._catalog = (time.monotonic(), ids)
        logger.info("%s catalog: loaded %d model id(s)", _PROVIDER, len(ids))
        return ids


__all__ = ["Mode", "OpenRouterLLMClient"]
