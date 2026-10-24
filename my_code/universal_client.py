"""Universal OpenAI-compatible chat client with live model routing.

Usage:

    client = UniversalLLMClient(
        openai_api_key=...,
        google_api_key=...,
        model="gpt-5-nano",
    )
    client.chat(user_prompt="Tell me a fun fact")
    client.chat(user_prompt="Tell me a fun fact", system_prompt="You are a helpful assistant")
    client.chat(messages=[{"role": "user", "content": "Tell me a fun fact"}])
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("universal_client")

_PROBE_ORDER = ("openai", "anthropic", "google", "ollama_local", "ollama_cloud")

# Family prefixes only — never specific model ids.
_PREFIX_ROUTES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "google"),
    ("llama", "ollama_local"),
)

_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "key_arg": "openai_api_key",
        "url_arg": "openai_base_url",
        "key_envs": ("OPENAI_API_KEY",),
        "url_envs": ("OPENAI_BASE_URL",),
        "default_url": "https://api.openai.com/v1",
        "headers": None,
    },
    "anthropic": {
        "key_arg": "anthropic_api_key",
        "url_arg": "anthropic_base_url",
        "key_envs": ("ANTHROPIC_API_KEY",),
        "url_envs": ("ANTHROPIC_BASE_URL",),
        "default_url": "https://api.anthropic.com/v1/",
        "headers": {"anthropic-version": "2023-06-01"},
    },
    "google": {
        "key_arg": "google_api_key",
        "url_arg": "google_base_url",
        "key_envs": ("GOOGLE_API_KEY",),
        "url_envs": ("GOOGLE_BASE_URL",),
        "default_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "headers": None,
    },
    "ollama_local": {
        "key_arg": "ollama_local_api_key",
        "url_arg": "ollama_local_base_url",
        "key_envs": ("OLLAMA_LOCAL_API_KEY",),
        "url_envs": ("OLLAMA_LOCAL_BASE_URL",),
        "default_url": "http://localhost:11434/v1",
        "headers": None,
    },
    "ollama_cloud": {
        "key_arg": "ollama_cloud_api_key",
        "url_arg": "ollama_cloud_base_url",
        "key_envs": ("OLLAMA_CLOUD_API_KEY", "OLLAMA_API_KEY"),
        "url_envs": ("OLLAMA_CLOUD_BASE_URL",),
        "default_url": "https://ollama.com/v1",
        "headers": None,
    },
}


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


def _prefix_for(model: str) -> tuple[str, str] | None:
    lowered = model.lower()
    for prefix, provider in _PREFIX_ROUTES:
        if lowered.startswith(prefix):
            return prefix, provider
    return None


def _catalog_aliases(model_id: str) -> list[str]:
    aliases = [model_id]
    if model_id.startswith("models/"):
        aliases.append(model_id.removeprefix("models/"))
    return aliases


class UniversalLLMClient:
    """Route a model name to an OpenAI-compatible chat endpoint.

    Each provider is enabled only when **both** an API key and a base URL are
    present. Values are resolved in this order: constructor argument, then
    environment variable (after ``load_dotenv(override=True)``), then a
    well-known default URL if one exists. Keys have no implicit default.
    Incomplete providers are skipped with a warning and never probed.

    Provider credentials
    --------------------
    openai
        Required key: ``openai_api_key`` or ``OPENAI_API_KEY``.
        URL: ``openai_base_url`` or ``OPENAI_BASE_URL``, else
        ``https://api.openai.com/v1``.
        Default route prefix: ``gpt-``.

    anthropic
        Required key: ``anthropic_api_key`` or ``ANTHROPIC_API_KEY``.
        URL: ``anthropic_base_url`` or ``ANTHROPIC_BASE_URL``, else
        ``https://api.anthropic.com/v1/``.
        Default route prefix: ``claude-``.

    google
        Required key: ``google_api_key`` or ``GOOGLE_API_KEY``.
        URL: ``google_base_url`` or ``GOOGLE_BASE_URL``, else
        ``https://generativelanguage.googleapis.com/v1beta/openai/``.
        Default route prefix: ``gemini-``.

    ollama_local
        Required key: ``ollama_local_api_key`` or ``OLLAMA_LOCAL_API_KEY``
        (local Ollama ignores the value; ``"ollama"`` is the usual placeholder).
        URL: ``ollama_local_base_url`` or ``OLLAMA_LOCAL_BASE_URL``, else
        ``http://localhost:11434/v1``.
        Default route prefix: ``llama``.

    ollama_cloud
        Required key: ``ollama_cloud_api_key``, ``OLLAMA_CLOUD_API_KEY``, or
        ``OLLAMA_API_KEY``.
        URL: ``ollama_cloud_base_url`` or ``OLLAMA_CLOUD_BASE_URL``, else
        ``https://ollama.com/v1``.
        No dedicated prefix; reached by catalog fallback (for example
        ``gpt-oss:...`` after OpenAI does not list it).

    Routing
    -------
    A model name is matched against family prefixes first, then confirmed
    against that provider's live ``/models`` catalog. If there is no prefix
    match, the preferred provider is not enabled, or that catalog does not
    list the model, every enabled provider is probed in this order: openai,
    anthropic, google, ollama_local, ollama_cloud. Successful
    ``model -> provider`` lookups are cached for ``cache_ttl`` seconds
    (default 1 hour).
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        anthropic_api_key: str | None = None,
        anthropic_base_url: str | None = None,
        google_api_key: str | None = None,
        google_base_url: str | None = None,
        ollama_local_api_key: str | None = None,
        ollama_local_base_url: str | None = None,
        ollama_cloud_api_key: str | None = None,
        ollama_cloud_base_url: str | None = None,
        model: str | None = None,
        cache_ttl: int = 3600,
        timeout: float = 90.0,
    ) -> None:
        """Create a client and enable every provider that has a key and URL.

        Args:
            openai_api_key: OpenAI secret. Falls back to ``OPENAI_API_KEY``.
            openai_base_url: OpenAI-compatible base URL. Falls back to
                ``OPENAI_BASE_URL``, then ``https://api.openai.com/v1``.
            anthropic_api_key: Anthropic secret. Falls back to
                ``ANTHROPIC_API_KEY``.
            anthropic_base_url: Anthropic OpenAI-compatible base URL. Falls
                back to ``ANTHROPIC_BASE_URL``, then
                ``https://api.anthropic.com/v1/``.
            google_api_key: Gemini secret. Falls back to ``GOOGLE_API_KEY``.
            google_base_url: Gemini OpenAI-compatible base URL. Falls back to
                ``GOOGLE_BASE_URL``, then
                ``https://generativelanguage.googleapis.com/v1beta/openai/``.
            ollama_local_api_key: Local Ollama placeholder key (typically
                ``"ollama"``). Falls back to ``OLLAMA_LOCAL_API_KEY``.
            ollama_local_base_url: Local Ollama OpenAI-compatible base URL.
                Falls back to ``OLLAMA_LOCAL_BASE_URL``, then
                ``http://localhost:11434/v1``.
            ollama_cloud_api_key: Ollama Cloud secret. Falls back to
                ``OLLAMA_CLOUD_API_KEY`` or ``OLLAMA_API_KEY``.
            ollama_cloud_base_url: Ollama Cloud OpenAI-compatible base URL.
                Falls back to ``OLLAMA_CLOUD_BASE_URL``, then
                ``https://ollama.com/v1``.
            model: Default model id used by ``chat()`` when none is passed.
            cache_ttl: Seconds to cache model-to-provider routes and catalogs.
                Defaults to 3600.
            timeout: Read timeout in seconds for chat and catalog calls.
                Defaults to 600. Local models on CPU often need more than
                20 seconds. Connect timeout stays at 10 seconds.
        """
        load_dotenv(override=True)
        self.model = model
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._clients: dict[str, OpenAI] = {}
        self._base_urls: dict[str, str] = {}
        self._skipped: dict[str, str] = {}
        self._catalogs: dict[str, tuple[float, set[str]]] = {}
        self._routes: dict[str, tuple[float, str, str, str]] = {}

        supplied = {
            "openai_api_key": openai_api_key,
            "openai_base_url": openai_base_url,
            "anthropic_api_key": anthropic_api_key,
            "anthropic_base_url": anthropic_base_url,
            "google_api_key": google_api_key,
            "google_base_url": google_base_url,
            "ollama_local_api_key": ollama_local_api_key,
            "ollama_local_base_url": ollama_local_base_url,
            "ollama_cloud_api_key": ollama_cloud_api_key,
            "ollama_cloud_base_url": ollama_cloud_base_url,
        }
        self._enable_providers(supplied)

    def _enable_providers(self, supplied: dict[str, str | None]) -> None:
        for name, spec in _PROVIDERS.items():
            api_key = _first_nonempty(supplied.get(spec["key_arg"]), _env(*spec["key_envs"]))
            base_url = _first_nonempty(
                supplied.get(spec["url_arg"]),
                _env(*spec["url_envs"]),
                spec["default_url"],
            )
            missing = []
            if not api_key:
                key_hint = " or ".join((spec["key_arg"], *spec["key_envs"]))
                missing.append(f"no api key (set {key_hint})")
            if not base_url:
                url_hint = " or ".join((spec["url_arg"], *spec["url_envs"]))
                missing.append(f"no base url (set {url_hint})")
            if missing:
                reason = "; ".join(missing)
                self._skipped[name] = reason
                logger.warning("skipping %s: %s", name, reason)
                continue

            self._clients[name] = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=spec["headers"],
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                max_retries=0 if name == "ollama_local" else 2,
            )
            self._base_urls[name] = base_url
            logger.info("enabled %s (%s)", name, base_url)

    def list_models(self) -> dict[str, list[str]]:
        """Return currently advertised model ids for each enabled provider.

        Refreshes each provider catalog. Providers that were skipped at init
        (missing key or URL) are omitted.
        """
        listed: dict[str, list[str]] = {}
        for name in self._probe_order():
            catalog = self._provider_catalog(name, force=True)
            if catalog is not None:
                listed[name] = sorted(catalog)
        return listed

    def chat(
        self,
        user_prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion to the provider that owns ``model``.

        Provide either ``user_prompt`` (optionally with ``system_prompt``) or a
        raw ``messages`` list, not both. Extra ``kwargs`` are forwarded to
        ``chat.completions.create``.

        Args:
            user_prompt: User message content. Required unless ``messages``
                is given.
            system_prompt: Optional system message, prepended when using
                ``user_prompt``.
            model: Model id to route. Defaults to the constructor ``model``.
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
            raise ValueError("No model specified; pass model= to chat() or to UniversalLLMClient()")

        provider, base_url, resolved_model = self._resolve(chosen_model)
        logger.info("resolved %r -> %s (%s)", resolved_model, provider, base_url)
        started = time.monotonic()
        response = self._clients[provider].chat.completions.create(
            model=resolved_model,
            messages=messages,
            **kwargs,
        )
        elapsed_s = time.monotonic() - started
        logger.info(
            "response from %s (%s) in %.2fs",
            resolved_model,
            provider,
            elapsed_s,
        )
        return response.choices[0].message.content

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

    def _resolve(self, model: str) -> tuple[str, str, str]:
        cached = self._cached_route(model)
        if cached is not None:
            provider, base_url, resolved_model = cached
            logger.info("cache hit: %r -> %s (%s)", resolved_model, provider, base_url)
            return provider, base_url, resolved_model

        preferred = _prefix_for(model)
        tried: list[str] = []

        if preferred is None:
            names = ", ".join(self._probe_order()) or "(none)"
            logger.info("no default prefix for %r; probing configured providers: %s", model, names)
        else:
            prefix, provider = preferred
            logger.info("default route: %r matches prefix %r -> %s", model, prefix, provider)
            if provider not in self._clients:
                reason = self._skipped.get(provider, "not enabled")
                logger.info("preferred provider %s is skipped (%s); falling back", provider, reason)
            else:
                match = self._lookup_on_provider(provider, model)
                tried.append(provider)
                if match is not None:
                    return self._store_route(model, provider, match)
                logger.info(
                    "%s did not list %r; falling back to remaining providers",
                    provider,
                    model,
                )

        for provider in self._probe_order():
            if provider in tried:
                continue
            match = self._lookup_on_provider(provider, model)
            tried.append(provider)
            if match is not None:
                return self._store_route(model, provider, match)

        skipped = ", ".join(f"{name} ({reason})" for name, reason in self._skipped.items()) or "none"
        tried_display = ", ".join(tried) or "none"
        raise ValueError(
            f"No configured provider listed model {model!r}. "
            f"Tried: {tried_display}. Skipped: {skipped}."
        )

    def _cached_route(self, model: str) -> tuple[str, str, str] | None:
        entry = self._routes.get(model)
        if entry is None:
            return None
        cached_at, provider, base_url, resolved_model = entry
        if time.monotonic() - cached_at > self.cache_ttl:
            logger.info("cache expired for %r", model)
            self._routes.pop(model, None)
            return None
        if provider not in self._clients:
            self._routes.pop(model, None)
            return None
        return provider, base_url, resolved_model

    def _store_route(self, requested: str, provider: str, resolved_model: str) -> tuple[str, str, str]:
        base_url = self._base_urls[provider]
        now = time.monotonic()
        entry = (now, provider, base_url, resolved_model)
        self._routes[requested] = entry
        self._routes[resolved_model] = entry
        return provider, base_url, resolved_model

    def _probe_order(self) -> list[str]:
        return [name for name in _PROBE_ORDER if name in self._clients]

    def _lookup_on_provider(self, provider: str, model: str) -> str | None:
        catalog = self._provider_catalog(provider)
        if catalog is None:
            return None
        if model in catalog:
            logger.info("%s catalog: found %r", provider, model)
            return model
        matches = sorted(model_id for model_id in catalog if model_id.startswith(model))
        if len(matches) == 1:
            logger.info("%s catalog: found unique prefix match %r for %r", provider, matches[0], model)
            return matches[0]
        if len(matches) > 1:
            logger.info("%s catalog: ambiguous prefix match for %r: %s", provider, model, matches[:8])
            return None
        logger.info("%s catalog: not found %r", provider, model)
        return None

    def _provider_catalog(self, provider: str, *, force: bool = False) -> set[str] | None:
        cached = self._catalogs.get(provider)
        if cached is not None and not force:
            fetched_at, ids = cached
            if time.monotonic() - fetched_at <= self.cache_ttl:
                return ids
            logger.info("%s catalog cache expired; refreshing", provider)

        client = self._clients.get(provider)
        if client is None:
            return None
        try:
            ids: set[str] = set()
            for item in client.models.list():
                for alias in _catalog_aliases(item.id):
                    ids.add(alias)
        except Exception as exc:
            logger.warning("skipping %s catalog: %s", provider, exc)
            self._catalogs.pop(provider, None)
            return None

        self._catalogs[provider] = (time.monotonic(), ids)
        logger.info("%s catalog: loaded %d model id(s)", provider, len(ids))
        return ids


__all__ = ["UniversalLLMClient"]
