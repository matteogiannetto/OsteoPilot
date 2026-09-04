"""Retrying Ollama chat model wrapper."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

from langchain_ollama import ChatOllama
from langchain_ollama._utils import merge_auth_headers, parse_url_with_auth
from ollama import AsyncClient, Client

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_ollama_usage_limit_error(error: BaseException) -> bool:
    """Return true for Ollama account/session quota exhaustion errors."""
    status_code = getattr(error, "status_code", None)
    if status_code != 429:
        return False

    text_parts = [str(error)]
    response_error = getattr(error, "error", None)
    if isinstance(response_error, str):
        text_parts.append(response_error)
    error_text = " ".join(text_parts).lower()

    if "usage limit" in error_text:
        return True
    if "session limit" in error_text or "weekly limit" in error_text:
        return True
    return (
        "reached your" in error_text
        and "limit" in error_text
        and any(marker in error_text for marker in ("ollama", "upgrade", "extra usage"))
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer value for %s=%r", name, value)
        return default
    return max(parsed, 1)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Ignoring invalid float value for %s=%r", name, value)
        return default
    return max(parsed, 0.1)


class RetryingChatOllama(ChatOllama):
    """ChatOllama with retries, increasing HTTP timeouts, and JSON-schema output."""

    max_retries: int = _env_int("OLLAMA_MAX_RETRIES", 5)
    retry_initial_timeout: float = _env_float("OLLAMA_RETRY_INITIAL_TIMEOUT", 60.0)
    retry_timeout_multiplier: float = _env_float("OLLAMA_RETRY_TIMEOUT_MULTIPLIER", 2.0)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that validates Ollama responses against ``schema``."""
        kwargs["method"] = "json_schema"
        return super().with_structured_output(schema, **kwargs)

    def _timeout_for_attempt(self, attempt_index: int) -> float:
        return self.retry_initial_timeout * (
            self.retry_timeout_multiplier ** attempt_index
        )

    def _client_kwargs_for_timeout(
        self,
        timeout: float,
        *,
        async_client: bool,
    ) -> dict[str, Any]:
        client_kwargs = dict(self.client_kwargs or {})
        cleaned_url, auth_headers = parse_url_with_auth(self.base_url)
        merge_auth_headers(client_kwargs, auth_headers)

        specific_kwargs = (
            self.async_client_kwargs if async_client else self.sync_client_kwargs
        )
        if specific_kwargs:
            client_kwargs = {**client_kwargs, **specific_kwargs}

        client_kwargs["timeout"] = timeout
        return {"host": cleaned_url, **client_kwargs}

    def _set_sync_client_timeout(self, timeout: float) -> None:
        self._client = Client(**self._client_kwargs_for_timeout(timeout, async_client=False))

    def _set_async_client_timeout(self, timeout: float) -> None:
        self._async_client = AsyncClient(
            **self._client_kwargs_for_timeout(timeout, async_client=True)
        )

    def _with_ollama_retries(
        self,
        operation_name: str,
        operation: Callable[[], T],
    ) -> T:
        last_error: Exception | None = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            timeout = self._timeout_for_attempt(attempt - 1)
            self._set_sync_client_timeout(timeout)
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if is_ollama_usage_limit_error(exc):
                    logger.error(
                        "Ollama %s stopped because the session usage limit was reached: %s",
                        operation_name,
                        exc,
                    )
                    break
                if attempt >= total_attempts:
                    break
                logger.warning(
                    "Ollama %s failed on attempt %d/%d with timeout %.1fs; retrying: %s",
                    operation_name,
                    attempt,
                    total_attempts,
                    timeout,
                    exc,
                )
                time.sleep(min(2.0, 0.25 * attempt))

        assert last_error is not None
        raise last_error

    async def _with_ollama_retries_async(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        last_error: Exception | None = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            timeout = self._timeout_for_attempt(attempt - 1)
            self._set_async_client_timeout(timeout)
            try:
                return await operation()
            except Exception as exc:
                last_error = exc
                if is_ollama_usage_limit_error(exc):
                    logger.error(
                        "Ollama %s stopped because the session usage limit was reached: %s",
                        operation_name,
                        exc,
                    )
                    break
                if attempt >= total_attempts:
                    break
                logger.warning(
                    "Ollama %s failed on attempt %d/%d with timeout %.1fs; retrying: %s",
                    operation_name,
                    attempt,
                    total_attempts,
                    timeout,
                    exc,
                )
                await asyncio.sleep(min(2.0, 0.25 * attempt))

        assert last_error is not None
        raise last_error

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._with_ollama_retries(
            "generate",
            lambda: super(RetryingChatOllama, self)._generate(*args, **kwargs),
        )

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        return await self._with_ollama_retries_async(
            "generate",
            lambda: super(RetryingChatOllama, self)._agenerate(*args, **kwargs),
        )

    def _stream(self, *args: Any, run_manager: Any = None, **kwargs: Any) -> Iterator[Any]:
        stream_kwargs = {**kwargs, "run_manager": None}
        chunks = self._with_ollama_retries(
            "stream",
            lambda: list(
                super(RetryingChatOllama, self)._stream(
                    *args,
                    **stream_kwargs,
                )
            ),
        )
        for chunk in chunks:
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, verbose=self.verbose)
            yield chunk

    async def _astream(
        self,
        *args: Any,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        stream_kwargs = {**kwargs, "run_manager": None}

        async def collect_chunks() -> list[Any]:
            return [
                chunk
                async for chunk in super(RetryingChatOllama, self)._astream(
                    *args,
                    **stream_kwargs,
                )
            ]

        chunks = await self._with_ollama_retries_async("stream", collect_chunks)
        for chunk in chunks:
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, verbose=self.verbose)
            yield chunk
