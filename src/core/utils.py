"""
Lab 11 — Helper Utilities
"""
from __future__ import annotations

import asyncio

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

# Lỗi tạm thời từ nhà cung cấp (free tier rất hay gặp) — thử lại thay vì bỏ cuộc.
#   OpenRouter (Blue): 429 "temporarily rate-limited upstream"
#   Google AI Studio (Red): 429 RESOURCE_EXHAUSTED / 503 UNAVAILABLE
_TRANSIENT_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "Too Many Requests",
    "rate-limited",
    "503",
    "UNAVAILABLE",
    "overloaded",
    "500",
    "INTERNAL",
)
_MAX_ATTEMPTS = 6
_BASE_DELAY_SECONDS = 5.0
_MAX_DELAY_SECONDS = 60.0

# Hạn mức NGÀY của free tier (vd Google AI Studio: 20 request/ngày/model).
# Thử lại không giúp gì — phải đợi reset theo ngày, đổi model, hoặc bật billing.
_DAILY_QUOTA_MARKERS = (
    "PerDay",
    "exceeded your current quota",
    "free_tier_requests",
    "per day",
)


def _is_daily_quota(exc: BaseException) -> bool:
    """True nếu lỗi là hết hạn mức theo NGÀY (thử lại là vô ích)."""
    text = f"{exc}"
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def _is_transient(exc: BaseException) -> bool:
    """True nếu lỗi có khả năng chỉ là tạm thời (nên thử lại)."""
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _TRANSIENT_MARKERS)


async def _with_retry(call, *args, **kwargs):
    """Gọi ``call`` và thử lại với backoff khi gặp lỗi tạm thời.

    Không có retry thì chỉ một lần 429/503 là đủ làm hỏng cả lượt chạy: crash
    ngay ở smoke test của ``main.py --part 4``, hoặc ghi nhầm ``layer: "error"``
    (kèm ``leaked: false``) vào ``attack_results.json`` — tức là bằng chứng sai.
    """
    delay = _BASE_DELAY_SECONDS
    last_error: BaseException | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await call(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_daily_quota(exc):
                print(
                    "    [quota] Hết hạn mức NGÀY của free tier — KHÔNG thử lại.\n"
                    "    Google AI Studio free tier: 20 request/ngày/model (reset theo ngày).\n"
                    "    Cách xử lý: đợi reset, đổi model (mỗi model có quota riêng),\n"
                    "    hoặc bật billing cho key. Xem https://ai.dev/rate-limit",
                    flush=True,
                )
                raise
            if not _is_transient(exc):
                raise
            last_error = exc
            if attempt == _MAX_ATTEMPTS:
                break
            print(
                f"    [retry {attempt}/{_MAX_ATTEMPTS - 1}] "
                f"{type(exc).__name__} — chờ {delay:.0f}s rồi thử lại",
                flush=True,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY_SECONDS)

    raise last_error  # type: ignore[misc]


async def _chat_adk(runner, app_name: str, user_message: str, session_id):
    """Nhánh Google ADK (Red / Red Advance khi RED_TEAM_PROVIDER=gemini)."""
    from google.genai import types

    user_id = "student"

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response, session


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).

    Tự động thử lại với backoff khi nhà cung cấp trả lỗi tạm thời (429 / 503),
    để free tier không làm hỏng kết quả thật.
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await _with_retry(runner.chat, agent, user_message)
        return text, None

    return await _with_retry(
        _chat_adk, runner, runner.app_name, user_message, session_id
    )
