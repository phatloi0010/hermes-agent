"""Regression tests: failed-connect path must call adapter.disconnect().

When adapter.connect() returns False or raises, the adapter may have
allocated resources (aiohttp.ClientSession, poll tasks, child
subprocesses) before giving up. Without a defensive disconnect() call
these leak and surface as "Unclosed client session" warnings at
process exit (seen on the 2026-04-18 18:08:16 gateway restart).

The fix: gateway/run.py wraps each adapter connect() with a safety-net
call to _safe_adapter_disconnect() in the failure branches.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """A GatewayRunner shell that only needs to support _safe_adapter_disconnect."""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_safe_disconnect_times_out_and_continues(bare_runner, monkeypatch, caplog):
    """A wedged adapter disconnect must not block gateway shutdown."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.001")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(0.2)

    adapter.disconnect = AsyncMock(side_effect=hang)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)

    adapter.disconnect.assert_awaited_once()
    assert "Timed out after 0.0s while disconnecting feishu adapter" in caplog.text


@pytest.mark.asyncio
async def test_safe_disconnect_detaches_cancellation_swallowing_disconnect(
    bare_runner, monkeypatch, caplog
):
    """A disconnect that catches cancellation cannot block fatal recovery.

    ``asyncio.wait_for`` cancels its child at the deadline but then waits for
    it to finish.  A half-closed transport can catch that cancellation while
    unwinding, so the runner must detach the old close task and continue to the
    reconnect queue instead of waiting indefinitely.
    """
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()

    adapter.disconnect = AsyncMock(side_effect=swallow_cancellation)
    operation = asyncio.create_task(
        bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)
    )
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        assert "Timed out after 0.0s while disconnecting feishu adapter" in caplog.text
    finally:
        # The implementation must detach rather than abandon the old task.
        # Release it here so this test leaves no cancellation-swallowing task
        # behind when it runs against the pre-fix implementation.
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_safe_disconnect_slack_releases_lock_when_workspace_client_close_raises(
    bare_runner, monkeypatch
):
    """Regression for #147845: RaisingClient.close() during Slack disconnect must still
    release the platform lock and clear transient state."""
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter

    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.05")
    released: list[str] = []

    class RaisingClient:
        def close(self):
            raise RuntimeError("workspace client close failed")

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock(client=RaisingClient())
    adapter._platform_lock_scope = "slack-app-token"
    adapter._platform_lock_identity = "xapp-fake"
    adapter._release_platform_lock = lambda: released.append("released")

    await bare_runner._safe_adapter_disconnect(adapter, Platform.SLACK)

    assert released == ["released"]
    assert adapter._platform_lock_identity is None
    assert adapter._app is None


@pytest.mark.asyncio
async def test_safe_disconnect_slack_releases_lock_and_finishes_cleanup_when_seal_stream_hangs(
    bare_runner, monkeypatch
):
    """Regression for #147845: when _seal_stream hangs waiting on chat.stopStream and bounded
    teardown cancels it, remaining cleanup and platform-lock release must still run."""
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter

    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.05")
    released: list[str] = []
    closed_clients: list[str] = []

    class HangingStreamClient:
        async def chat_stopStream(self, **kwargs):
            await asyncio.Event().wait()

    class TeamClient:
        async def close(self):
            closed_clients.append("team")

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._active_streams = {"stream-1": {"ts": "1.0", "sent": ""}}
    adapter._team_clients = {"T1": TeamClient()}
    adapter._get_client = lambda *args, **kwargs: HangingStreamClient()
    adapter._platform_lock_scope = "slack-app-token"
    adapter._platform_lock_identity = "xapp-fake"
    adapter._release_platform_lock = lambda: released.append("released")

    await bare_runner._safe_adapter_disconnect(adapter, Platform.SLACK)

    assert released == ["released"]
    assert closed_clients == ["team"]
    assert len(adapter._active_streams) == 0
    assert adapter._team_clients == {}
    assert adapter._platform_lock_identity is None

