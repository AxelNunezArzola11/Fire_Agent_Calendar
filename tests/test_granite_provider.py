"""Granite provider + Qwen failover tests (scripted transports only).

No live calls here. A scripted success is never reported as live.
Dummy credential "DUMMY-KEY" is fixture text, never a real secret.
"""

import pytest

from src.calendar.agents import granite as granite_mod
from src.calendar.agents import providers
from src.calendar.events import qwen as qwen_mod


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class ScriptedPost:
    """Route-aware fake: iam + generation scripts, full call log."""

    def __init__(self, iam, generation):
        self._iam = list(iam)
        self._generation = list(generation)
        self.calls = []

    def __call__(self, url, *, headers, json_data=None, form_data=None,
                 timeout):
        kind = "iam" if "identity/token" in url else "generation"
        self.calls.append({"kind": kind, "url": url, "headers": headers,
                           "json": json_data, "form": form_data,
                           "timeout": timeout})
        queue = self._iam if kind == "iam" else self._generation
        if not queue:
            raise AssertionError("over-called: script exhausted")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _config(**over):
    args = {"api_key": "DUMMY-KEY", "project_id": "DUMMY-PROJECT",
            "url": "https://example.invalid", "model": "ibm/granite-4-h-small",
            "timeout_s": 5.0, "max_attempts": 2}
    args.update(over)
    return granite_mod.GraniteConfig(**args)


def _chat(text, usage=None):
    body = {"choices": [{"message": {"content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return FakeResponse(200, body)


def _ok_post(text="hello granite"):
    return ScriptedPost(
        iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
        generation=[_chat(text, {"prompt_tokens": 10,
                                 "completion_tokens": 5})])


class TestGraniteProvider:
    def test_success_path(self):
        post = _ok_post()
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert attempt.ok and attempt.text == "hello granite"
        assert attempt.provider == "granite"
        assert [c["kind"] for c in post.calls] == ["iam", "generation"]
        iam, gen = post.calls
        assert iam["form"]["grant_type"].endswith("apikey")
        assert iam["form"]["apikey"] == "DUMMY-KEY"
        assert gen["headers"]["Authorization"] == "Bearer DUMMY-TOKEN"
        assert gen["json"]["model_id"] == "ibm/granite-4-h-small"
        assert gen["json"]["project_id"] == "DUMMY-PROJECT"
        assert gen["json"]["parameters"]["decoding_method"] == "greedy"
        assert [m["role"] for m in gen["json"]["messages"]] == [
            "system", "user"]
        assert "text/chat" in gen["url"]
        assert "version=2024-05-01" in gen["url"]
        assert iam["timeout"] == gen["timeout"] == 5.0

    def test_missing_credentials_unavailable(self):
        provider = granite_mod.GraniteProvider(_config(api_key=""))
        usable, reason = provider.available()
        assert usable is False
        assert "WATSONX_API_KEY" in reason
        assert "DUMMY" not in reason
        attempt = provider.complete("p", "s")
        assert attempt.ok is False

    def test_iam_rejected(self):
        post = ScriptedPost(iam=[FakeResponse(401, {})], generation=[])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert attempt.ok is False
        assert "HTTP 401" in attempt.error
        assert "DUMMY" not in attempt.error

    def test_invalid_output(self):
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[FakeResponse(200, {})])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert attempt.ok is False
        assert "no choices" in attempt.error

    def test_timeout_bounded(self):
        import requests

        post = ScriptedPost(iam=[requests.Timeout("slow")],
                            generation=[])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert attempt.ok is False
        assert "Timeout" in attempt.error
        assert len(post.calls) == 1  # iam faults are not retried

    def test_retry_once_then_success(self, monkeypatch):
        monkeypatch.setattr(granite_mod, "RETRY_WAIT_S", 0.0)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[FakeResponse(429, {}),
                        _chat("recovered")])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert attempt.ok and attempt.text == "recovered"
        assert len(post.calls) == 3

    def test_secret_shaped_errors_redacted(self):
        def _evil(url, *, headers, json_data=None, form_data=None,
                  timeout):
            raise RuntimeError("store api_key=DUMMY-KEY failed")

        attempt = granite_mod.GraniteProvider(
            _config(), post=_evil).complete("p", "s")
        assert attempt.ok is False
        assert "DUMMY-KEY" not in attempt.error


class TestFailover:
    def _dead_qwen(self):
        return providers.QwenProvider(
            config=qwen_mod.QwenConfig(enabled=False))

    def test_qwen_down_granite_up(self):
        granite = granite_mod.GraniteProvider(_config(), post=_ok_post())
        attempts = providers.ProviderChain(
            [self._dead_qwen(), granite]).complete("p", "s")
        assert [a.ok for a in attempts] == [False, True]
        assert attempts[1].provider == "granite"

    def test_both_down(self):
        granite = granite_mod.GraniteProvider(_config(api_key=""))
        attempts = providers.ProviderChain(
            [self._dead_qwen(), granite]).complete("p", "s")
        assert [a.ok for a in attempts] == [False, False]

    def test_neither_configured(self):
        attempts = providers.ProviderChain(
            [self._dead_qwen(),
             granite_mod.GraniteProvider(_config(api_key=""))]).complete(
                 "p", "s")
        assert all("unavailable" in a.error for a in attempts)

    def test_coordinator_falls_back(self):
        from src.calendar.agents import coordinator

        granite = granite_mod.GraniteProvider(_config(api_key=""))
        result = coordinator.analyze(
            "event", "ev-1", {"kind": "event", "event_id": "ev-1"},
            chain=providers.ProviderChain([self._dead_qwen(), granite]))
        assert result.steps[0].status == "fallback"
        assert "granite" in result.steps[0].fallback_reason

    def test_default_chain_order(self):
        chain = providers.default_chain()
        assert [p.name for p in chain.providers] == ["qwen", "granite"]


class TestGraniteShapes:
    def test_iam_non_json(self):
        post = ScriptedPost(iam=[FakeResponse(200, ValueError("x"))],
                            generation=[])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "not JSON" in attempt.error

    def test_iam_non_dict(self):
        post = ScriptedPost(iam=[FakeResponse(200, ["x"])],
                            generation=[])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "lacks access_token" in attempt.error

    @pytest.mark.parametrize("body", [{"choices": {}},
                                      {"choices": "x"},
                                      {"choices": [[]]},
                                      {"choices": [{"n": 1}]},
                                      {"choices": [{"message": {"content": 7}}]},
                                      {}, []])
    def test_generation_shapes_rejected(self, body):
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[FakeResponse(200, body)])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "no choices" in attempt.error

    def test_missing_text_retries_then_empty(self, monkeypatch):
        monkeypatch.setattr(granite_mod, "_sleep", lambda s: None)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[_chat(""), _chat("  ")])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "empty response" in attempt.error
        assert attempt.retries == 1

    def test_whitespace_text_is_empty(self, monkeypatch):
        monkeypatch.setattr(granite_mod, "_sleep", lambda s: None)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[_chat("   "), _chat("  ")])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "empty response" in attempt.error
        assert attempt.retries == 1

    def test_empty_then_text_recovers(self, monkeypatch):
        monkeypatch.setattr(granite_mod, "_sleep", lambda s: None)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[_chat(""),
                        _chat("hello", {"prompt_tokens": 12,
                                        "completion_tokens": 3})])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert attempt.ok and attempt.text == "hello"
        assert attempt.retries == 1
        assert attempt.response_chars == 5
        assert (attempt.input_tokens, attempt.generated_tokens) == (12, 3)

    def test_retry_exhausted(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(granite_mod, "_sleep", sleeps.append)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[FakeResponse(429, {}), FakeResponse(503, {})])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "HTTP 503" in attempt.error
        assert len(post.calls) == 3 and sleeps == [2.0]

    def test_non_retryable_no_sleep(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(granite_mod, "_sleep", sleeps.append)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[FakeResponse(400, {})])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "HTTP 400" in attempt.error
        assert sleeps == [] and len(post.calls) == 2

    def test_single_attempt_no_sleep(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(granite_mod, "_sleep", sleeps.append)
        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[FakeResponse(429, {})])
        attempt = granite_mod.GraniteProvider(
            _config(max_attempts=1), post=post).complete("p", "s")
        assert "HTTP 429" in attempt.error
        assert sleeps == []

    def test_generation_transport_fault(self):
        import requests

        post = ScriptedPost(
            iam=[FakeResponse(200, {"access_token": "DUMMY-TOKEN"})],
            generation=[requests.ConnectionError("down")])
        attempt = granite_mod.GraniteProvider(
            _config(), post=post).complete("p", "s")
        assert "ConnectionError" in attempt.error
        assert len(post.calls) == 2  # faults are not retried

    def test_oversize_prompt_rejected(self):
        provider = granite_mod.GraniteProvider(_config())
        attempt = provider.complete("x" * 100001, "")
        assert "size budget" in attempt.error

    def test_error_message_redacted_not_dropped(self, monkeypatch):
        provider = granite_mod.GraniteProvider(_config(), post=_ok_post())

        def _boom():
            raise RuntimeError('store api_key="DUMMY-KEY" failed')

        monkeypatch.setattr(provider, "_iam_token", _boom)
        attempt = provider.complete("p", "s")
        assert "RuntimeError" in attempt.error  # type kept (diagnosable)
        assert "DUMMY-KEY" not in attempt.error  # value scrubbed
        assert "[redacted]" in attempt.error  # marker proves redaction


class TestGraniteConfig:
    @pytest.mark.parametrize("field", ["api_key", "project_id", "url",
                                       "model"])
    def test_missing_fields_unavailable(self, field):
        provider = granite_mod.GraniteProvider(_config(**{field: ""}))
        usable, reason = provider.available()
        assert usable is False
        assert "DUMMY" not in reason

    @pytest.mark.parametrize("timeout", [0.0, -1.0, "30"])
    def test_bad_timeout_unavailable(self, timeout):
        provider = granite_mod.GraniteProvider(_config(timeout_s=timeout))
        assert provider.available()[0] is False

    def test_attempts_clamped(self):
        config = granite_mod.GraniteConfig(
            api_key="k", project_id="p", url="u", model="m",
            timeout_s=5.0, max_attempts=99)
        assert config.max_attempts == 99  # direct construction untouched
        usable, _ = granite_mod.GraniteProvider(config).available()
        assert usable is False  # out-of-range rejected at status time

    def test_qwen_raises_granite_tried(self):
        class _Raising(providers.Provider):
            name = "qwen"

            def available(self):
                return True, "claims usable"

            def complete(self, prompt, system):
                raise RuntimeError("qwen exploded")

        granite = granite_mod.GraniteProvider(_config(), post=_ok_post())
        attempts = providers.ProviderChain(
            [_Raising(), granite]).complete("p", "s")
        assert [a.ok for a in attempts] == [False, True]
        assert "exploded" in attempts[0].error
