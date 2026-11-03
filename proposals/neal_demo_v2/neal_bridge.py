"""Private, append-only Modal Dict bridge; no SDK import or calls on import.

SDK requests run serially in one daemon broker thread. Its unknown disposition
latches refusal; cancellation of an await never permits another command. The
separate parent process retains the native-generation watchdog independently.
Dict receipts prove transport state only, never actual action or game success.
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import queue
import threading
import time

from neal_limits import (ACTION_RESPONSE_SECONDS, ASSISTANT_READY_SECONDS, BRIDGE_NAME,
    SESSION_ID, MAX_ASSISTANT_COMMANDS, MAX_MESSAGE_BYTES, MAX_PNG_BYTES, MAX_VISIBLE_DOM_BYTES, NOT_AFTER_UNIX)


IDENTITY_FIELDS = ("session_id", "actor", "level", "seq", "observation_sha256", "frame_id", "task_phase")


def bounded_json(value):
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) >= MAX_MESSAGE_BYTES:
        raise ValueError("bridge message exceeds fixed byte cap")
    return encoded


def validate_observation(value):
    if (not isinstance(value, dict) or value.get("session_id") != SESSION_ID
            or value.get("actor") != "assistant" or value.get("level") not in (1, 3, 5)
            or type(value.get("seq")) is not int or not 1 <= value["seq"] <= MAX_ASSISTANT_COMMANDS
            or value.get("task_phase") != "active"
            or not isinstance(value.get("frame_id"), str) or not value["frame_id"]
            or not isinstance(value.get("observation_sha256"), str) or len(value["observation_sha256"]) != 64):
        raise ValueError("invalid assistant observation identity")
    if len(json.dumps(value.get("visible_dom"), allow_nan=False).encode()) > MAX_VISIBLE_DOM_BYTES:
        raise ValueError("visible DOM exceeds bridge cap")
    png = base64.b64decode(value["png_base64"], validate=True)
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or not 0 < len(png) <= MAX_PNG_BYTES:
        raise ValueError("invalid bounded PNG observation")
    bounded_json(value)
    canonical = {key: item for key, item in value.items() if key != "observation_sha256"}
    if hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != value["observation_sha256"]:
        raise ValueError("observation payload does not match its hash")
    return png


def validate_action_identity(command, observation):
    if not isinstance(command, dict) or set(command) != set(IDENTITY_FIELDS) | {"native_action_json"}:
        raise ValueError("exact command envelope required")
    if any(command[field] != observation[field] or type(command[field]) is not type(observation[field]) for field in IDENTITY_FIELDS):
        raise ValueError("command does not match pending observation/frame/phase")
    if not isinstance(command["native_action_json"], str) or len(command["native_action_json"].encode()) > 32768:
        raise ValueError("bounded native action JSON text required")
    bounded_json(command)
    return command


class BrokerUnknown(RuntimeError):
    pass


class SerialBroker:
    """One outstanding SDK request; never block the asyncio owner or watchdog."""
    def __init__(self):
        self.jobs = queue.Queue(maxsize=1)
        self.unknown = False
        self.active = False
        self.thread = threading.Thread(target=self._work, name="neal-private-dict-broker", daemon=True)
        self.thread.start()

    def _work(self):
        while True:
            loop, future, operation = self.jobs.get()
            try:
                value, error = operation(), None
            except BaseException as exc:
                value, error = None, type(exc).__name__
            def deliver(value=value, error=error, future=future):
                if not future.done():
                    if error is None:
                        future.set_result(value)
                    else:
                        future.set_exception(BrokerUnknown("private Dict request failed: " + error))
            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                # Parent process still owns termination if the event loop ended.
                return

    async def call(self, operation, timeout=5.0):
        if self.unknown or self.active:
            raise BrokerUnknown("broker is consumed by a pending/unknown request")
        if not 0 < timeout <= 5:
            raise ValueError("bounded broker timeout required")
        self.active = True
        future = asyncio.get_running_loop().create_future()
        self.jobs.put_nowait((asyncio.get_running_loop(), future, operation))
        try:
            value = await asyncio.wait_for(asyncio.shield(future), timeout)
        except BaseException:
            self.unknown = True
            raise
        finally:
            self.active = False
        return value


class AssistantBridge:
    def __init__(self, store, source_manifest_sha256, *, publish=None, broker=None, clock=time.monotonic):
        self.store, self.source_sha, self.publish = store, source_manifest_sha256, publish
        self.broker = broker or SerialBroker()
        self.clock = clock
        self.last_seq = 0
        self.pending = self.failed = False
        self.events = []

    def _record(self, kind, **fields):
        self.events.append({"kind": kind, "monotonic": self.clock(), **fields})
        if len(self.events) > 256:
            self.failed = True
            raise ValueError("bridge event cap reached")
        if self.publish is not None:
            self.publish(list(self.events))

    async def wait_ready(self):
        deadline = self.clock() + ASSISTANT_READY_SECONDS
        try:
            while self.clock() < deadline:
                ready = await self.broker.call(lambda: self.store.get(SESSION_ID + ":ready"),
                    min(5, deadline - self.clock()))
                if ready is not None:
                    if (not isinstance(ready, dict) or ready.get("session_id") != SESSION_ID
                            or ready.get("actor") != "assistant" or ready.get("source_manifest_sha256") != self.source_sha
                            or ready.get("ready") is not True
                            or type(ready.get("expires_unix")) not in (int, float)
                            or not time.time() < ready["expires_unix"] <= NOT_AFTER_UNIX):
                        raise ValueError("assistant readiness identity/expiry does not match")
                    self._record("assistant_ready_observed", expires_unix=ready["expires_unix"])
                    return ready
                await asyncio.sleep(min(0.25, max(0, deadline - self.clock())))
            raise TimeoutError("assistant not ready within30s; no model load")
        except BaseException:
            self.failed = True
            raise

    async def exchange(self, observation, timeout_s=30):
        validate_observation(observation)
        if (self.failed or self.pending or observation["seq"] != self.last_seq + 1
                or not 0 < timeout_s <= ACTION_RESPONSE_SECONDS):
            raise ValueError("one next bounded observation only")
        self.pending = True
        deadline = self.clock() + timeout_s
        seq = observation["seq"]
        try:
            created = await self.broker.call(lambda: self.store.put(
                SESSION_ID + f":observation:{seq}", observation, skip_if_exists=True), min(5, timeout_s))
            if created is not True:
                raise ValueError("observation key already consumed; no overwrite")
            self._record("observation_published", seq=seq, level=observation["level"],
                         frame_id=observation["frame_id"], observation_sha256=observation["observation_sha256"])
            while self.clock() < deadline:
                command = await self.broker.call(lambda: self.store.get(SESSION_ID + f":action:{seq}"),
                    min(5, deadline - self.clock()))
                if command is not None:
                    validate_action_identity(command, observation)
                    self.last_seq = seq
                    self._record("command_received", seq=seq,
                                 command_sha256=hashlib.sha256(bounded_json(command)).hexdigest())
                    return command
                await asyncio.sleep(min(0.25, max(0, deadline - self.clock())))
            raise TimeoutError("assistant response timeout; no retry or next level")
        except BaseException:
            self.failed = True
            self._record("exchange_failed_or_unknown", seq=seq)
            raise
        finally:
            self.pending = False


def private_store():
    """Existing workspace SDK authentication; no secret or public route."""
    import modal
    return modal.Dict.from_name(BRIDGE_NAME, create_if_missing=False, environment_name="main")
