#!/usr/bin/env python3
"""Publish a WAV file as a LiveKit microphone and print the STT callback."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

from livekit import rtc


def request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object")
    return value


async def publish(wav_path: Path, web_url: str) -> str:
    session_id = f"smoke-{uuid.uuid4()}"
    config = await asyncio.to_thread(
        request_json,
        f"{web_url}/api/livekit/token",
        {"session_id": session_id},
    )
    ready_deadline = time.monotonic() + 45
    ready_query = urllib.parse.urlencode({"session_id": session_id})
    while time.monotonic() < ready_deadline:
        status = await asyncio.to_thread(
            request_json,
            f"{web_url}/api/livekit/transcript?{ready_query}",
        )
        if status.get("error"):
            raise RuntimeError(str(status["error"]))
        if status.get("status") == "ready":
            break
        await asyncio.sleep(0.25)
    else:
        raise TimeoutError("Speech worker did not become ready")

    room = rtc.Room()
    await room.connect(str(config["url"]), str(config["token"]))
    source = rtc.AudioSource(16_000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    publication = await room.local_participant.publish_track(track, options)
    await asyncio.sleep(0.75)

    with wave.open(str(wav_path), "rb") as audio:
        if (
            audio.getframerate() != 16_000
            or audio.getnchannels() != 1
            or audio.getsampwidth() != 2
        ):
            raise ValueError("WAV must be mono 16 kHz signed 16-bit PCM")
        while chunk := audio.readframes(160):
            samples = len(chunk) // 2
            await source.capture_frame(rtc.AudioFrame(chunk, 16_000, 1, samples))
            await asyncio.sleep(samples / 16_000)

    await source.wait_for_playout()
    await room.local_participant.unpublish_track(publication.sid)
    await source.aclose()
    await room.disconnect()

    deadline = time.monotonic() + 30
    query = urllib.parse.urlencode({"session_id": session_id})
    while time.monotonic() < deadline:
        event = await asyncio.to_thread(
            request_json,
            f"{web_url}/api/livekit/transcript?{query}",
        )
        if event.get("transcript"):
            return str(event["transcript"])
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        await asyncio.sleep(0.25)
    raise TimeoutError("Timed out waiting for speech recognition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--web-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    print(asyncio.run(publish(args.wav, args.web_url.rstrip("/"))))


if __name__ == "__main__":
    main()
