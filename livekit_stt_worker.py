#!/usr/bin/env python3
"""Receive microphone audio from one LiveKit room and transcribe it locally."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from livekit import api, rtc


LOGGER = logging.getLogger("livekit-stt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--callback", default="http://127.0.0.1:8765/api/livekit/transcript")
    parser.add_argument("--callback-secret", required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:7880")
    parser.add_argument("--api-key", default="devkey")
    parser.add_argument("--api-secret", default="secret")
    parser.add_argument("--model", default=os.getenv("LIVEKIT_STT_MODEL", "base"))
    parser.add_argument(
        "--language",
        default=os.getenv("LIVEKIT_STT_LANGUAGE", "auto"),
        help="ISO language code or 'auto' for multilingual speech detection.",
    )
    parser.add_argument("--model-cache", default=".models/faster-whisper")
    return parser.parse_args()


def post_event(args: argparse.Namespace, **payload: object) -> None:
    body = json.dumps({"session_id": args.session_id, **payload}).encode()
    request = urllib.request.Request(
        args.callback,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Worker-Secret": args.callback_secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except Exception as exc:  # The browser can reconnect even if one status update is lost.
        LOGGER.warning("callback failed: %s", exc)


async def transcribe_stream(
    stream: rtc.AudioStream,
    model: WhisperModel,
    args: argparse.Namespace,
) -> None:
    chunks: list[bytes] = []
    try:
        async for event in stream:
            chunks.append(bytes(event.frame.data))
    finally:
        await stream.aclose()

    if not chunks:
        post_event(args, status="ready", error="No microphone audio was received")
        return

    samples = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(samples) / 16_000
    if duration < 0.35:
        post_event(args, status="ready", error="Recording was too short")
        return

    post_event(args, status="transcribing", duration=round(duration, 2))
    loop = asyncio.get_running_loop()

    def run_whisper() -> str:
        segments, _ = model.transcribe(
            samples,
            language=None if args.language == "auto" else args.language,
            beam_size=3,
            vad_filter=True,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    text = await loop.run_in_executor(None, run_whisper)
    if text:
        post_event(args, status="ready", transcript=text)
    else:
        post_event(args, status="ready", error="No speech was detected")


async def run(args: argparse.Namespace) -> None:
    model_cache = Path(args.model_cache).resolve()
    post_event(args, status="loading")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        download_root=str(model_cache),
        local_files_only=True,
    )
    token = (
        api.AccessToken(args.api_key, args.api_secret)
        .with_identity(f"stt-{args.session_id[:48]}")
        .with_name("Local Whisper STT")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=args.room,
                can_publish=False,
                can_subscribe=True,
                can_publish_data=False,
            )
        )
        .to_jwt()
    )
    room = rtc.Room()
    tasks: set[asyncio.Task[None]] = set()
    streams: dict[str, rtc.AudioStream] = {}
    stopped = asyncio.Event()

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or participant.identity.startswith("stt-"):
            return
        LOGGER.info("microphone subscribed from %s", participant.identity)
        stream = rtc.AudioStream(track, sample_rate=16_000, num_channels=1)
        streams[track.sid] = stream
        task = asyncio.create_task(transcribe_stream(stream, model, args))
        tasks.add(task)
        task.add_done_callback(lambda finished, sid=track.sid: (tasks.discard(finished), streams.pop(sid, None)))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        stream = streams.get(track.sid)
        if stream is not None:
            LOGGER.info("microphone stopped by %s", participant.identity)
            asyncio.create_task(stream.aclose())

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if publication.kind == rtc.TrackKind.KIND_AUDIO:
            LOGGER.info("microphone published by %s", participant.identity)
            publication.set_subscribed(True)

    @room.on("disconnected")
    def on_disconnected(*_args: object) -> None:
        stopped.set()

    await room.connect(args.url, token, rtc.RoomOptions(auto_subscribe=True))
    LOGGER.info("connected to room %s", args.room)
    post_event(args, status="ready")
    try:
        await asyncio.wait_for(stopped.wait(), timeout=3_600)
    except TimeoutError:
        LOGGER.info("worker timeout")
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        LOGGER.exception("speech worker failed")
        post_event(args, status="ready", error=f"Speech recognition failed: {exc}")


if __name__ == "__main__":
    main()
