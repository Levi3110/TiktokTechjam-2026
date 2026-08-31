import type { ChatResponse, Intent } from "./types";

const parse = async <T,>(response: Response): Promise<T> => {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
};

export const createSession = (initialIntent: Intent, userId: string) =>
  fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, initial_intent: initialIntent }),
  }).then((response) => parse<{ session_id: string }>(response));

export const sendChat = (sessionId: string, message: string) =>
  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  }).then((response) => parse<ChatResponse>(response));

export const transcribe = async (audio: Blob) => {
  const form = new FormData();
  form.append("audio", audio, "voice.webm");
  const response = await fetch("/api/transcribe", { method: "POST", body: form });
  return parse<{ text: string }>(response);
};

export const createAvatarOffer = (sdp: RTCSessionDescriptionInit) =>
  fetch("/api/avatar/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sdp),
  }).then((response) => parse<RTCSessionDescriptionInit & { sessionid: string }>(response));

export const avatarSpeak = (sessionId: string, text: string) =>
  fetch("/api/avatar/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text, interrupt: true }),
  }).then((response) => parse<Record<string, unknown>>(response));

export const piperSpeak = async (text: string) => {
  const response = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error("Piper unavailable");
  return response.blob();
};
