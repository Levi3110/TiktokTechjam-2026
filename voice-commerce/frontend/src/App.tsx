import { FormEvent, useEffect, useRef, useState } from "react";
import { avatarSpeak, createAvatarOffer, createSession, piperSpeak, sendChat, transcribe } from "./api";
import { Avatar } from "./components/Avatar";
import { IntentPicker } from "./components/IntentPicker";
import { ProductCard } from "./components/ProductCard";
import type { Intent, Message } from "./types";

const welcome: Record<Intent, string> = {
  buying: "Chào bạn, mình là Mây. Hãy nói sản phẩm, ngân sách và điều bạn ưu tiên — mình sẽ lọc giúp bạn.",
  browsing: "Chào bạn, mình là Mây. Cứ tự nhiên khám phá nhé. Bạn muốn bắt đầu với danh mục nào?",
};

const getOrCreateUserId = () => {
  const key = "may-commerce-user-id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const created = crypto.randomUUID();
  window.localStorage.setItem(key, created);
  return created;
};

export default function App() {
  const [intent, setIntent] = useState<Intent | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");
  const [avatarStream, setAvatarStream] = useState<MediaStream | null>(null);
  const [avatarSession, setAvatarSession] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const peer = useRef<RTCPeerConnection | null>(null);
  const chatEnd = useRef<HTMLDivElement | null>(null);
  const userId = useRef(getOrCreateUserId());

  useEffect(() => chatEnd.current?.scrollIntoView({ behavior: "smooth" }), [messages, loading]);
  useEffect(() => () => peer.current?.close(), []);

  const chooseIntent = async (choice: Intent) => {
    setError("");
    try {
      const session = await createSession(choice, userId.current);
      setSessionId(session.session_id);
      setIntent(choice);
      setMessages([{ id: crypto.randomUUID(), role: "assistant", text: welcome[choice] }]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không thể tạo phiên.");
    }
  };

  const speak = async (text: string) => {
    setSpeaking(true);
    if (avatarSession) {
      try {
        await avatarSpeak(avatarSession, text);
        window.setTimeout(() => setSpeaking(false), Math.max(2200, text.length * 55));
        return;
      } catch {
        // Continue with browser speech when LiveTalking becomes unavailable.
      }
    }
    try {
      const audio = new Audio(URL.createObjectURL(await piperSpeak(text)));
      audio.onended = () => {
        URL.revokeObjectURL(audio.src);
        setSpeaking(false);
      };
      audio.onerror = () => setSpeaking(false);
      await audio.play();
      return;
    } catch {
      // Continue with built-in browser speech in lightweight mode.
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "vi-VN";
      utterance.rate = 1.02;
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(utterance);
    } else {
      setSpeaking(false);
    }
  };

  const submitMessage = async (raw: string) => {
    const text = raw.trim();
    if (!text || !sessionId || loading) return;
    setInput("");
    setError("");
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text }]);
    setLoading(true);
    try {
      const result = await sendChat(sessionId, text);
      setIntent(result.intent);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: result.answer,
          products: result.products,
          intentChanged: result.intent_changed,
        },
      ]);
      void speak(result.answer);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Có lỗi xảy ra.");
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submitMessage(input);
  };

  const toggleRecording = async () => {
    if (recording && recorder.current) {
      recorder.current.stop();
      setRecording(false);
      return;
    }
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      const chunks: BlobPart[] = [];
      mediaRecorder.ondataavailable = (event) => chunks.push(event.data);
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setLoading(true);
        try {
          const result = await transcribe(new Blob(chunks, { type: "audio/webm" }));
          setLoading(false);
          if (!result.text.trim()) {
            setError("Mình chưa nghe rõ. Bạn hãy thử nói lại gần microphone hơn.");
            return;
          }
          await submitMessage(result.text);
        } catch (reason) {
          setError(reason instanceof Error ? reason.message : "Không nhận diện được giọng nói.");
          setLoading(false);
        }
      };
      mediaRecorder.start();
      recorder.current = mediaRecorder;
      setRecording(true);
    } catch {
      setError("Không truy cập được microphone. Hãy cấp quyền và thử lại.");
    }
  };

  const connectAvatar = async () => {
    setError("");
    try {
      peer.current?.close();
      const connection = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      peer.current = connection;
      const remote = new MediaStream();
      connection.ontrack = (event) => {
        event.streams[0]?.getTracks().forEach((track) => remote.addTrack(track));
        setAvatarStream(remote);
      };
      connection.addTransceiver("video", { direction: "recvonly" });
      connection.addTransceiver("audio", { direction: "recvonly" });
      const offer = await connection.createOffer();
      await connection.setLocalDescription(offer);
      await new Promise<void>((resolve) => {
        if (connection.iceGatheringState === "complete") return resolve();
        const timeout = window.setTimeout(resolve, 2500);
        connection.addEventListener("icegatheringstatechange", () => {
          if (connection.iceGatheringState === "complete") {
            window.clearTimeout(timeout);
            resolve();
          }
        });
      });
      const localDescription = connection.localDescription;
      if (!localDescription) throw new Error("Không tạo được WebRTC offer");
      const answer = await createAvatarOffer({
        sdp: localDescription.sdp,
        type: localDescription.type,
      });
      await connection.setRemoteDescription({ type: "answer", sdp: answer.sdp });
      setAvatarSession(answer.sessionid);
    } catch (reason) {
      peer.current?.close();
      setError(`LiveTalking chưa sẵn sàng: ${reason instanceof Error ? reason.message : "lỗi kết nối"}`);
    }
  };

  const latestProducts = [...messages].reverse().find((message) => message.products?.length)?.products ?? [];

  return (
    <main className="app-shell">
      {!intent && <IntentPicker onSelect={chooseIntent} />}
      <header className="topbar">
        <div className="logo"><span>M</span><div><strong>Mây</strong><small>VOICE COMMERCE</small></div></div>
        <div className="mode-pill"><i /> {intent === "buying" ? "Đang mua sắm" : "Đang khám phá"}</div>
        <button className="avatar-connect" onClick={connectAvatar} disabled={Boolean(avatarSession)}>
          {avatarSession ? "Avatar online" : "Kết nối avatar"}
        </button>
      </header>

      <section className="workspace">
        <aside className="avatar-panel">
          <Avatar stream={avatarStream} speaking={speaking} connected={Boolean(avatarSession)} />
          <div className="hint-card">
            <span>✦</span>
            <p><strong>Mẹo nhỏ</strong>Nói “tôi muốn mua…” hoặc “tôi chỉ đang xem…” để đổi luồng tự động.</p>
          </div>
        </aside>

        <section className="conversation">
          <div className="conversation-head">
            <div><p className="eyebrow">CUỘC TRÒ CHUYỆN</p><h2>Mình có thể giúp gì cho bạn?</h2></div>
            <button onClick={() => intent && chooseIntent(intent)} title="Tạo phiên mới">↻</button>
          </div>
          <div className="messages">
            {messages.map((message) => (
              <div key={message.id} className={`message-row ${message.role}`}>
                {message.role === "assistant" && <div className="mini-avatar">M</div>}
                <div className="message-content">
                  {message.intentChanged && <span className="intent-change">✦ Đã đổi luồng theo ý định mới</span>}
                  <div className="bubble">{message.text}</div>
                  {message.products && message.products.length > 0 && (
                    <div className="mobile-products">
                      {message.products.map((product) => <ProductCard key={product.id} product={product} />)}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && <div className="message-row assistant"><div className="mini-avatar">M</div><div className="thinking"><i /><i /><i /></div></div>}
            <div ref={chatEnd} />
          </div>
          {error && <div className="error-banner">{error}</div>}
          <form className="composer" onSubmit={onSubmit}>
            <button type="button" className={`mic ${recording ? "recording" : ""}`} onClick={toggleRecording} aria-label="Thu âm">
              {recording ? "■" : "●"}
            </button>
            <input value={input} onChange={(event) => setInput(event.target.value)} placeholder={recording ? "Đang nghe…" : "Hỏi Mây về sản phẩm…"} disabled={loading} />
            <button className="send" type="submit" disabled={!input.trim() || loading}>↑</button>
          </form>
        </section>

        <aside className="products-panel">
          <div className="product-title"><div><p className="eyebrow">GỢI Ý CHO BẠN</p><h2>Sản phẩm nổi bật</h2></div><span>{latestProducts.length}</span></div>
          <div className="product-list">
            {latestProducts.length ? latestProducts.map((product) => <ProductCard key={product.id} product={product} />) : (
              <div className="empty-products"><span>⌁</span><p>Sản phẩm phù hợp sẽ xuất hiện tại đây sau khi bạn trò chuyện với Mây.</p></div>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
