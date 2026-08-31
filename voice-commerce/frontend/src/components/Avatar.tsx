import { useEffect, useRef } from "react";

interface Props {
  stream: MediaStream | null;
  speaking: boolean;
  connected: boolean;
}

export function Avatar({ stream, speaking, connected }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream]);

  return (
    <div className={`avatar-stage ${speaking ? "is-speaking" : ""}`}>
      {stream ? (
        <video ref={videoRef} autoPlay playsInline />
      ) : (
        <div className="avatar-fallback">
          <div className="avatar-halo" />
          <div className="avatar-face"><span>✦</span></div>
        </div>
      )}
      <div className="avatar-status">
        <i className={connected ? "online" : "demo"} />
        {connected ? "LiveTalking đã kết nối" : "Mây · browser voice"}
      </div>
    </div>
  );
}
