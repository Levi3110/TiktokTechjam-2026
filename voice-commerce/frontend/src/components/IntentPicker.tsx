import type { Intent } from "../types";

export function IntentPicker({ onSelect }: { onSelect: (intent: Intent) => void }) {
  return (
    <div className="picker-overlay">
      <section className="picker-card">
        <div className="brand-mark">M</div>
        <p className="eyebrow">TRỢ LÝ MUA SẮM BẰNG GIỌNG NÓI</p>
        <h1>Hôm nay bạn muốn<br />tìm kiếm theo cách nào?</h1>
        <p className="picker-copy">Bạn có thể đổi ý bất cứ lúc nào trong cuộc trò chuyện.</p>
        <div className="intent-grid">
          <button className="intent-option buying" onClick={() => onSelect("buying")}>
            <span className="intent-icon">🛍</span>
            <span><strong>Tôi muốn mua</strong><small>Lọc nhanh theo nhu cầu và ngân sách</small></span>
            <b>→</b>
          </button>
          <button className="intent-option browsing" onClick={() => onSelect("browsing")}>
            <span className="intent-icon">◉</span>
            <span><strong>Tôi đang xem</strong><small>Khám phá và so sánh các lựa chọn</small></span>
            <b>→</b>
          </button>
        </div>
      </section>
    </div>
  );
}

