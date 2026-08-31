import type { Product } from "../types";

const money = new Intl.NumberFormat("vi-VN", { style: "currency", currency: "VND" });

export function ProductCard({ product }: { product: Product }) {
  return (
    <article className="product-card">
      <div className="product-image-wrap">
        <img src={product.image} alt={product.name} className="product-image" />
        <span className="stock">Còn {product.stock}</span>
      </div>
      <div className="product-body">
        <p className="product-category">{product.category.replace("-", " ")}</p>
        <h3>{product.name}</h3>
        <p className="product-description">{product.description}</p>
        <div className="product-footer">
          <strong>{money.format(product.price)}</strong>
          <button type="button" aria-label={`Xem ${product.name}`}>↗</button>
        </div>
      </div>
    </article>
  );
}

