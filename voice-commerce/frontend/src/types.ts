export type Intent = "buying" | "browsing";

export interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  currency: string;
  description: string;
  attributes: Record<string, string | string[]>;
  image: string;
  stock: number;
}

export interface ChatResponse {
  session_id: string;
  intent: Intent;
  intent_changed: boolean;
  answer: string;
  products: Product[];
  extracted: Record<string, unknown>;
  memory_used: string[];
  debug: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  products?: Product[];
  intentChanged?: boolean;
}

