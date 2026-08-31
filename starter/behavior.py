from __future__ import annotations

import json
import os
import re
import ssl
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlencode, urlparse


class BehaviorStore:
    """Small local, anonymous preference store used only by the demo UI."""

    def __init__(self, path: str | Path = ".cache/user_behavior.json") -> None:
        self.path = Path(path)
        self.lock = RLock()
        self.records: dict[str, list[dict[str, Any]]] = {}
        if self.path.exists():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    self.records = {
                        str(key): list(items)[-20:]
                        for key, items in value.items()
                        if isinstance(items, list)
                    }
            except (OSError, json.JSONDecodeError):
                self.records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix="behavior-", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self.records, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            Path(temporary).replace(self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def record(self, client_id: str, product: dict[str, Any], size: str) -> None:
        categories = product.get("categories") or []
        with self.lock:
            history = self.records.setdefault(client_id, [])
            history.append(
                {
                    "parent_asin": str(product.get("parent_asin", "")),
                    "title": str(product.get("title", ""))[:240],
                    "size": size[:24],
                    "store": str(product.get("store", ""))[:100],
                    "category": str(categories[-1])[:100] if categories else "",
                    "price": product.get("price"),
                }
            )
            history[:] = history[-20:]
            self._save()

    def summary(self, client_id: str) -> str:
        with self.lock:
            history = list(self.records.get(client_id, []))
        if not history:
            return ""
        categories = Counter(item.get("category") for item in history if item.get("category"))
        sizes = Counter(item.get("size") for item in history if item.get("size"))
        recent = "; ".join(
            f"{item.get('title', '')} (size {item.get('size', 'unspecified')})"
            for item in history[-3:]
        )
        pieces = [f"Previously confirmed products: {recent}."]
        if categories:
            pieces.append(f"Most selected category: {categories.most_common(1)[0][0]}.")
        if sizes:
            pieces.append(f"Most selected size: {sizes.most_common(1)[0][0]}.")
        return " ".join(pieces)


class ProductImageCache:
    """Optional DDG image lookup with a bounded local cache and offline fallback."""

    MAX_BYTES = 8 * 1024 * 1024
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

    def __init__(self, directory: str | Path = ".cache/product_images") -> None:
        self.directory = Path(directory)
        self.lock = RLock()

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        # Python.org builds on macOS do not always inherit the system trust
        # store. truststore keeps HTTPS verification enabled while using it.
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def _image_results(self, query: str) -> list[dict[str, Any]]:
        context = self._ssl_context()
        headers = {
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) NAmazon/1.0",
        }
        search_url = "https://duckduckgo.com/?" + urlencode({"q": query})
        request = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(request, timeout=8, context=context) as response:
            page = response.read(512_000).decode("utf-8", "ignore")
        match = re.search(r"vqd=([^&\" ]+)", page)
        if not match:
            return []
        image_url = "https://duckduckgo.com/i.js?" + urlencode(
            {
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": match.group(1),
                "f": ",,,",
                "p": "1",
            }
        )
        image_headers = dict(headers)
        image_headers["Referer"] = "https://duckduckgo.com/"
        request = urllib.request.Request(image_url, headers=image_headers)
        with urllib.request.urlopen(request, timeout=8, context=context) as response:
            payload = json.loads(response.read(2_000_000))
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return [item for item in results[:6] if isinstance(item, dict)]

    def _paths(self, asin: str) -> tuple[Path, Path]:
        safe_asin = "".join(character for character in asin if character.isalnum())[:32]
        return self.directory / f"{safe_asin}.image", self.directory / f"{safe_asin}.json"

    def cached(self, asin: str) -> tuple[Path, str] | None:
        image_path, metadata_path = self._paths(asin)
        if not image_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mime_type = str(metadata.get("mime_type", ""))
        except (OSError, json.JSONDecodeError):
            return None
        return (image_path, mime_type) if mime_type in self.ALLOWED_TYPES else None

    def find(self, product: dict[str, Any]) -> tuple[Path, str] | None:
        asin = str(product.get("parent_asin", ""))
        if cached := self.cached(asin):
            return cached
        if os.getenv("NAMAZON_WEB_IMAGE_SEARCH", "true").lower() not in {"1", "true", "yes"}:
            return None
        with self.lock:
            if cached := self.cached(asin):
                return cached
            try:
                title = str(product.get("title", ""))[:220]
                store = str(product.get("store", ""))[:80]
                query = f'"{title}" {store} product'.strip()
                context = self._ssl_context()
                for result in self._image_results(query):
                    remote_url = str(result.get("image") or "")
                    if urlparse(remote_url).scheme not in {"http", "https"}:
                        continue
                    request = urllib.request.Request(
                        remote_url,
                        headers={"User-Agent": "Mozilla/5.0 NAmazon/1.0"},
                    )
                    try:
                        with urllib.request.urlopen(
                            request, timeout=8, context=context
                        ) as response:
                            mime_type = response.headers.get_content_type().lower()
                            length = int(response.headers.get("Content-Length", "0") or 0)
                            if mime_type not in self.ALLOWED_TYPES or length > self.MAX_BYTES:
                                continue
                            content = response.read(self.MAX_BYTES + 1)
                        if not content or len(content) > self.MAX_BYTES:
                            continue
                        image_path, metadata_path = self._paths(asin)
                        self.directory.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(content)
                        metadata_path.write_text(
                            json.dumps(
                                {"mime_type": mime_type, "source_url": remote_url},
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        return image_path, mime_type
                    except (OSError, ValueError):
                        continue
            except Exception:
                return None
        return None
