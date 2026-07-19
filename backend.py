"""ShopSense — Consumer multi-agent demo backend.

Stack: SQLite + FastAPI + OpenAI Agents SDK.

Endpoints
----------
GET  /api/products          -> list catalog (the page auto-loads this on open)
GET  /api/products?q=...  -> search
GET  /api/orders            -> list orders
POST /api/cart              -> add item to cart
GET  /api/cart              -> current cart
POST /api/chat              -> multi-agent assistant (products / orders / cart)

The agent endpoint uses the OpenAI Agents SDK. If OPENAI_API_KEY is not set it
degrades gracefully to a local keyword router so the demo still runs offline.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "shop.db"
FRONTEND = ROOT / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_if_empty()
    yield


app = FastAPI(title="ShopSense API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def seed_if_empty() -> None:
    """Create tables and seed a small catalog the first time the server starts."""
    conn = _get_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                rating REAL NOT NULL,
                description TEXT NOT NULL,
                stock INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer TEXT NOT NULL,
                items TEXT NOT NULL,
                total REAL NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cart (
                id INTEGER PRIMARY KEY,
                product_id INTEGER NOT NULL,
                qty INTEGER NOT NULL
            );
            """
        )
        count = conn.execute("SELECT COUNT(*) AS c FROM products").fetchone()["c"]
        if count == 0:
            products = [
                (1, "Aero Wireless Mouse", "accessories", 29.99, 4.6, "Ergonomic wireless mouse with silent clicks and 12-month battery.", 120),
                (2, "Lumen 4K Monitor", "displays", 349.0, 4.8, "27-inch 4K IPS monitor with USB-C charging and HDR.", 40),
                (3, "Nimbus Laptop 14", "laptops", 899.0, 4.7, "Lightweight 14-inch laptop, 16GB RAM, 512GB SSD, all-day battery.", 25),
                (4, "Quill Mechanical Keyboard", "accessories", 119.0, 4.5, "Hot-swappable mechanical keyboard with tactile switches.", 80),
                (5, "Pulse Noise-Cancel Headphones", "audio", 199.0, 4.9, "Over-ear ANC headphones with 30-hour playback.", 60),
                (6, "Terra Standing Desk", "furniture", 459.0, 4.4, "Electric height-adjustable standing desk, 60-inch top.", 15),
                (7, "Cobalt Webcam 1080p", "accessories", 49.99, 4.2, "Plug-and-play 1080p webcam with auto light correction.", 200),
                (8, "Echo Bluetooth Speaker", "audio", 59.99, 4.3, "Compact waterproof speaker with 14-hour battery.", 150),
                (9, "Vertex Gaming Chair", "furniture", 279.0, 4.1, "Ergonomic racing-style chair with lumbar support.", 30),
                (10, "Slate Tablet 11", "laptops", 549.0, 4.6, "11-inch tablet with stylus support and laptop dock.", 45),
            ]
            conn.executemany(
                "INSERT INTO products (id,name,category,price,rating,description,stock) "
                "VALUES (?,?,?,?,?,?,?)",
                products,
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def search_products(query: str | None) -> list[dict]:
    conn = _get_conn()
    try:
        if query:
            like = f"%{query.strip()}%"
            rows = conn.execute(
                "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR description LIKE ? "
                "ORDER BY rating DESC",
                (like, like, like),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM products ORDER BY rating DESC").fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    if not FRONTEND.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    return FileResponse(str(FRONTEND))


# ---------------------------------------------------------------------------
# Catalog / orders / cart API
# ---------------------------------------------------------------------------
@app.get("/api/products")
def get_products(q: str | None = None) -> dict:
    try:
        items = search_products(q)
    except sqlite3.Error as exc:  # surface a real 500 only on genuine DB failure
        raise HTTPException(status_code=500, detail=f"database error: {exc}") from exc
    return {"products": items, "count": len(items)}


@app.get("/api/orders")
def get_orders() -> dict:
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM orders ORDER BY id DESC").fetchall()
        orders = [_row_to_dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"database error: {exc}") from exc
    finally:
        conn.close()
    return {"orders": orders}


@app.get("/api/cart")
def get_cart() -> dict:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT c.product_id, c.qty, p.name, p.price FROM cart c "
            "JOIN products p ON p.id = c.product_id"
        ).fetchall()
        items = [_row_to_dict(r) for r in rows]
        total = round(sum(i["price"] * i["qty"] for i in items), 2)
    finally:
        conn.close()
    return {"items": items, "total": total}


class CartItem(BaseModel):
    product_id: int
    qty: int = 1


@app.post("/api/cart")
def add_to_cart(item: CartItem) -> dict:
    conn = _get_conn()
    try:
        prod = conn.execute(
            "SELECT id, stock FROM products WHERE id = ?", (item.product_id,)
        ).fetchone()
        if prod is None:
            raise HTTPException(status_code=404, detail="product not found")
        if item.qty <= 0:
            raise HTTPException(status_code=400, detail="qty must be positive")
        if item.qty > prod["stock"]:
            raise HTTPException(status_code=400, detail="not enough stock")
        existing = conn.execute(
            "SELECT qty FROM cart WHERE product_id = ?", (item.product_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE cart SET qty = qty + ? WHERE product_id = ?",
                (item.qty, item.product_id),
            )
        else:
            conn.execute(
                "INSERT INTO cart (product_id, qty) VALUES (?, ?)",
                (item.product_id, item.qty),
            )
        conn.commit()
    finally:
        conn.close()
    return get_cart()


# ---------------------------------------------------------------------------
# Multi-agent assistant (OpenAI Agents SDK)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


_AGENT_READY = False
_AGENT_ERROR = ""


def _build_agent():
    """Build the shopping assistant using the OpenAI Agents SDK.

    Returns the Runner/agent pair, or (None, None) when the SDK cannot run
    (e.g. no API key). The chat route falls back to a local router in that case.
    """
    from agents import Agent, Runner, function_tool

    @function_tool
    def list_products(category: str | None = None) -> str:
        """List products, optionally filtered by category."""
        rows = search_products(category)
        return json.dumps(rows, default=str)

    @function_tool
    def find_under_budget(budget: float) -> str:
        """Return products priced at or below the given budget."""
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM products WHERE price <= ? ORDER BY price ASC", (budget,)
            ).fetchall()
        finally:
            conn.close()
        return json.dumps([_row_to_dict(r) for r in rows], default=str)

    @function_tool
    def get_orders() -> str:
        """Return the customer's recent orders."""
        conn = _get_conn()
        try:
            rows = conn.execute("SELECT * FROM orders ORDER BY id DESC").fetchall()
        finally:
            conn.close()
        return json.dumps([_row_to_dict(r) for r in rows], default=str)

    agent = Agent(
        name="ShopSense Assistant",
        instructions=(
            "You are the ShopSense shopping assistant. Help the customer find products, "
            "recommend items under a budget, check orders, and manage the cart. "
            "Use the available tools to answer accurately. Be concise."
        ),
        tools=[list_products, find_under_budget, get_orders],
    )
    return agent, Runner


def _local_router(message: str) -> str:
    """Offline fallback used when the Agents SDK is not configured."""
    msg = message.lower()
    if "order" in msg:
        conn = _get_conn()
        try:
            rows = conn.execute("SELECT * FROM orders ORDER BY id DESC").fetchall()
        finally:
            conn.close()
        return "Recent orders:\n" + (
            "\n".join(f"#{o['id']} — {o['status']} (${o['total']})" for o in rows)
            or "No orders yet."
        )
    if "budget" in msg or "under" in msg:
        import re

        m = re.search(r"\$?(\d+(?:\.\d+)?)", msg)
        budget = float(m.group(1)) if m else 100.0
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT name, price FROM products WHERE price <= ? ORDER BY price ASC",
                (budget,),
            ).fetchall()
        finally:
            conn.close()
        return f"Under ${budget:.2f}:\n" + (
            "\n".join(f"- {r['name']} (${r['price']:.2f})" for r in rows) or "Nothing in that range."
        )
    rows = search_products(message if len(message) > 2 else None)
    return "Here are some products:\n" + "\n".join(
        f"- {r['name']} (${r['price']:.2f}, {r['category']})" for r in rows[:5]
    )


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    global _AGENT_READY, _AGENT_ERROR
    if not _AGENT_READY:
        try:
            agent, _ = _build_agent()
            _AGENT_READY = agent is not None
        except Exception as exc:  # noqa: BLE001
            _AGENT_ERROR = str(exc)
            _AGENT_READY = False

    if _AGENT_READY and os.environ.get("OPENAI_API_KEY"):
        try:
            from agents import Runner

            agent, _ = _build_agent()
            result = await Runner.run(agent, req.message)
            return {"reply": result.final_output, "mode": "agent"}
        except Exception as exc:  # noqa: BLE001
            return {"reply": _local_router(req.message), "mode": "fallback", "note": str(exc)}

    return {"reply": _local_router(req.message), "mode": "fallback"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
