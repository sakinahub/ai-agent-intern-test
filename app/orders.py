from __future__ import annotations

import json
import re
from pathlib import Path

from .config import ORDERS_FILE


class OrderTool:
    """
    Safe order lookup tool.

    The raw orders.json contains private customer information
    and internal operational fields.

    This class deliberately exposes only customer-safe fields.
    """

    # Fields that are safe to expose to the customer.
    SAFE_FIELDS = {
        "order_id",
        "status",
        "status_updated_at",
        "shipped_at",
        "delivered_at",
        "carrier",
        "tracking_number",
        "estimated_delivery",
        "customer_safe_message",
        "membership_tier",
        "items",
    }

    def __init__(self):
        self.orders = self._load_orders()

    # =========================================================
    # LOAD ORDERS
    # =========================================================

    def _load_orders(self) -> dict:
        if not ORDERS_FILE.exists():
            raise FileNotFoundError(
                f"Orders file not found: {ORDERS_FILE}"
            )

        with open(
            ORDERS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        orders = data.get("orders", [])

        return {
            order["order_id"]: order
            for order in orders
            if "order_id" in order
        }

    # =========================================================
    # ORDER ID NORMALIZATION
    # =========================================================

    def _normalize_order_id(
        self,
        order_id: str,
    ) -> str | None:
        """
        Normalize common user variations such as:

        ord-1007
        ORD1007
        Order ORD-1007
        order 1007
        """

        if not order_id:
            return None

        cleaned = str(order_id).strip().upper()

        # Extract the numeric portion.
        match = re.search(
            r"(?:ORD[\s\-_]*)?(\d{4})",
            cleaned,
        )

        if not match:
            return None

        number = match.group(1)

        return f"ORD-{number}"

    # =========================================================
    # SAFE ORDER RESPONSE
    # =========================================================

    def _safe_order_view(
        self,
        order: dict,
    ) -> dict:
        """
        Return ONLY customer-safe order information.

        Never expose:
        - customer email
        - shipping address
        - risk score
        - warehouse notes
        - internal support tags
        """

        status = order.get("status")

        result = {
            "order_id": order.get("order_id"),
            "status": status,
            "status_updated_at": order.get(
                "status_updated_at"
            ),
            "carrier": order.get("carrier"),
            "tracking_number": order.get(
                "tracking_number"
            ),
            "shipped_at": order.get("shipped_at"),
            "delivered_at": order.get(
                "delivered_at"
            ),
            "membership_tier": order.get(
                "membership_tier"
            ),
            "items": order.get("items", []),
            "customer_safe_message": order.get(
                "customer_safe_message"
            ),
        }

        # -----------------------------------------------------
        # IMPORTANT:
        # ETA is NOT blindly copied.
        #
        # Cancelled orders can contain stale ETA values.
        # -----------------------------------------------------

        if status not in {
            "cancelled",
            "returned",
            "delivered",
        }:
            result["estimated_delivery"] = order.get(
                "estimated_delivery"
            )

        return result

    # =========================================================
    # PUBLIC LOOKUP
    # =========================================================

    def lookup(
        self,
        order_id: str,
    ) -> dict:
        """
        Look up an order safely.
        """

        normalized_id = self._normalize_order_id(
            order_id
        )

        if not normalized_id:
            return {
                "success": False,
                "error": "invalid_order_id",
                "message": (
                    "I couldn't recognize that order ID. "
                    "Please provide an order ID such as ORD-1007."
                ),
            }

        order = self.orders.get(
            normalized_id
        )

        if not order:
            return {
                "success": False,
                "error": "order_not_found",
                "order_id": normalized_id,
                "message": (
                    f"I couldn't find an order with ID "
                    f"{normalized_id}."
                ),
            }

        return {
            "success": True,
            "order": self._safe_order_view(order),
        }


# =============================================================
# LOCAL TEST
# =============================================================

if __name__ == "__main__":

    tool = OrderTool()

    test_ids = [
        "ORD-1007",
        "ord-1007",
        "Order 1007",
        "ORD1007",
        "ORD-9999",
        "hello",
        "ORD-1004",
    ]

    print("\n" + "=" * 70)
    print("ASTER & ROW — ORDER TOOL TEST")
    print("=" * 70)

    for order_id in test_ids:

        print(
            f"\nQUERY: {order_id}"
        )

        result = tool.lookup(order_id)

        print(json.dumps(
            result,
            indent=2,
        ))

    print("\n" + "=" * 70)