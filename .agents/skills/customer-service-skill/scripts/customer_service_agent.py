"""
Offline customer service QA agent.

This script follows the local customer-service skill instructions:
identify intent, gather relevant details, provide a clear response,
ask for missing information, and escalate issues that need a human agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Intent(str, Enum):
    ORDER_TRACKING = "order_tracking"
    RETURN = "return"
    REFUND = "refund"
    ACCOUNT = "account"
    PRODUCT_INFO = "product_info"
    GENERAL_SUPPORT = "general_support"
    ESCALATE = "escalate"


@dataclass
class ConversationState:
    order_id: str | None = None
    product_name: str | None = None
    account_identifier: str | None = None
    refund_reason: str | None = None
    return_reason: str | None = None
    last_intent: Intent | None = None
    history: list[tuple[str, str]] = field(default_factory=list)


class CustomerServiceAgent:
    """Simple deterministic customer-service assistant."""

    EXIT_COMMANDS = {"exit", "quit", "done"}

    SENSITIVE_PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "Social Security numbers"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "full payment card numbers"),
        (re.compile(r"\bpassword\s*(?:is|:)\s*\S+", re.IGNORECASE), "passwords"),
        (re.compile(r"\bpin\s*(?:is|:)\s*\d+", re.IGNORECASE), "PINs"),
    ]

    ESCALATION_KEYWORDS = {
        "lawsuit",
        "lawyer",
        "attorney",
        "sue",
        "legal",
        "fraud",
        "stolen",
        "chargeback",
        "harassment",
        "discrimination",
        "threat",
        "police",
        "medical",
        "injury",
        "unsafe",
    }

    ORDER_KEYWORDS = {
        "order",
        "tracking",
        "track",
        "shipment",
        "shipping",
        "delivery",
        "delivered",
        "package",
        "arrived",
    }

    RETURN_KEYWORDS = {"return", "exchange", "replacement", "send back", "defective"}
    REFUND_KEYWORDS = {"refund", "money back", "charged", "billing", "cancel payment"}
    ACCOUNT_KEYWORDS = {
        "account",
        "login",
        "log in",
        "password",
        "username",
        "email",
        "sign in",
        "reset",
        "locked",
    }
    PRODUCT_KEYWORDS = {
        "product",
        "item",
        "color",
        "size",
        "available",
        "availability",
        "stock",
        "warranty",
        "feature",
        "material",
    }

    def __init__(self) -> None:
        self.state = ConversationState()

    def respond(self, message: str) -> str:
        """Return a polite customer-service response for one customer message."""
        cleaned = message.strip()
        if not cleaned:
            return "I can help with that. Please tell me what you need help with."

        sensitive_warning = self._find_sensitive_data(cleaned)
        if sensitive_warning:
            response = (
                f"For your security, please do not share {sensitive_warning} here. "
                "I can still help using non-sensitive details like an order ID, product name, "
                "or the email/username on the account."
            )
            self._remember(cleaned, response)
            return response

        self._update_known_details(cleaned)
        intent = self._detect_intent(cleaned)
        self.state.last_intent = intent

        if intent == Intent.ESCALATE:
            response = self._escalation_response()
        elif intent == Intent.ORDER_TRACKING:
            response = self._order_tracking_response()
        elif intent == Intent.RETURN:
            response = self._return_response()
        elif intent == Intent.REFUND:
            response = self._refund_response()
        elif intent == Intent.ACCOUNT:
            response = self._account_response()
        elif intent == Intent.PRODUCT_INFO:
            response = self._product_info_response()
        else:
            response = self._general_support_response()

        self._remember(cleaned, response)
        return response

    def should_exit(self, message: str) -> bool:
        return message.strip().lower() in self.EXIT_COMMANDS

    def _detect_intent(self, message: str) -> Intent:
        lowered = message.lower()

        if self._contains_any(lowered, self.ESCALATION_KEYWORDS):
            return Intent.ESCALATE
        if self._contains_any(lowered, self.REFUND_KEYWORDS):
            return Intent.REFUND
        if self._contains_any(lowered, self.RETURN_KEYWORDS):
            return Intent.RETURN
        if self._contains_any(lowered, self.ORDER_KEYWORDS):
            return Intent.ORDER_TRACKING
        if self._contains_any(lowered, self.ACCOUNT_KEYWORDS):
            return Intent.ACCOUNT
        if self._contains_any(lowered, self.PRODUCT_KEYWORDS):
            return Intent.PRODUCT_INFO
        return Intent.GENERAL_SUPPORT

    def _update_known_details(self, message: str) -> None:
        order_id = self._extract_order_id(message)
        if order_id:
            self.state.order_id = order_id

        account_identifier = self._extract_account_identifier(message)
        if account_identifier:
            self.state.account_identifier = account_identifier

        product_name = self._extract_product_name(message)
        if product_name:
            self.state.product_name = product_name

        lowered = message.lower()
        if self._contains_any(lowered, self.REFUND_KEYWORDS):
            self.state.refund_reason = self._extract_reason(message)
        if self._contains_any(lowered, self.RETURN_KEYWORDS):
            self.state.return_reason = self._extract_reason(message)

    def _order_tracking_response(self) -> str:
        if not self.state.order_id:
            return (
                "I can help check the order status. Please provide the order ID, "
                "but do not share payment information or other sensitive details."
            )

        return (
            f"Thanks. For order {self.state.order_id}, please check the tracking link in your "
            "order confirmation email first. If the carrier page has not updated in 24 hours, "
            "contact support with this order ID so a team member can review the shipment record."
        )

    def _return_response(self) -> str:
        missing = []
        if not self.state.order_id:
            missing.append("order ID")
        if not self.state.return_reason:
            missing.append("reason for the return")

        if missing:
            return (
                "I can help start a return. Please provide the "
                f"{self._join_items(missing)}. Do not include payment details."
            )

        return (
            f"Thanks. For order {self.state.order_id}, the next step is to review the return "
            "eligibility and package the item in its original condition when possible. A support "
            "team member can confirm the return label and any policy limits before you ship it."
        )

    def _refund_response(self) -> str:
        missing = []
        if not self.state.order_id:
            missing.append("order ID")
        if not self.state.refund_reason:
            missing.append("reason for the refund")

        if missing:
            return (
                "I can help with a refund request. Please provide the "
                f"{self._join_items(missing)}. For safety, do not share full card numbers."
            )

        return (
            f"Thanks. For order {self.state.order_id}, I can note the refund reason and recommend "
            "that support review the order against the refund policy. If approved, refunds are "
            "usually sent back to the original payment method by the billing team."
        )

    def _account_response(self) -> str:
        if not self.state.account_identifier:
            return (
                "I can help with account access. Please provide the email address or username "
                "on the account, but never share your password or verification code."
            )

        return (
            f"Thanks. For account {self.state.account_identifier}, use the password reset option "
            "on the sign-in page and check your email for the reset link. If the account is locked "
            "or the reset email does not arrive, support should verify ownership before making changes."
        )

    def _product_info_response(self) -> str:
        if not self.state.product_name:
            return "I can help with product information. Which product or item are you asking about?"

        return (
            f"For {self.state.product_name}, please check the product page for current options, "
            "availability, size/color variants, warranty details, and shipping estimates. If you "
            "need a specific detail that is not listed, tell me what you want to know."
        )

    def _general_support_response(self) -> str:
        return (
            "I can help with orders, returns, refunds, account issues, and product questions. "
            "Please share the issue type and any relevant non-sensitive details, such as an order ID "
            "or product name."
        )

    def _escalation_response(self) -> str:
        return (
            "This issue should be escalated to a human support agent. Please avoid sharing sensitive "
            "personal, legal, payment, or medical details here. A human agent can review the situation "
            "and follow the correct company process."
        )

    def _find_sensitive_data(self, message: str) -> str | None:
        for pattern, label in self.SENSITIVE_PATTERNS:
            if pattern.search(message):
                return label
        return None

    def _extract_order_id(self, message: str) -> str | None:
        patterns = [
            r"\border\s*(?:id|number|#|no\.?)?\s*(?:is|:|#|-)?\s*([A-Z0-9-]{4,})\b",
            r"\b(?:id|#)\s*[:#-]?\s*([A-Z0-9-]{4,})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _extract_account_identifier(self, message: str) -> str | None:
        email = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", message, re.IGNORECASE)
        if email:
            return email.group(0).lower()

        username = re.search(r"\busername\s*(?:is|:)?\s*([A-Z0-9._-]{3,})\b", message, re.IGNORECASE)
        if username:
            return username.group(1)

        return None

    def _extract_product_name(self, message: str) -> str | None:
        patterns = [
            r"\b(?:product|item)\s+(?:name\s+is|called|named|is|:)\s+([A-Z0-9][A-Z0-9 ._-]{2,})",
            r"\babout\s+(?:the\s+)?([A-Z0-9][A-Z0-9 ._-]{2,})",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return self._clean_phrase(match.group(1))
        return None

    def _extract_reason(self, message: str) -> str | None:
        reason_patterns = [
            r"\bbecause\s+(.+)",
            r"\breason\s*(?:is|:)\s*(.+)",
            r"\bit\s+(?:is|was)\s+(.+)",
        ]
        for pattern in reason_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return self._clean_phrase(match.group(1))
        return None

    def _remember(self, message: str, response: str) -> None:
        self.state.history.append((message, response))

    @staticmethod
    def _contains_any(text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _clean_phrase(value: str) -> str:
        return value.strip(" .,!?:;")

    @staticmethod
    def _join_items(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def run_interactive_cli() -> None:
    agent = CustomerServiceAgent()
    print("Customer Service QA Agent")
    print("Type your customer-service question. Type 'exit', 'quit', or 'done' to end.")
    print()

    while True:
        try:
            message = input("Customer: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Goodbye.")
            break

        if agent.should_exit(message):
            print("Agent: Goodbye. Thanks for using the customer service QA agent.")
            break

        print(f"Agent: {agent.respond(message)}")
        print()


if __name__ == "__main__":
    run_interactive_cli()
