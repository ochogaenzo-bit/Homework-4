"""
Customer service QA agent.

This version is built as a hybrid agent:
- If OPENAI_API_KEY is set, it uses the OpenAI Responses API for natural,
  context-aware replies.
- If no API key is set, it falls back to a local case manager so the script can
  still run in class or inside Codex without internet/API access.

The local case manager owns facts and simulated actions. The model owns wording.
That keeps the conversation flexible without pretending the agent has touched a
real payment, order, or account system.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Intent(str, Enum):
    REFUND = "refund"
    RETURN = "return"
    ORDER_TRACKING = "order_tracking"
    ACCOUNT = "account"
    PRODUCT_INFO = "product_info"
    GENERAL_SUPPORT = "general_support"
    ESCALATE = "escalate"


class CaseStatus(str, Enum):
    COLLECTING = "collecting"
    READY = "ready"
    SUBMITTED = "submitted"
    ESCALATED = "escalated"
    AFTERCARE = "aftercare"
    CLOSED = "closed"


@dataclass
class LineItem:
    name: str
    amount: float | None = None


@dataclass
class SupportCase:
    case_type: Intent | None = None
    status: CaseStatus = CaseStatus.COLLECTING
    order_id: str | None = None
    account_identifier: str | None = None
    product_name: str | None = None
    reason: str | None = None
    issue_description: str | None = None
    items: list[LineItem] = field(default_factory=list)
    case_number: str | None = None
    completed_cases: list[str] = field(default_factory=list)


@dataclass
class TurnPlan:
    intent: Intent
    missing: list[str] = field(default_factory=list)
    action: str | None = None
    action_result: str | None = None
    ask_aftercare: bool = False
    close_conversation: bool = False


class CustomerServiceAgent:
    """Customer-service agent with model-backed wording and local guardrails."""

    EXIT_COMMANDS = {"exit", "quit", "done"}

    REFUND_KEYWORDS = {"refund", "money back", "charged", "billing", "overcharged", "cancel payment"}
    RETURN_KEYWORDS = {"return", "exchange", "replacement", "send back", "defective"}
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
        "compatible",
    }
    TRACKING_KEYWORDS = {
        "tracking",
        "track",
        "shipment",
        "shipping",
        "delivery",
        "delivered",
        "package",
        "arrived",
        "where is my order",
        "where's my order",
    }
    ORDER_DETAIL_KEYWORDS = {"order", "order id", "order number", "id"}
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
    SENSITIVE_PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "Social Security numbers"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "full card numbers"),
        (re.compile(r"\bpassword\s*(?:is|:)\s*\S+", re.IGNORECASE), "passwords"),
        (re.compile(r"\bpin\s*(?:is|:)\s*\d+", re.IGNORECASE), "PINs"),
        (re.compile(r"\bcode\s*(?:is|:)\s*\d{4,8}\b", re.IGNORECASE), "verification codes"),
    ]
    CONFIRM_WORDS = {"submit", "send", "confirm", "approved", "yes", "yep", "yeah", "do it", "start it"}
    CLOSE_WORDS = {
        "no",
        "no thanks",
        "no thank you",
        "nothing else",
        "that is all",
        "that's all",
        "all good",
        "thanks",
        "thank you",
        "ty",
    }

    def __init__(self, use_ai: bool | None = None) -> None:
        self.case = SupportCase()
        self.transcript: list[dict[str, str]] = []
        self.case_sequence = 1000
        self.closed = False

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.use_ai = bool(self.api_key) if use_ai is None else bool(use_ai and self.api_key)
        self.last_ai_error: str | None = None

    def respond(self, message: str) -> str:
        cleaned = message.strip()
        if not cleaned:
            return "Tell me what happened and I will help from there."

        if self.closed:
            self._reset_for_new_case()

        sensitive_warning = self._find_sensitive_data(cleaned)
        if sensitive_warning:
            response = (
                f"For your security, please do not share {sensitive_warning}. "
                "I can help using safe details like an order ID, item name, amount, "
                "product name, or account email/username."
            )
            self._remember(cleaned, response)
            return response

        if self.case.status == CaseStatus.AFTERCARE:
            aftercare = self._handle_aftercare(cleaned)
            if aftercare:
                self._remember(cleaned, aftercare)
                return aftercare

        intent = self._resolve_intent(cleaned)
        self._start_case_if_needed(intent)
        self._update_case_from_message(cleaned, intent)
        turn = self._plan_turn(cleaned, intent)

        response = self._ai_response(cleaned, turn) if self.use_ai else None
        if not response:
            response = self._local_response(turn)

        self._remember(cleaned, response)
        return response

    def should_exit(self, message: str) -> bool:
        return self._normalize(message) in self.EXIT_COMMANDS

    def conversation_finished(self) -> bool:
        return self.closed

    def runtime_mode(self) -> str:
        if self.use_ai:
            return f"AI mode using {self.model}"
        return "offline fallback mode - set OPENAI_API_KEY for model-backed replies"

    def _resolve_intent(self, message: str) -> Intent:
        detected = self._detect_intent(message)

        if not self.case.case_type or self.case.status in {CaseStatus.SUBMITTED, CaseStatus.ESCALATED, CaseStatus.CLOSED}:
            return detected

        if self._is_confirmation(message):
            return self.case.case_type

        if self._is_detail_followup(message):
            return self.case.case_type

        if detected == self.case.case_type or detected == Intent.GENERAL_SUPPORT:
            return self.case.case_type

        if detected == Intent.ORDER_TRACKING and self.case.case_type in {Intent.REFUND, Intent.RETURN}:
            if not self._contains_any(message.lower(), self.TRACKING_KEYWORDS):
                return self.case.case_type

        return detected

    def _detect_intent(self, message: str) -> Intent:
        lowered = message.lower()
        if self._contains_any(lowered, self.ESCALATION_KEYWORDS):
            return Intent.ESCALATE
        if self._contains_any(lowered, self.REFUND_KEYWORDS):
            return Intent.REFUND
        if self._contains_any(lowered, self.RETURN_KEYWORDS):
            return Intent.RETURN
        if self._contains_any(lowered, self.TRACKING_KEYWORDS):
            return Intent.ORDER_TRACKING
        if self._contains_any(lowered, self.ACCOUNT_KEYWORDS):
            return Intent.ACCOUNT
        if self._contains_any(lowered, self.PRODUCT_KEYWORDS):
            return Intent.PRODUCT_INFO
        if self._extract_order_id(message):
            return Intent.ORDER_TRACKING
        return Intent.GENERAL_SUPPORT

    def _start_case_if_needed(self, intent: Intent) -> None:
        if intent == Intent.GENERAL_SUPPORT:
            return
        if self.case.case_type != intent or self.case.status in {CaseStatus.SUBMITTED, CaseStatus.ESCALATED, CaseStatus.CLOSED}:
            completed = self.case.completed_cases
            self.case = SupportCase(case_type=intent, completed_cases=completed)

    def _update_case_from_message(self, message: str, intent: Intent) -> None:
        if intent == Intent.GENERAL_SUPPORT:
            return

        self.case.case_type = intent

        order_id = self._extract_order_id(message)
        if order_id:
            self.case.order_id = order_id

        account = self._extract_account_identifier(message)
        if account:
            self.case.account_identifier = account

        product = self._extract_product_name(message)
        if product:
            self.case.product_name = product

        items = self._extract_items(message)
        if items:
            self._merge_items(items)

        reason = self._extract_reason(message)
        if reason:
            self.case.reason = reason
        elif len(message.split()) > 5 and not self.case.issue_description:
            self.case.issue_description = self._clean_phrase(message)

    def _plan_turn(self, message: str, intent: Intent) -> TurnPlan:
        if intent == Intent.ESCALATE:
            case_number = self._ensure_case_number("ESC")
            self.case.status = CaseStatus.ESCALATED
            self._complete_case(case_number)
            return TurnPlan(
                intent=intent,
                action="escalate_to_human",
                action_result=f"Escalation case {case_number} recorded for human review.",
                ask_aftercare=True,
            )

        if intent == Intent.REFUND:
            missing = self._refund_missing_fields()
            if self._is_confirmation(message) and not missing:
                case_number = self._ensure_case_number("REF")
                self.case.status = CaseStatus.SUBMITTED
                self._complete_case(case_number)
                return TurnPlan(
                    intent=intent,
                    action="submit_refund_request",
                    action_result=(
                        f"Refund case {case_number} recorded for review. "
                        f"Requested total: ${self._refund_total():.2f}."
                    ),
                    ask_aftercare=True,
                )
            self.case.status = CaseStatus.COLLECTING if missing else CaseStatus.READY
            return TurnPlan(intent=intent, missing=missing, action="prepare_refund_request" if not missing else None)

        if intent == Intent.RETURN:
            missing = self._return_missing_fields()
            if self._is_confirmation(message) and not missing:
                case_number = self._ensure_case_number("RET")
                self.case.status = CaseStatus.SUBMITTED
                self._complete_case(case_number)
                return TurnPlan(
                    intent=intent,
                    action="submit_return_request",
                    action_result=f"Return case {case_number} recorded for review.",
                    ask_aftercare=True,
                )
            self.case.status = CaseStatus.COLLECTING if missing else CaseStatus.READY
            return TurnPlan(intent=intent, missing=missing, action="prepare_return_request" if not missing else None)

        if intent == Intent.ORDER_TRACKING:
            if not self.case.order_id:
                self.case.status = CaseStatus.COLLECTING
                return TurnPlan(intent=intent, missing=["order ID"])
            case_number = self._ensure_case_number("TRK")
            self.case.status = CaseStatus.SUBMITTED
            self._complete_case(case_number)
            return TurnPlan(
                intent=intent,
                action="open_tracking_investigation",
                action_result=f"Tracking case {case_number} recorded for order {self.case.order_id}.",
                ask_aftercare=True,
            )

        if intent == Intent.ACCOUNT:
            if not self.case.account_identifier:
                self.case.status = CaseStatus.COLLECTING
                return TurnPlan(intent=intent, missing=["account email or username"])
            case_number = self._ensure_case_number("ACC")
            self.case.status = CaseStatus.SUBMITTED
            self._complete_case(case_number)
            return TurnPlan(
                intent=intent,
                action="open_account_support_case",
                action_result=f"Account support case {case_number} recorded.",
                ask_aftercare=True,
            )

        if intent == Intent.PRODUCT_INFO:
            if not self.case.product_name:
                self.case.status = CaseStatus.COLLECTING
                return TurnPlan(intent=intent, missing=["product name"])
            case_number = self._ensure_case_number("PRD")
            self.case.status = CaseStatus.SUBMITTED
            self._complete_case(case_number)
            return TurnPlan(
                intent=intent,
                action="open_product_info_case",
                action_result=f"Product information case {case_number} recorded.",
                ask_aftercare=True,
            )

        return TurnPlan(intent=Intent.GENERAL_SUPPORT, missing=["support issue"])

    def _ai_response(self, user_message: str, turn: TurnPlan) -> str | None:
        prompt = self._build_model_prompt(user_message, turn)
        payload = {
            "model": self.model,
            "instructions": self._system_instructions(),
            "input": prompt,
            "max_output_tokens": 350,
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self.last_ai_error = str(exc)
            return None

        return self._extract_response_text(data)

    def _build_model_prompt(self, user_message: str, turn: TurnPlan) -> str:
        recent_turns = self.transcript[-8:]
        state = {
            "current_case": self._case_state_dict(),
            "turn_plan": {
                "intent": turn.intent.value,
                "missing": turn.missing,
                "action": turn.action,
                "action_result": turn.action_result,
                "ask_aftercare": turn.ask_aftercare,
                "close_conversation": turn.close_conversation,
            },
            "recent_transcript": recent_turns,
            "latest_customer_message": user_message,
        }
        return json.dumps(state, indent=2)

    def _system_instructions(self) -> str:
        return (
            "You are a realistic customer-service QA agent. Respond naturally, not like a fixed script. "
            "Use the provided JSON state as truth. Do not invent order data, payment success, account "
            "changes, tracking details, policies, or real system access. The local case manager may give "
            "you an action_result; when it does, say that the request was recorded or submitted for review "
            "in this QA session. For refunds, never call the case a tracking request. For returns, never "
            "call the case a refund unless the state says refund. If fields are missing, ask for only the "
            "missing fields in one friendly question and include a compact example only when helpful. "
            "If the case is ready but not submitted, summarize the facts and ask for confirmation. "
            "If an action was completed, close the loop and ask whether there is anything else. "
            "If the customer says they are done, close politely. Never request passwords, full card "
            "numbers, SSNs, PINs, or verification codes. Escalate legal, fraud, safety, harassment, "
            "or sensitive complaints to a human."
        )

    def _local_response(self, turn: TurnPlan) -> str:
        if turn.close_conversation:
            return "You are all set. Thanks for reaching out, and have a good day."

        if turn.action_result:
            return f"{turn.action_result}\n{self._case_summary()}\nIs there anything else I can help with?"

        if turn.intent == Intent.GENERAL_SUPPORT:
            return (
                "I can help with orders, refunds, returns, account access, or product questions. "
                "Tell me what is going on, and include safe details like an order ID, item name, "
                "amount, product name, or account email/username."
            )

        if turn.missing:
            return self._missing_fields_response(turn)

        if turn.intent == Intent.REFUND:
            return (
                "I have the refund details ready for review:\n"
                f"{self._case_summary()}\n"
                f"Requested refund total: ${self._refund_total():.2f}\n"
                "Reply 'submit' if you want me to record the refund request in this QA session."
            )

        if turn.intent == Intent.RETURN:
            return (
                "I have the return details ready for review:\n"
                f"{self._case_summary()}\n"
                "Reply 'submit' if you want me to record the return request in this QA session."
            )

        return "I have the details I need. I recorded the case for review. Anything else I can help with?"

    def _missing_fields_response(self, turn: TurnPlan) -> str:
        missing = self._join_items(turn.missing)
        if turn.intent == Intent.REFUND:
            return (
                f"I can work on the refund, but I still need the {missing}. "
                "For example: Order 12345, refund headphones $59.99 because they arrived broken."
            )
        if turn.intent == Intent.RETURN:
            return (
                f"I can start the return, but I still need the {missing}. "
                "For example: Order 12345, return blue jacket because it was the wrong size."
            )
        if turn.intent == Intent.ORDER_TRACKING:
            return "I can check that order issue. What is the order ID?"
        if turn.intent == Intent.ACCOUNT:
            return "I can help with the account. What email address or username is on it? Do not send your password."
        if turn.intent == Intent.PRODUCT_INFO:
            return "Which product are you asking about?"
        return f"What is the {missing}?"

    def _handle_aftercare(self, message: str) -> str | None:
        lowered = self._normalize(message)
        if lowered in self.CLOSE_WORDS or self._contains_any(lowered, self.CLOSE_WORDS):
            self.case.status = CaseStatus.CLOSED
            self.closed = True
            return "You are all set. Thanks for reaching out, and have a good day."

        intent = self._detect_intent(message)
        if intent != Intent.GENERAL_SUPPORT or len(message.split()) > 3:
            self._reset_for_new_case()
            return None

        if self._contains_any(lowered, {"yes", "yeah", "yep", "sure", "another", "one more"}):
            self._reset_for_new_case()
            return "Of course. Tell me what else is going on and I will help with the next issue."

        return "Do you have another support issue, or are you all set?"

    def _refund_missing_fields(self) -> list[str]:
        missing = []
        if not self.case.order_id:
            missing.append("order ID")
        if not self.case.items:
            missing.append("item names and refund amounts")
        elif any(item.amount is None for item in self.case.items):
            missing.append("refund amount for each item")
        if not self.case.reason:
            missing.append("reason for the refund")
        return missing

    def _return_missing_fields(self) -> list[str]:
        missing = []
        if not self.case.order_id:
            missing.append("order ID")
        if not self.case.items:
            missing.append("item or items to return")
        if not self.case.reason:
            missing.append("reason for the return")
        return missing

    def _case_state_dict(self) -> dict[str, object]:
        return {
            "case_type": self.case.case_type.value if self.case.case_type else None,
            "status": self.case.status.value,
            "case_number": self.case.case_number,
            "order_id": self.case.order_id,
            "items": [{"name": item.name, "amount": item.amount} for item in self.case.items],
            "refund_total": self._refund_total(),
            "reason": self.case.reason,
            "account_identifier": self.case.account_identifier,
            "product_name": self.case.product_name,
            "issue_description": self.case.issue_description,
            "completed_cases": self.case.completed_cases,
        }

    def _case_summary(self) -> str:
        parts = []
        if self.case.case_number:
            parts.append(f"Case: {self.case.case_number}")
        if self.case.case_type:
            parts.append(f"Type: {self.case.case_type.value.replace('_', ' ')}")
        if self.case.order_id:
            parts.append(f"Order ID: {self.case.order_id}")
        if self.case.items:
            parts.append(f"Items: {self._format_items()}")
        if self.case.reason:
            parts.append(f"Reason: {self.case.reason}")
        if self.case.account_identifier:
            parts.append(f"Account: {self.case.account_identifier}")
        if self.case.product_name:
            parts.append(f"Product: {self.case.product_name}")
        return "\n".join(parts) if parts else "No case details captured yet."

    def _extract_order_id(self, message: str) -> str | None:
        patterns = [
            r"\border\s*(?:id|number|#|no\.?)?\s*(?:is|:|#|-)?\s*([A-Z0-9-]{4,})\b",
            r"\b(?:order\s*)?#\s*([A-Z0-9-]{4,})\b",
            r"\bid\s*(?:is|:)?\s*([A-Z0-9-]{4,})\b",
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

    def _extract_items(self, message: str) -> list[LineItem]:
        before_reason = re.split(r"\bbecause\b|\breason\s*(?:is|:)", message, maxsplit=1, flags=re.IGNORECASE)[0]
        items = self._extract_money_items(before_reason)
        if items:
            return items

        plain_match = re.search(
            r"\b(?:refund|return|exchange|replace)\s+(?:the\s+)?([A-Z0-9 ,._-]{3,})",
            before_reason,
            re.IGNORECASE,
        )
        if not plain_match:
            return []

        plain_items = []
        for part in re.split(r",|\band\b", plain_match.group(1), flags=re.IGNORECASE):
            name = self._clean_item_name(part)
            if name and not self._looks_like_reason(name):
                plain_items.append(LineItem(name=name))
        return plain_items

    def _extract_money_items(self, text: str) -> list[LineItem]:
        items = []
        for match in re.finditer(r"\$([0-9]+(?:\.[0-9]{1,2})?)", text):
            amount = float(match.group(1))
            before_amount = text[: match.start()]
            name_chunk = re.split(
                r",|\band\b|\brefund\b|\breturn\b|\bexchange\b|\breplace\b",
                before_amount,
                flags=re.IGNORECASE,
            )[-1]
            name = self._clean_item_name(name_chunk)
            if name:
                items.append(LineItem(name=name, amount=amount))
        return items

    def _extract_reason(self, message: str) -> str | None:
        patterns = [
            r"\bbecause\s+(.+)",
            r"\breason\s*(?:is|:)\s*(.+)",
            r"\b(?:it|they)\s+(?:is|are|was|were)\s+(.+)",
            r"\barrived\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return self._clean_phrase(match.group(1))
        return None

    def _merge_items(self, new_items: list[LineItem]) -> None:
        existing = {item.name.lower(): item for item in self.case.items}
        for item in new_items:
            current = existing.get(item.name.lower())
            if current:
                if item.amount is not None:
                    current.amount = item.amount
            else:
                self.case.items.append(item)

    def _complete_case(self, case_number: str) -> None:
        if case_number not in self.case.completed_cases:
            self.case.completed_cases.append(case_number)
        self.case.status = CaseStatus.AFTERCARE

    def _ensure_case_number(self, prefix: str) -> str:
        if not self.case.case_number or not self.case.case_number.startswith(prefix):
            self.case_sequence += 1
            self.case.case_number = f"{prefix}-{self.case_sequence}"
        return self.case.case_number

    def _reset_for_new_case(self) -> None:
        completed = self.case.completed_cases
        self.case = SupportCase(completed_cases=completed)
        self.closed = False

    def _find_sensitive_data(self, message: str) -> str | None:
        for pattern, label in self.SENSITIVE_PATTERNS:
            if pattern.search(message):
                return label
        return None

    def _is_confirmation(self, message: str) -> bool:
        lowered = self._normalize(message)
        return lowered in self.CONFIRM_WORDS or self._contains_any(lowered, self.CONFIRM_WORDS)

    def _is_detail_followup(self, message: str) -> bool:
        lowered = message.lower()
        return bool(
            self._extract_order_id(message)
            or self._extract_items(message)
            or self._extract_reason(message)
            or self._extract_account_identifier(message)
            or re.search(r"\$[0-9]+(?:\.[0-9]{1,2})?", message)
            or self._contains_any(lowered, self.ORDER_DETAIL_KEYWORDS)
        )

    def _refund_total(self) -> float:
        return sum(item.amount or 0.0 for item in self.case.items)

    def _format_items(self) -> str:
        formatted = []
        for item in self.case.items:
            if item.amount is None:
                formatted.append(item.name)
            else:
                formatted.append(f"{item.name} (${item.amount:.2f})")
        return ", ".join(formatted)

    def _remember(self, user_message: str, agent_response: str) -> None:
        self.transcript.append({"role": "user", "content": user_message})
        self.transcript.append({"role": "assistant", "content": agent_response})

    @staticmethod
    def _extract_response_text(data: dict[str, object]) -> str | None:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        pieces = []
        for output in data.get("output", []) if isinstance(data.get("output"), list) else []:
            if not isinstance(output, dict):
                continue
            for content in output.get("content", []) if isinstance(output.get("content"), list) else []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
        text = "".join(pieces).strip()
        return text or None

    @staticmethod
    def _contains_any(text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _normalize(message: str) -> str:
        return re.sub(r"\s+", " ", message.lower().strip(" .,!?:;"))

    @staticmethod
    def _clean_phrase(value: str) -> str:
        return value.strip(" .,!?:;")

    @staticmethod
    def _clean_item_name(value: str) -> str:
        value = re.sub(
            r"\b(order|id|number|refund|return|exchange|replace|item|items|the|a|an|for|my|this|that)\b",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\b(?=[A-Z0-9-]*\d)[A-Z0-9-]{4,}\b", " ", value, flags=re.IGNORECASE)
        value = value.replace("`", " ")
        value = re.sub(r"\s+", " ", value).strip(" .,!?:;-")
        if value.lower() in {"", "order", "item", "items"}:
            return ""
        return value.lower()

    @staticmethod
    def _looks_like_reason(value: str) -> bool:
        return any(word in value.lower() for word in {"because", "broken", "damaged", "wrong", "late", "missing"})

    @staticmethod
    def _join_items(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def run_interactive_cli() -> None:
    agent = CustomerServiceAgent()
    print("Customer Service QA Agent")
    print(f"Mode: {agent.runtime_mode()}")
    print("Type a customer-service message. Type 'exit', 'quit', or 'done' to end.")
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
        if agent.conversation_finished():
            break
        print()


if __name__ == "__main__":
    run_interactive_cli()
