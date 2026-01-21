"""Notification service for order execution alerts via Email and Slack."""
from __future__ import annotations

import logging
import os
import smtplib
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    @abstractmethod
    def send(self, subject: str, message: str) -> bool:
        """Send a notification. Returns True on success."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Gmail SMTP notification provider."""

    def __init__(self) -> None:
        self.gmail_address = os.getenv("GMAIL_ADDRESS", "")
        self.gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_configured(self) -> bool:
        enabled = os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() == "true"
        return enabled and bool(self.gmail_address) and bool(self.gmail_app_password)

    def send(self, subject: str, message: str) -> bool:
        if not self.is_configured():
            return False

        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.gmail_address
            msg["To"] = self.gmail_address

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.gmail_address, self.gmail_app_password)
                server.sendmail(self.gmail_address, self.gmail_address, msg.as_string())

            self.logger.info(f"Email notification sent: {subject}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class SlackNotificationProvider(NotificationProvider):
    """Slack webhook notification provider."""

    def __init__(self) -> None:
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_configured(self) -> bool:
        enabled = os.getenv("NOTIFICATION_SLACK_ENABLED", "false").lower() == "true"
        return enabled and bool(self.webhook_url)

    def send(self, subject: str, message: str) -> bool:
        if not self.is_configured():
            return False

        try:
            payload = {"text": f"*{subject}*\n```{message}```"}
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            self.logger.info(f"Slack notification sent: {subject}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False


class NotificationService:
    """Service for sending order execution notifications."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.providers: List[NotificationProvider] = []
        self._auto_configure_providers()

    def _auto_configure_providers(self) -> None:
        """Auto-configure providers based on environment variables."""
        email_provider = EmailNotificationProvider()
        if email_provider.is_configured():
            self.providers.append(email_provider)
            self.logger.info("Email notification provider configured")

        slack_provider = SlackNotificationProvider()
        if slack_provider.is_configured():
            self.providers.append(slack_provider)
            self.logger.info("Slack notification provider configured")

        if not self.providers:
            self.logger.info("No notification providers configured")

    def notify_order_submitted(self, order_dict: Dict[str, Any]) -> None:
        """Send notification when order is submitted."""
        if not self.providers:
            return
        subject = self._format_subject(order_dict, "SUBMITTED")
        message = self._format_order_message(order_dict)
        for provider in self.providers:
            provider.send(subject, message)

    def notify_order_filled(self, order_dict: Dict[str, Any]) -> None:
        """Send notification when order is filled."""
        if not self.providers:
            return
        subject = self._format_subject(order_dict, "FILLED")
        message = self._format_order_message(order_dict)
        for provider in self.providers:
            provider.send(subject, message)

    def notify_order_executed(self, order_dict: Dict[str, Any]) -> None:
        """Send notification for an executed order. Deprecated: use submitted/filled."""
        self.notify_order_submitted(order_dict)

    def _format_subject(self, order: Dict[str, Any], event: str = "ORDER") -> str:
        """Format notification subject line."""
        symbol = order.get("symbol", "UNKNOWN")
        side = str(order.get("side", "UNKNOWN")).upper()
        if hasattr(order.get("side"), "value"):
            side = order["side"].value.upper()
        return f"[GaussWorldTrader] {event}: {side} {symbol}"

    def _format_order_message(self, order: Dict[str, Any]) -> str:
        """Format order details into readable message."""
        symbol = order.get("symbol", "UNKNOWN")
        side = order.get("side", "UNKNOWN")
        if hasattr(side, "value"):
            side = side.value
        qty = order.get("qty", 0)
        order_type = order.get("type", "UNKNOWN")
        if hasattr(order_type, "value"):
            order_type = order_type.value
        status = order.get("status", "UNKNOWN")
        if hasattr(status, "value"):
            status = status.value
        order_id = order.get("id", "UNKNOWN")
        submitted_at = order.get("submitted_at", datetime.now())
        filled_at = order.get("filled_at")
        filled_qty = order.get("filled_qty", 0)
        price = order.get("filled_avg_price") or order.get("limit_price") or order.get("stop_price")

        lines = [
            f"Symbol: {symbol}",
            f"Side: {str(side).upper()}",
            f"Quantity: {qty}",
            f"Type: {str(order_type).upper()}",
            f"Status: {str(status).lower()}",
        ]

        if filled_qty:
            lines.append(f"Filled Qty: {filled_qty}")
        if price:
            lines.append(f"Price: ${price}")

        lines.append(f"Order ID: {order_id}")
        lines.append(f"Submitted: {submitted_at}")
        if filled_at:
            lines.append(f"Filled: {filled_at}")

        return "\n".join(lines)


class TradeStreamHandler:
    """Monitors order fills using Alpaca's streaming API."""

    def __init__(self, notification_service: NotificationService) -> None:
        self.notification_service = notification_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start listening for trade updates in a background thread."""
        if self._running:
            self.logger.warning("Trade stream already running")
            return

        try:
            from alpaca.trading.stream import TradingStream
            from config import Config

            if not Config.validate_alpaca_config():
                self.logger.error("Alpaca credentials not configured")
                return

            paper = Config.ALPACA_BASE_URL != "https://api.alpaca.markets"
            self._stream = TradingStream(
                api_key=Config.ALPACA_API_KEY,
                secret_key=Config.ALPACA_SECRET_KEY,
                paper=paper
            )
            self._stream.subscribe_trade_updates(self._handle_trade_update)
            self._running = True
            self._thread = threading.Thread(target=self._run_stream, daemon=True)
            self._thread.start()
            self.logger.info("Trade stream started")
        except Exception as e:
            self.logger.error(f"Failed to start trade stream: {e}")

    def _run_stream(self) -> None:
        """Run the stream in background thread."""
        try:
            self._stream.run()
        except Exception as e:
            self.logger.error(f"Trade stream error: {e}")
            self._running = False

    async def _handle_trade_update(self, data) -> None:
        """Handle incoming trade update events."""
        try:
            event = data.event
            order = data.order

            if event in ("fill", "partial_fill"):
                order_dict = {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": float(order.qty) if order.qty else 0,
                    "side": order.side,
                    "type": order.type,
                    "status": order.status,
                    "submitted_at": order.submitted_at,
                    "filled_at": order.filled_at,
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                }
                self.notification_service.notify_order_filled(order_dict)
                self.logger.info(f"Order filled notification sent: {order.symbol}")
        except Exception as e:
            self.logger.error(f"Error handling trade update: {e}")

    def stop(self) -> None:
        """Stop the trade stream."""
        if self._stream and self._running:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._running = False
            self.logger.info("Trade stream stopped")
