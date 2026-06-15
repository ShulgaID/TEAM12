"""Communication layer between simulator and controller."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from enum import Enum
import numpy as np


class MessageType(Enum):
    """Types of messages in the communication protocol."""
    # Simulator -> Communication -> Controller
    OBSERVATION = "observation"
    STATE_REQUEST = "state_request"

    # Controller -> Communication -> Simulator
    CONTROL_COMMAND = "control_command"
    STATE_REPORT = "state_report"

    # Bidirectional
    STATUS = "status"
    ERROR = "error"
    RESET = "reset"


@dataclass
class Message:
    """Base message structure for communication."""
    message_type: MessageType
    timestamp: float
    sender: str  # "simulator", "controller", "communication"
    receiver: str
    payload: Dict[str, Any] = field(default_factory=dict)
    sequence_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "sequence_id": self.sequence_id,
        }

    def to_json(self) -> str:
        """Convert message to JSON string."""
        data = self.to_dict()
        # Handle numpy arrays
        data["payload"] = self._serialize_payload(data["payload"])
        return json.dumps(data)

    @staticmethod
    def _serialize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize payload for JSON encoding."""
        serialized = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized


class CommunicationBuffer:
    """Thread-safe communication buffer for message passing."""

    def __init__(self, max_buffer_size: int = 100):
        """
        Initialize communication buffer.

        Args:
            max_buffer_size: Maximum number of messages to store
        """
        self.max_buffer_size = max_buffer_size
        self.incoming_messages = []
        self.outgoing_messages = []
        self.message_counter = 0

    def put_incoming(self, message: Message) -> None:
        """
        Add incoming message to buffer.

        Args:
            message: Message to add
        """
        message.sequence_id = self.message_counter
        self.message_counter += 1

        self.incoming_messages.append(message)

        # Keep buffer bounded
        if len(self.incoming_messages) > self.max_buffer_size:
            self.incoming_messages.pop(0)

    def put_outgoing(self, message: Message) -> None:
        """
        Add outgoing message to buffer.

        Args:
            message: Message to add
        """
        message.sequence_id = self.message_counter
        self.message_counter += 1

        self.outgoing_messages.append(message)

        # Keep buffer bounded
        if len(self.outgoing_messages) > self.max_buffer_size:
            self.outgoing_messages.pop(0)

    def get_incoming(self, message_type: Optional[MessageType] = None) -> Optional[Message]:
        """
        Get and remove first incoming message.

        Args:
            message_type: Filter by message type (None for any)

        Returns:
            Message if available, None otherwise
        """
        if not self.incoming_messages:
            return None

        if message_type is None:
            return self.incoming_messages.pop(0)

        for i, msg in enumerate(self.incoming_messages):
            if msg.message_type == message_type:
                return self.incoming_messages.pop(i)

        return None

    def get_outgoing(self, message_type: Optional[MessageType] = None) -> Optional[Message]:
        """
        Get and remove first outgoing message.

        Args:
            message_type: Filter by message type (None for any)

        Returns:
            Message if available, None otherwise
        """
        if not self.outgoing_messages:
            return None

        if message_type is None:
            return self.outgoing_messages.pop(0)

        for i, msg in enumerate(self.outgoing_messages):
            if msg.message_type == message_type:
                return self.outgoing_messages.pop(i)

        return None

    def peek_incoming(self) -> Optional[Message]:
        """Peek at first incoming message without removing."""
        return self.incoming_messages[0] if self.incoming_messages else None

    def peek_outgoing(self) -> Optional[Message]:
        """Peek at first outgoing message without removing."""
        return self.outgoing_messages[0] if self.outgoing_messages else None

    def incoming_count(self) -> int:
        """Get count of incoming messages."""
        return len(self.incoming_messages)

    def outgoing_count(self) -> int:
        """Get count of outgoing messages."""
        return len(self.outgoing_messages)

    def clear_incoming(self) -> None:
        """Clear all incoming messages."""
        self.incoming_messages.clear()

    def clear_outgoing(self) -> None:
        """Clear all outgoing messages."""
        self.outgoing_messages.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {
            "incoming_count": len(self.incoming_messages),
            "outgoing_count": len(self.outgoing_messages),
            "total_messages_processed": self.message_counter,
        }


class CommunicationLayer:
    """Central communication layer managing message flow."""

    def __init__(self):
        """Initialize communication layer."""
        self.buffer = CommunicationBuffer()
        self.message_log = []
        self.max_log_size = 1000
        self.enabled = True

    def send_observation(
        self,
        timestamp: float,
        observation: Dict[str, Any],
    ) -> None:
        """
        Send observation from simulator to controller.

        Args:
            timestamp: Current simulation time
            observation: Observation dictionary from simulator
        """
        message = Message(
            message_type=MessageType.OBSERVATION,
            timestamp=timestamp,
            sender="simulator",
            receiver="controller",
            payload={
                "observation": observation,
            },
        )

        self.buffer.put_incoming(message)
        self._log_message(message)

    def receive_observation(self) -> Optional[Dict[str, Any]]:
        """
        Receive observation in controller.

        Returns:
            Observation dictionary or None
        """
        message = self.buffer.get_incoming(MessageType.OBSERVATION)
        if message:
            return message.payload.get("observation")
        return None

    def send_control_command(
        self,
        timestamp: float,
        control_output: np.ndarray,
        controller_state: str,
    ) -> None:
        """
        Send control command from controller to simulator.

        Args:
            timestamp: Current time
            control_output: Motor control values
            controller_state: Current FSM state name
        """
        message = Message(
            message_type=MessageType.CONTROL_COMMAND,
            timestamp=timestamp,
            sender="controller",
            receiver="simulator",
            payload={
                "control_output": control_output,
                "controller_state": controller_state,
            },
        )

        self.buffer.put_outgoing(message)
        self._log_message(message)

    def receive_control_command(self) -> Optional[tuple]:
        """
        Receive control command in simulator.

        Returns:
            Tuple of (control_output, controller_state) or None
        """
        message = self.buffer.get_outgoing(MessageType.CONTROL_COMMAND)
        if message:
            return (
                message.payload.get("control_output"),
                message.payload.get("controller_state"),
            )
        return None

    def send_state_report(
        self,
        timestamp: float,
        controller_state: Dict[str, Any],
    ) -> None:
        """
        Send state report from controller.

        Args:
            timestamp: Current time
            controller_state: Current controller state information
        """
        message = Message(
            message_type=MessageType.STATE_REPORT,
            timestamp=timestamp,
            sender="controller",
            receiver="simulator",
            payload={
                "state_info": controller_state,
            },
        )

        self.buffer.put_outgoing(message)
        self._log_message(message)

    def receive_state_report(self) -> Optional[Dict[str, Any]]:
        """
        Receive state report.

        Returns:
            State information dictionary or None
        """
        message = self.buffer.get_outgoing(MessageType.STATE_REPORT)
        if message:
            return message.payload.get("state_info")
        return None

    def send_reset(self, timestamp: float) -> None:
        """
        Send reset signal.

        Args:
            timestamp: Current time
        """
        message = Message(
            message_type=MessageType.RESET,
            timestamp=timestamp,
            sender="simulator",
            receiver="controller",
            payload={},
        )

        self.buffer.put_incoming(message)
        self._log_message(message)

    def receive_reset(self) -> bool:
        """
        Check if reset message received.

        Returns:
            True if reset message exists
        """
        message = self.buffer.get_incoming(MessageType.RESET)
        return message is not None

    def send_error(self, timestamp: float, error_message: str) -> None:
        """
        Send error message.

        Args:
            timestamp: Current time
            error_message: Error description
        """
        message = Message(
            message_type=MessageType.ERROR,
            timestamp=timestamp,
            sender="system",
            receiver="all",
            payload={
                "error_message": error_message,
            },
        )

        self.buffer.put_incoming(message)
        self._log_message(message)

    def receive_error(self) -> Optional[str]:
        """
        Receive error message.

        Returns:
            Error message or None
        """
        message = self.buffer.get_incoming(MessageType.ERROR)
        if message:
            return message.payload.get("error_message")
        return None

    def _log_message(self, message: Message) -> None:
        """Log message for debugging."""
        self.message_log.append(message)

        if len(self.message_log) > self.max_log_size:
            self.message_log.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "buffer_stats": self.buffer.get_stats(),
            "message_log_size": len(self.message_log),
            "enabled": self.enabled,
        }

    def get_message_log(self, last_n: int = 50) -> list:
        """
        Get last N logged messages.

        Args:
            last_n: Number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        messages = self.message_log[-last_n:]
        return [msg.to_dict() for msg in messages]

    def clear_log(self) -> None:
        """Clear message log."""
        self.message_log.clear()
