from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ontocellia.framework.cell import CellPosition
from ontocellia.framework.topology import TissueTopology


MessageScope = Literal["direct", "local", "fate", "broadcast"]


@dataclass(slots=True)
class TissueMessage:
    id: str
    sender_cell_id: int
    scope: MessageScope | str
    kind: str
    content: str
    confidence: float = 0.5
    recipient_cell_id: int | None = None
    recipient_fate: str | None = None
    references: list[str] = field(default_factory=list)
    created_tick: int = 0
    ttl: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MessageDelivery:
    message_id: str
    recipient_cell_id: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandoffRequest:
    id: str
    source_cell_id: int
    target_fate: str
    message_id: str
    content: str
    created_tick: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandoffReceipt:
    request_id: str
    recipient_cell_id: int
    accepted: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatrixRecord:
    id: str
    source_cell_id: int
    kind: str
    content: str
    tags: list[str]
    position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any]
    confidence: float = 0.5
    created_tick: int = 0
    expires_tick: int | None = None
    fate: str | None = None

    def __post_init__(self) -> None:
        self.position = CellPosition.from_value(self.position)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["position"] = {
            "node_id": self.position.node_id,
            "region": self.position.region,
            "neighbors": list(self.position.neighbors),
            "embedding": list(self.position.embedding),
        }
        return data


@dataclass(slots=True)
class ExtracellularMatrix:
    records: list[MatrixRecord] = field(default_factory=list)

    def deposit(self, record: MatrixRecord) -> None:
        self.records.append(record)

    def query(
        self,
        tags: list[str] | None = None,
        fate: str | None = None,
        position: CellPosition | None = None,
        topology: TissueTopology | None = None,
        limit: int = 5,
    ) -> list[MatrixRecord]:
        wanted_tags = {str(tag) for tag in tags or []}
        candidates = [
            record
            for record in self.records
            if (not wanted_tags or wanted_tags <= set(record.tags))
            and (fate is None or record.fate == fate)
        ]
        if position is not None:
            candidates.sort(key=lambda record: (_matrix_distance(record.position, position, topology), record.created_tick, record.id))
        else:
            candidates.sort(key=lambda record: (-record.created_tick, record.id))
        return candidates[:limit]

    def decay(self, current_tick: int) -> None:
        self.records = [
            record
            for record in self.records
            if record.expires_tick is None or record.expires_tick > current_tick
        ]


@dataclass(slots=True)
class CommunicationPolicy:
    matrix_query_limit: int = 5
    default_ttl: int = 3
    promote_confidence_threshold: float = 0.6
    allow_broadcast: bool = True
    broadcast_limit: int = 8


@dataclass(slots=True)
class CommunicationRuntime:
    message_counter: int = 0
    record_counter: int = 0
    handoff_counter: int = 0

    def emit_from_actions(self, tissue: Any, actions: list[dict[str, Any]]) -> list[TissueMessage]:
        messages: list[TissueMessage] = []
        for action in actions:
            payload = dict(action.get("payload", {}))
            content = str(payload.get("message") or action.get("rationale") or action.get("intent_type") or action.get("gene_id") or "")
            if not content:
                continue
            sender_cell_id = int(action.get("cell_id", 0))
            handoff_to_fate = payload.get("handoff_to_fate")
            intent_type = str(action.get("intent_type", ""))
            kind = "memory" if intent_type == "record_memory" else "handoff" if handoff_to_fate else "observation"
            message = TissueMessage(
                id=self._next_message_id(),
                sender_cell_id=sender_cell_id,
                recipient_fate=str(handoff_to_fate) if handoff_to_fate else None,
                scope="fate" if handoff_to_fate else "broadcast" if kind == "memory" else "local",
                kind=kind,
                content=content,
                confidence=float(action.get("confidence", 0.5)),
                references=[str(item) for item in payload.get("matrix_tags", [])],
                created_tick=int(getattr(tissue, "tick_count", 0)),
                ttl=int(getattr(tissue.environment.communication_policy, "default_ttl", 3)),
            )
            messages.append(message)
            tissue.trace.record("message_emitted", **message.as_dict())
        return messages

    def route(self, tissue: Any, messages: list[TissueMessage] | None = None) -> list[MessageDelivery]:
        policy = tissue.environment.communication_policy
        routed_messages = list(messages or [])
        deliveries: list[MessageDelivery] = []
        for message in routed_messages:
            recipients = self._recipients(tissue, message, policy)
            for recipient in recipients:
                delivery = MessageDelivery(message.id, recipient.id)
                deliveries.append(delivery)
                recipient.record_event("message_received", message_id=message.id, sender_cell_id=message.sender_cell_id, kind=message.kind)
                tissue.trace.record("message_delivered", message_id=message.id, recipient_cell_id=recipient.id, sender_cell_id=message.sender_cell_id)
            self._maybe_deposit(tissue, message, policy)
            if message.kind == "handoff":
                self._record_handoff_request(tissue, message)
        return deliveries

    def resolve_handoffs(self, tissue: Any) -> list[HandoffReceipt]:
        requests = [event for event in tissue.trace.events if event["type"] == "handoff_requested"]
        receipts: list[HandoffReceipt] = []
        completed = {event["request_id"] for event in tissue.trace.events if event["type"] == "handoff_completed"}
        for request in requests:
            if request["request_id"] in completed:
                continue
            recipients = sorted(
                [cell for cell in tissue.cells.values() if cell.alive and cell.fate == request["target_fate"]],
                key=lambda cell: cell.id,
            )
            if not recipients:
                continue
            receipt = HandoffReceipt(request_id=request["request_id"], recipient_cell_id=recipients[0].id)
            receipts.append(receipt)
            tissue.trace.record("handoff_completed", request_id=receipt.request_id, recipient_cell_id=receipt.recipient_cell_id, accepted=receipt.accepted)
        return receipts

    def _recipients(self, tissue: Any, message: TissueMessage, policy: CommunicationPolicy) -> list[Any]:
        sender = tissue.cells.get(message.sender_cell_id)
        cells = [cell for cell in tissue.cells.values() if cell.alive and cell.id != message.sender_cell_id]
        if message.scope == "direct":
            return sorted([cell for cell in cells if cell.id == message.recipient_cell_id], key=lambda cell: cell.id)
        if message.scope == "fate":
            return sorted([cell for cell in cells if cell.fate == message.recipient_fate], key=lambda cell: cell.id)
        if message.scope == "broadcast":
            if not policy.allow_broadcast:
                return []
            return sorted(cells, key=lambda cell: cell.id)[: policy.broadcast_limit]
        if message.scope == "local" and sender is not None:
            return sorted([cell for cell in cells if _local_to(sender.position, cell.position, tissue.environment.topology)], key=lambda cell: cell.id)
        return []

    def _maybe_deposit(self, tissue: Any, message: TissueMessage, policy: CommunicationPolicy) -> None:
        if message.kind != "memory" and message.confidence < policy.promote_confidence_threshold:
            return
        sender = tissue.cells.get(message.sender_cell_id)
        position = sender.position if sender is not None else CellPosition("")
        record = MatrixRecord(
            id=self._next_record_id(),
            source_cell_id=message.sender_cell_id,
            kind=message.kind,
            content=message.content,
            tags=list(message.references or [message.kind]),
            position=position,
            confidence=message.confidence,
            created_tick=message.created_tick,
            expires_tick=None if message.kind == "memory" else message.created_tick + message.ttl,
            fate=getattr(sender, "fate", None),
        )
        tissue.environment.matrix.deposit(record)
        tissue.trace.record("matrix_deposit", **record.as_dict())

    def _record_handoff_request(self, tissue: Any, message: TissueMessage) -> None:
        if not message.recipient_fate:
            return
        request = HandoffRequest(
            id=self._next_handoff_id(),
            source_cell_id=message.sender_cell_id,
            target_fate=message.recipient_fate,
            message_id=message.id,
            content=message.content,
            created_tick=message.created_tick,
        )
        tissue.trace.record("handoff_requested", request_id=request.id, source_cell_id=request.source_cell_id, target_fate=request.target_fate, message_id=request.message_id, content=request.content)

    def _next_message_id(self) -> str:
        self.message_counter += 1
        return f"msg-{self.message_counter}"

    def _next_record_id(self) -> str:
        self.record_counter += 1
        return f"matrix-{self.record_counter}"

    def _next_handoff_id(self) -> str:
        self.handoff_counter += 1
        return f"handoff-{self.handoff_counter}"


def _local_to(left: CellPosition, right: CellPosition, topology: TissueTopology | None) -> bool:
    if left.node_id == right.node_id:
        return True
    if left.node_id in right.neighbors or right.node_id in left.neighbors:
        return True
    if left.region and left.region == right.region:
        return True
    if topology is not None:
        return topology.distance(left, right) <= 1.0
    return False


def _matrix_distance(left: CellPosition, right: CellPosition, topology: TissueTopology | None) -> float:
    if topology is not None:
        return topology.distance(left, right)
    if left.node_id == right.node_id:
        return 0.0
    if left.region and left.region == right.region:
        return 1.0
    return 10.0
