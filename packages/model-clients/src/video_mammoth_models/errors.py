from enum import StrEnum


class ErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"


class ModelClientError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        request_id: str | None = None,
        details: object | None = None,
    ) -> None:
        if not isinstance(kind, ErrorKind):
            raise TypeError("kind must be an ErrorKind.")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a nonempty string.")

        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean.")

        if status_code is not None:
            if type(status_code) is not int:
                raise TypeError("status_code must be an integer or None.")
            if not 100 <= status_code <= 599:
                raise ValueError("status_code must be between 100 and 599.")

        if request_id is not None and (
            not isinstance(request_id, str) or not request_id.strip()
        ):
            raise ValueError("request_id must be a nonempty string or None.")

        super().__init__(kind, message)

        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        self.details = details

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, str | bool | int | None]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "request_id": self.request_id,
        }
