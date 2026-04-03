class ServiceError(RuntimeError):
    """Base class for service-layer errors."""


class BadInputError(ServiceError):
    """Raised when a request payload is invalid."""


class NotFoundError(ServiceError):
    """Raised when a requested file or session cannot be found."""


class ExternalServiceError(ServiceError):
    """Raised when a required upstream service fails."""
