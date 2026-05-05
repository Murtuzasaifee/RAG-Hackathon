class RAGError(Exception):
    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(message)


class IngestionError(RAGError):
    pass


class RetrievalError(RAGError):
    pass


class GenerationError(RAGError):
    pass


class GuardError(RAGError):
    pass


class GatewayError(RAGError):
    pass
