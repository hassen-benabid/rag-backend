"""Domain errors raised by the ML layer. The API layer maps them to HTTP codes."""


class DocumentNotFoundError(Exception):
    """A document the caller asked for isn't in the vector store."""


class MLServiceUnavailableError(Exception):
    """A model couldn't be loaded (e.g. missing API key)."""
