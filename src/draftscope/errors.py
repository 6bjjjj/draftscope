class DraftScopeError(RuntimeError):
    """An actionable configuration, compatibility, or result-integrity error."""


class CompatibilityError(DraftScopeError):
    """The installed software or selected GPU cannot execute the requested path."""
