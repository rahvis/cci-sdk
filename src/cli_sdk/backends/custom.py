"""Re-export of ``CustomBackend`` so ``from cli_sdk.backends.custom import CustomBackend`` works."""

from cli_sdk.backends.base import CustomBackend

__all__ = ["CustomBackend"]
