"""Compatibility shims for vLLM-Omni 0.24 running on Python 3.10."""

import enum


if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        """Backport the Python 3.11 enum.StrEnum behavior used by vLLM-Omni."""

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            del start, count, last_values
            return name.lower()

        def __str__(self):
            return str(self.value)

    enum.StrEnum = StrEnum
