from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from ..missions import Mission


class MissionList(list):
    """Mission list, accessible from name: list.ztf, list["ztf"]."""

    def _by_name(self):
        return {m.name.lower(): m for m in self}

    def __getattr__(self, key):
        try:
            return self._by_name()[key.lower()]
        except KeyError:
            raise AttributeError(f"Pas de mission '{key}' dans le réseau") from None

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)

    def __dir__(self):
        return list(super().__dir__()) + list(self._by_name())


@dataclass(repr=False)
class Network:
    """Base class for a network of detectors (missions)."""

    name: str
    """Name of the telescope network"""

    telescopes: list[Mission]
    """List of the telescopes present in the network"""

    priorities: dict[str, float] = field(default_factory=dict)
    """Priority of each telescope, keyed by mission name.

    Like matplotlib's ``zorder``: the higher the value, the higher the priority.
    Missions that are not listed get a priority of 0.
    """

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, got {type(self.name).__name__}")

        # Accepts Mission instances or mission names (same parsing as the CLI)
        self.telescopes = MissionList(Mission._cli_parse(t) for t in self.telescopes)

        names = [t.name.lower() for t in self.telescopes]
        if len(set(names)) != len(names):
            raise ValueError("Missions can't be used twice in the same network")

        self.priorities = {k.lower(): float(p) for k, p in self.priorities.items()}
        unknown = set(self.priorities) - set(names)
        if unknown:
            raise ValueError(
                f"Priorities given for missions not in the network: {sorted(unknown)}"
            )

    # ---------- CLI / pydantic parsing ----------

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Let Pydantic and Typer parse a network by name."""
        return core_schema.no_info_plain_validator_function(cls._cli_parse)

    @classmethod
    def _cli_parse(cls, value: Any) -> "Network":
        if isinstance(value, cls):
            return value
        # Deferred to avoid a circular import.
        from .. import networks

        choices = {
            name: obj
            for name in networks.__all__
            if isinstance(obj := getattr(networks, name), cls)
        }
        try:
            return choices[value]
        except KeyError:
            raise ValueError(
                f"{value!r} is not one of {', '.join(map(repr, choices))}."
            ) from None

    # ---------- priorities ----------

    def priority(self, mission: "Mission | str") -> float:
        name = mission if isinstance(mission, str) else mission.name
        return self.priorities.get(name.lower(), 0)

    def by_priority(self, *, reverse: bool = True) -> list[Mission]:
        """Missions sorted by priority (highest first by default).

        The sort is stable: ties keep the order of ``telescopes``.
        """
        return sorted(self.telescopes, key=self.priority, reverse=reverse)

    # ---------- container behaviour ----------

    def __iter__(self) -> Iterator[Mission]:
        return iter(self.telescopes)

    def __len__(self) -> int:
        return len(self.telescopes)

    def __getattr__(self, key):
        # Only called when normal lookup fails: net.ztf -> net.telescopes.ztf
        telescopes = self.__dict__.get("telescopes")
        if telescopes is None or key.startswith("__"):
            raise AttributeError(key)
        return getattr(telescopes, key)

    def __dir__(self):
        return list(super().__dir__()) + [m.name.lower() for m in self.telescopes]
