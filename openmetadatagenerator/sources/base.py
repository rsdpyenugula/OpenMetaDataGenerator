"""Metadata source abstraction.

A :class:`MetadataSource` knows how to enumerate tables and columns from one kind of
catalog and normalize them into the shared :mod:`openmetadatagenerator.model` types.
Lineage is optional: sources that expose it (e.g. DataHub) populate ``upstreams`` on
tables and columns; sources that don't simply leave them empty and the generator
falls back to schema/context signals.
"""
from __future__ import annotations

import abc

from ..model import Table


class MetadataSource(abc.ABC):
    """Read-only interface over a data catalog's technical metadata."""

    name: str = "base"

    @abc.abstractmethod
    def fetch_tables(self, keyword: str = "", limit: int | None = None) -> list[Table]:
        """Return tables (with columns, and lineage if available) matching ``keyword``.

        ``keyword`` is an optional substring/search filter on the fully-qualified
        name; an empty string means "everything the credentials can see".
        """
        raise NotImplementedError
