"""Context provider abstraction.

A context provider inspects an external corpus (source code, documentation, ...) and
attaches free-text evidence to each table that the generator can ground its
descriptions in. Providers are additive and order-independent; the pipeline runs all
configured providers over the table set before generation.
"""
from __future__ import annotations

import abc

from ..model import Table


class ContextProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def attach(self, tables: list[Table]) -> None:
        """Mutate ``tables`` in place, populating the relevant context field."""
        raise NotImplementedError
