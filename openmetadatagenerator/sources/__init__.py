"""Metadata sources and a small factory."""
from __future__ import annotations

from .base import MetadataSource


def get_source(name: str, **kwargs) -> MetadataSource:
    """Instantiate a metadata source by name.

    Imports are lazy so that installing only the extras you use is enough
    (e.g. ``pip install openmetadatagenerator[snowflake]``).
    """
    name = name.lower()
    if name == "datahub":
        from .datahub import DataHubSource
        return DataHubSource(**kwargs)
    if name == "databricks":
        from .databricks import DatabricksSource
        return DatabricksSource(**kwargs)
    if name == "snowflake":
        from .snowflake import SnowflakeSource
        return SnowflakeSource(**kwargs)
    raise ValueError(f"unknown source: {name!r} (expected datahub|databricks|snowflake)")


__all__ = ["MetadataSource", "get_source"]
