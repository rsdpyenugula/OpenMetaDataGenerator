"""Snowflake metadata source.

Reads technical metadata from ``INFORMATION_SCHEMA`` (``TABLES``, ``COLUMNS``,
``VIEWS``) for a given database. Object-level lineage is available in Snowflake's
``SNOWFLAKE.ACCOUNT_USAGE.OBJECT_DEPENDENCIES`` (queried when accessible); column
lineage is not exposed there and is left empty.

Connection settings come from the environment:

    OMDG_SF_ACCOUNT, OMDG_SF_USER, OMDG_SF_PASSWORD (or key-pair),
    OMDG_SF_WAREHOUSE, OMDG_SF_DATABASE, OMDG_SF_ROLE
"""
from __future__ import annotations

import os

from ..model import Column, Table
from .base import MetadataSource


class SnowflakeSource(MetadataSource):
    name = "snowflake"

    def __init__(self, database: str | None = None):
        self.database = database or os.environ.get("OMDG_SF_DATABASE", "")

    def _connect(self):
        import snowflake.connector
        return snowflake.connector.connect(
            account=os.environ.get("OMDG_SF_ACCOUNT", ""),
            user=os.environ.get("OMDG_SF_USER", ""),
            password=os.environ.get("OMDG_SF_PASSWORD", ""),
            warehouse=os.environ.get("OMDG_SF_WAREHOUSE", ""),
            database=self.database,
            role=os.environ.get("OMDG_SF_ROLE", ""),
        )

    def fetch_tables(self, keyword: str = "", limit: int | None = None) -> list[Table]:
        db = self.database or keyword
        tables: dict[str, Table] = {}
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT table_schema, table_name, comment, table_type
                FROM {db}.INFORMATION_SCHEMA.TABLES
                WHERE table_schema <> 'INFORMATION_SCHEMA'
                ORDER BY table_schema, table_name
            """)
            for schema, name, comment, ttype in cur.fetchall():
                if limit and len(tables) >= limit:
                    break
                tables[f"{schema}.{name}".lower()] = Table(
                    catalog=db, schema=schema, name=name, description=comment or "",
                    view_definition="VIEW" if (ttype or "").upper() == "VIEW" else "")

            cur.execute(f"""
                SELECT table_schema, table_name, column_name, data_type, comment
                FROM {db}.INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema <> 'INFORMATION_SCHEMA'
                ORDER BY table_schema, table_name, ordinal_position
            """)
            for schema, tname, cname, dtype, ccomment in cur.fetchall():
                t = tables.get(f"{schema}.{tname}".lower())
                if t is not None:
                    t.columns.append(Column(name=cname, data_type=dtype or "",
                                            description=ccomment or ""))
        finally:
            conn.close()
        return list(tables.values())
