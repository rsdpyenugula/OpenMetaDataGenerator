"""Databricks (Unity Catalog) metadata source.

Reads technical metadata directly from ``system.information_schema`` — the
``tables``, ``columns``, and ``views`` relations — via the Databricks SQL
connector. Connection settings come from the environment:

    OMDG_DBX_HOST        Databricks SQL warehouse host
    OMDG_DBX_HTTP_PATH   warehouse HTTP path
    OMDG_DBX_TOKEN       access token
    OMDG_DBX_CATALOG     catalog to enumerate (default: the search keyword)

Unity Catalog has no first-class column lineage in information_schema, so lineage
is left empty here; pair with :class:`DataHubSource` when lineage is required.
"""
from __future__ import annotations

import os

from ..model import Column, Table
from .base import MetadataSource


class DatabricksSource(MetadataSource):
    name = "databricks"

    def __init__(self, catalog: str | None = None):
        self.host = os.environ.get("OMDG_DBX_HOST", "")
        self.http_path = os.environ.get("OMDG_DBX_HTTP_PATH", "")
        self.token = os.environ.get("OMDG_DBX_TOKEN", "")
        self.catalog = catalog or os.environ.get("OMDG_DBX_CATALOG", "")

    def _connect(self):
        from databricks import sql
        return sql.connect(server_hostname=self.host, http_path=self.http_path,
                           access_token=self.token)

    def fetch_tables(self, keyword: str = "", limit: int | None = None) -> list[Table]:
        catalog = self.catalog or keyword
        tables: dict[str, Table] = {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT table_schema, table_name, comment, table_type
                FROM {catalog}.information_schema.tables
                ORDER BY table_schema, table_name
            """)
            for schema, name, comment, ttype in cur.fetchall():
                if limit and len(tables) >= limit:
                    break
                tables[f"{schema}.{name}"] = Table(
                    catalog=catalog, schema=schema, name=name,
                    description=comment or "",
                    view_definition="VIEW" if (ttype or "").upper().endswith("VIEW") else "")

            cur.execute(f"""
                SELECT table_schema, table_name, column_name, full_data_type, comment
                FROM {catalog}.information_schema.columns
                ORDER BY table_schema, table_name, ordinal_position
            """)
            for schema, tname, cname, dtype, ccomment in cur.fetchall():
                t = tables.get(f"{schema}.{tname}")
                if t is not None:
                    t.columns.append(Column(name=cname, data_type=dtype or "",
                                            description=ccomment or ""))

            # View SQL (best-effort; the relation may be restricted).
            try:
                cur.execute(f"""
                    SELECT table_schema, table_name, view_definition
                    FROM {catalog}.information_schema.views
                """)
                for schema, tname, vdef in cur.fetchall():
                    t = tables.get(f"{schema}.{tname}")
                    if t is not None and vdef:
                        t.view_definition = vdef
            except Exception:
                pass
        return list(tables.values())
