"""DataHub Core metadata source (via the GraphQL API).

Pulls datasets, their schema fields, coarse table lineage, and fine-grained
column lineage. Works against an open-source DataHub deployment; only the GMS URL
and a personal access token are required, both read from the environment:

    OMDG_DATAHUB_GRAPHQL   e.g. https://<host>/api/graphql
    OMDG_DATAHUB_TOKEN     a personal access token

The GraphQL ``FacetFilterInput`` uses ``values`` (a list); this is the portable
form accepted across DataHub versions.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import MetadataSource
from ..model import Column, Table

_SEARCH = """
query($input: SearchInput!) {
  search(input: $input) {
    total
    searchResults { entity { urn
      ... on Dataset {
        name
        properties { description }
        schemaMetadata { fields { fieldPath type description } }
      } } }
  }
}"""

_LINEAGE = """
query($urn: String!) {
  dataset(urn: $urn) {
    upstream: lineage(input: {direction: UPSTREAM, start: 0, count: 50}) {
      relationships { entity { urn } } }
    fineGrainedLineages { upstreams { urn path } downstreams { urn path } }
  }
}"""


class DataHubSource(MetadataSource):
    name = "datahub"

    def __init__(self, graphql_url: str | None = None, token: str | None = None,
                 platform: str = "databricks", workers: int = 16):
        self.url = graphql_url or os.environ["OMDG_DATAHUB_GRAPHQL"]
        self.token = token or os.environ.get("OMDG_DATAHUB_TOKEN", "")
        self.platform = platform
        self.workers = workers

    def _gql(self, query: str, variables: dict) -> dict:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        r = requests.post(self.url, json={"query": query, "variables": variables},
                          headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_tables(self, keyword: str = "", limit: int | None = None) -> list[Table]:
        tables: list[Table] = []
        start, page = 0, 100
        while True:
            data = self._gql(_SEARCH, {"input": {
                "type": "DATASET", "query": keyword or "*", "start": start, "count": page,
                "filters": [{"field": "platform",
                             "values": [f"urn:li:dataPlatform:{self.platform}"]}],
            }})
            search = data["data"]["search"]
            results = search["searchResults"]
            for r in results:
                e = r["entity"]
                urn = e.get("urn", "")
                m = re.search(rf":{self.platform},([^,]+),", urn)
                if not m:
                    continue
                parts = m.group(1).split(".")
                if len(parts) < 3:
                    continue
                catalog, schema, name = parts[0], parts[1], ".".join(parts[2:])
                cols = [Column(name=f["fieldPath"].split(".")[-1],
                               data_type=f.get("type") or "",
                               description=f.get("description") or "")
                        for f in ((e.get("schemaMetadata") or {}).get("fields") or [])]
                t = Table(catalog=catalog, schema=schema, name=name, columns=cols,
                          description=(e.get("properties") or {}).get("description") or "")
                t._urn = urn  # type: ignore[attr-defined]
                tables.append(t)
                if limit and len(tables) >= limit:
                    return self._attach_lineage(tables)
            start += len(results)
            if start >= search["total"] or not results:
                break
        return self._attach_lineage(tables)

    def _attach_lineage(self, tables: list[Table]) -> list[Table]:
        by_urn = {getattr(t, "_urn", ""): t for t in tables}

        def one(t: Table):
            try:
                d = self._gql(_LINEAGE, {"urn": getattr(t, "_urn", "")})["data"]["dataset"] or {}
                rels = (d.get("upstream") or {}).get("relationships") or []
                t.upstreams = [r["entity"]["urn"] for r in rels if r.get("entity")]
                col_map: dict[str, list[tuple[str, str]]] = {}
                for entry in (d.get("fineGrainedLineages") or []):
                    for dn in (entry.get("downstreams") or []):
                        tgt = (dn.get("path") or "").strip()
                        if not tgt:
                            continue
                        for up in (entry.get("upstreams") or []):
                            if up.get("urn") and up.get("path"):
                                col_map.setdefault(tgt.lower(), []).append(
                                    (up["urn"], up["path"].split(".")[-1]))
                for c in t.columns:
                    c.upstreams = col_map.get(c.name.lower(), [])
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(as_completed([pool.submit(one, t) for t in tables]))
        return tables
