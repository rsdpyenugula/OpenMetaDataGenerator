"""Sakila public-schema benchmark (a more complex real schema).

Sakila is a widely-used open-source sample database (a DVD-rental store). We encode a
15-table subset. It exercises every mechanism in OMDG:

* **canonicalize-first** — ``last_update`` appears in *every* table, and
  ``first_name``/``last_name``/``email``/``name``/``active`` recur across several, so the
  canonical pre-pass describes each concept once and seeds it everywhere;
* **lineage-aware waves** — foreign keys form a deep DAG
  (country -> city -> address -> {store, staff, customer} -> ... -> rental -> payment),
  including a ``store`` <-> ``staff`` cycle that exercises cycle handling;
* **inherit** — foreign-key columns carry fine-grained lineage to their referenced PK;
* **sibling** — sparse root tables leave obscure columns for the controller to fill;
* **confidence tags** — grounded vs. ungrounded objects receive High vs. Low tags.

Gold descriptions are paraphrased from the public Sakila documentation.
"""
from __future__ import annotations

from openmetadatagenerator.model import Column, Table

from .generate import Gold

# table: (columns[(name,type)], fk{col: upstream_table}, table_gold, {col: gold})
_SAKILA = {
    "country": ([("country_id", "int"), ("country", "string"), ("last_update", "timestamp")],
                {}, "Reference table of countries. One row per country.",
                {"country_id": "Unique identifier of the country.", "country": "Name of the country."}),
    "city": ([("city_id", "int"), ("city", "string"), ("country_id", "int"), ("last_update", "timestamp")],
             {"country_id": "country"}, "Reference table of cities, each in a country. One row per city.",
             {"city_id": "Unique identifier of the city.", "city": "Name of the city.",
              "country_id": "Country the city belongs to."}),
    "address": ([("address_id", "int"), ("address", "string"), ("district", "string"),
                 ("city_id", "int"), ("postal_code", "string"), ("phone", "string"),
                 ("last_update", "timestamp")], {"city_id": "city"},
                "Postal addresses used by customers, staff, and stores. One row per address.",
                {"address_id": "Unique identifier of the address.", "city_id": "City of the address.",
                 "district": "District or region of the address."}),
    "store": ([("store_id", "int"), ("manager_staff_id", "int"), ("address_id", "int"),
               ("last_update", "timestamp")], {"manager_staff_id": "staff", "address_id": "address"},
              "Physical rental stores. One row per store.",
              {"store_id": "Unique identifier of the store.",
               "manager_staff_id": "Staff member managing the store."}),
    "staff": ([("staff_id", "int"), ("first_name", "string"), ("last_name", "string"),
               ("address_id", "int"), ("email", "string"), ("store_id", "int"),
               ("active", "boolean"), ("username", "string"), ("last_update", "timestamp")],
              {"address_id": "address", "store_id": "store"},
              "Staff members employed at the stores. One row per staff member.",
              {"staff_id": "Unique identifier of the staff member.",
               "store_id": "Store the staff member works at.", "username": "Login username of the staff member."}),
    "customer": ([("customer_id", "int"), ("store_id", "int"), ("first_name", "string"),
                  ("last_name", "string"), ("email", "string"), ("address_id", "int"),
                  ("active", "boolean"), ("create_date", "timestamp"), ("last_update", "timestamp")],
                 {"store_id": "store", "address_id": "address"},
                 "Customers of the rental stores. One row per customer.",
                 {"customer_id": "Unique identifier of the customer.",
                  "store_id": "Home store of the customer.", "create_date": "When the customer record was created."}),
    "language": ([("language_id", "int"), ("name", "string"), ("last_update", "timestamp")],
                 {}, "Reference table of film languages. One row per language.",
                 {"language_id": "Unique identifier of the language.", "name": "Name of the language."}),
    "film": ([("film_id", "int"), ("title", "string"), ("description", "string"),
              ("release_year", "int"), ("language_id", "int"), ("rental_duration", "int"),
              ("rental_rate", "decimal"), ("length", "int"), ("replacement_cost", "decimal"),
              ("rating", "string"), ("last_update", "timestamp")], {"language_id": "language"},
             "Catalog of films available for rent. One row per film.",
             {"film_id": "Unique identifier of the film.", "title": "Title of the film.",
              "language_id": "Language of the film.", "rental_rate": "Cost to rent the film."}),
    "actor": ([("actor_id", "int"), ("first_name", "string"), ("last_name", "string"),
               ("last_update", "timestamp")], {},
              "Reference table of actors. One row per actor.",
              {"actor_id": "Unique identifier of the actor."}),
    "film_actor": ([("actor_id", "int"), ("film_id", "int"), ("last_update", "timestamp")],
                   {"actor_id": "actor", "film_id": "film"},
                   "Association of actors to the films they appear in. One row per actor-film pair.",
                   {"actor_id": "Actor appearing in the film.", "film_id": "Film the actor appears in."}),
    "category": ([("category_id", "int"), ("name", "string"), ("last_update", "timestamp")],
                 {}, "Reference table of film categories. One row per category.",
                 {"category_id": "Unique identifier of the category.", "name": "Name of the category."}),
    "film_category": ([("film_id", "int"), ("category_id", "int"), ("last_update", "timestamp")],
                      {"film_id": "film", "category_id": "category"},
                      "Association of films to categories. One row per film-category pair.",
                      {"film_id": "Film being categorized.", "category_id": "Category assigned to the film."}),
    "inventory": ([("inventory_id", "int"), ("film_id", "int"), ("store_id", "int"),
                   ("last_update", "timestamp")], {"film_id": "film", "store_id": "store"},
                  "Physical inventory copies of films at stores. One row per inventory item.",
                  {"inventory_id": "Unique identifier of the inventory item.",
                   "film_id": "Film this copy is of.", "store_id": "Store holding this copy."}),
    "rental": ([("rental_id", "int"), ("rental_date", "timestamp"), ("inventory_id", "int"),
                ("customer_id", "int"), ("return_date", "timestamp"), ("staff_id", "int"),
                ("last_update", "timestamp")],
               {"inventory_id": "inventory", "customer_id": "customer", "staff_id": "staff"},
               "Rentals of inventory items by customers. One row per rental.",
               {"rental_id": "Unique identifier of the rental.", "rental_date": "When the item was rented.",
                "return_date": "When the item was returned."}),
    "payment": ([("payment_id", "int"), ("customer_id", "int"), ("staff_id", "int"),
                 ("rental_id", "int"), ("amount", "decimal"), ("payment_date", "timestamp"),
                 ("last_update", "timestamp")],
                {"customer_id": "customer", "staff_id": "staff", "rental_id": "rental"},
                "Payments made by customers for rentals. One row per payment.",
                {"payment_id": "Unique identifier of the payment.", "amount": "Amount paid.",
                 "payment_date": "When the payment was made."}),
}

# Shared canonical concepts (paraphrased once; gold used for every occurrence).
_SHARED = {
    "last_update": "Timestamp when this row was last updated.",
    "first_name": "First name of the person.",
    "last_name": "Last name of the person.",
    "email": "Email address of the person.",
    "active": "Whether the record is currently active.",
}


def build_sakila(with_doc_context: bool = False) -> tuple[list[Table], Gold]:
    tables: list[Table] = []
    gold = Gold()
    for name, (cols, fks, tdesc, cdesc) in _SAKILA.items():
        fqn = f"sakila.public.{name}"
        columns = []
        for cn, ct in cols:
            up = fks.get(cn)
            ups = [(f"sakila.public.{up}", f"{up}_id")] if up else []
            columns.append(Column(cn, ct, upstreams=ups))
        t = Table("sakila", "public", name, columns=columns,
                  upstreams=[f"sakila.public.{u}" for u in dict.fromkeys(fks.values())])
        if with_doc_context:
            t.doc_context = f"[sakila] {name}: {tdesc}"
        tables.append(t)
        gold.table_desc[fqn] = tdesc
        for cn, cd in cdesc.items():
            gold.column_desc[f"{fqn}.{cn}"] = cd
        for cn, cd in _SHARED.items():
            if any(c.name == cn for c in columns):
                gold.column_desc[f"{fqn}.{cn}"] = cd
    return tables, gold
