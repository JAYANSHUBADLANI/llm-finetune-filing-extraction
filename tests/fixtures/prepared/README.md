# Frozen prepared filings

Two filings, copied from the `filing-extraction-benchmark` project's
`data/cache/prepared/` cache, so that `tests/test_data_prep.py` runs without
that project checked out next to this one.

| File | Filing |
| --- | --- |
| `0000012927_0000012927-20-000014.json` | Boeing, FY2019 10-K |
| `0000021344_0000021344-21-000008.json` | Coca-Cola, FY2020 10-K |

Each is reduced to the fields `load_statement_text` reads: `doc_id`, and for
every table that the source project's parser resolved to a financial statement,
its `index`, `statement`, `scale_phrase` and `rows`. Tables the parser left
unresolved are dropped, which is exactly what `load_statement_text` does with
them, so the reduced file produces output identical to the full one. That was
checked against the originals at several `max_chars` values before committing,
not assumed. Dropping the rest takes the pair from 1.6 MB to 87 KB.

The underlying documents are SEC EDGAR filings and are public records. The
second one is here specifically because it has a bare `$` cell on some rows and
not others, the case `_is_meaningful_cell` exists to handle.
