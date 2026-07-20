# StreamXor Architecture

## 1. What this system is

StreamXor is a chunked, columnar time-series storage and query engine for OHLCV
(open/high/low/close/volume) market data, with **bitemporal versioning**: every
stored fact carries both a *valid time* (when the event happened) and a
*transaction time* (when the system learned about it), so a query can ask "what
did we know, as of this point in time" without ever leaking a future
correction into a past-dated decision — the core guarantee a backtesting
engine needs to avoid look-ahead bias.

The design center of the project is **conservative predicate pushdown**: rather
than compressing data (the project's first iteration used XOR/Gorilla-style
delta compression and deliberately moved away from it — see [§7](#7-design-decisions-and-why)), every chunk carries
a small, cheap-to-check *zone map* (min/max bounds on symbol, valid time,
price, and transaction time). A query prunes chunks it can *prove* are
irrelevant using only that zone map, and only decompresses the survivors. The
system's correctness bar, enforced throughout by tests, is that this pruning
must never produce a false negative — it may keep a chunk that turns out not
to match, but it must never discard one that does.

## 2. Component overview

```mermaid
graph TD
    subgraph Ingestion
        RH[ReplayHarness<br/>sequential-only, no lookahead]
        DL[data_loader<br/>CSV to validated DataFrame]
    end

    subgraph Storage["Store (src/store/)"]
        CB[ChunkBuilder<br/>accumulates rows per symbol]
        CBD[ChunkBoundary<br/>Monthly / Duration]
        CH[Chunk<br/>raw column bytes + label]
        CL[ChunkLabel<br/>zone map: symbol, valid time,<br/>price, transaction time]
        CS[ChunkStore<br/>orchestration, indexes, compaction]
    end

    subgraph Query["Query engine (src/query/)"]
        SP[sql_parser<br/>sqlglot to PredicateNode]
        PR[predicate<br/>Comparison / And / In / AlwaysTrue]
        PD[pushdown<br/>evaluate_pushdown]
        VS[versioning<br/>resolve_versions]
        QE[QueryExecutor]
    end

    FS[FullScanQueryEngine<br/>naive baseline, src/baselines/]

    DL --> RH --> CS
    CS --> CB --> CBD
    CB --> CH --> CL
    QE --> SP --> PR
    QE --> PD --> CS
    QE --> VS
    QE -.decompresses candidates.-> CH
    FS -.decompresses everything, no pruning.-> CH

    Bench[benchmarks/] -.compares.-> QE
    Bench -.compares.-> FS
    Bench --> Charts[charts/]
```

## 3. The storage model

### `ChunkLabel` — the zone map (`src/store/label.py`)

A frozen, immutable dataclass — the only thing pushdown pruning is allowed to
look at before deciding whether to decompress a chunk:

```python
@dataclass(frozen=True)
class ChunkLabel:
    symbol: str
    time_start: datetime          # valid-time zone map
    time_end: datetime
    min_price: float               # price zone map
    max_price: float
    row_count: int
    chunk_id: int
    transaction_time_start: datetime | None = None   # transaction-time zone map
    transaction_time_end: datetime | None = None      # None => base chunk
```

`can_contain_symbol` / `can_contain_time` / `can_contain_price` are pure,
conservative predicates: `False` is a *proof* of non-membership; `True` means
"maybe — decompress and check exactly." An all-NaN chunk's price range is
represented as `(+inf, -inf)`, which makes every finite/NaN comparison
correctly return `False` with no special-cased branch.

`transaction_time_start`/`_end` being `None` means "this is base data, learned
at its own valid time, one row at a time — derive transaction-time visibility
from `time_start`/`time_end` directly." A non-`None` pair marks a **correction
chunk**, whose rows were learned later than their own valid time. This single
nullable pair is what lets the exact same struct, and the exact same pruning
code path, serve both plain time-series storage and bitemporal correction
storage — see [§5](#5-bitemporal-versioning).

### `Chunk` — the raw payload (`src/store/chunk.py`)

Each value column (`open`, `high`, `low`, `close`, `volume`) is stored as a
contiguous run of raw `float64` bytes, back to back, with a byte-length table
for re-slicing (`column_arrays()`). `decompress()` reassembles a full
`pandas.DataFrame` when a query actually needs one; `column_arrays()` skips
that construction entirely and hands back plain `numpy` arrays for callers
(like compaction) that only need the underlying values, not a DataFrame — a
real, measured performance difference (see
[methodology.md §4](methodology.md#4-the-first-audit-finding-correctness-and-performance-bugs-with-evidence-not-guesses)).

A correction chunk additionally carries `correction_transaction_times: list[datetime]`,
positionally aligned with its rows — required once a single chunk can hold
corrections learned at different times (after compaction merges several).

### `ChunkBuilder` / `ChunkBoundary` — accumulation and framing

`ChunkBuilder` accumulates one symbol's rows until `ChunkBoundary.crosses()`
says the window is full (`MonthlyBoundary` for daily bars, `DurationBoundary`
for intraday data), then finalizes a `Chunk` + `ChunkLabel` pair. The
**non-decreasing valid-timestamp check** here is the project's core
no-lookahead invariant: a builder physically cannot accept a row that goes
backward in time relative to what it has already seen.

### `ChunkStore` — orchestration (`src/store/chunk_store.py`)

Owns one `ChunkBuilder` per symbol, a `chunk_id -> Chunk` dict for O(1) lookup,
and (new) a **sorted transaction-time index** — two parallel lists,
`_as_of_keys`/`_as_of_labels`, kept sorted via `bisect` — that turns "which
chunks might be visible as of time T" from an O(n) linear scan into an O(log n)
bisect plus a cheap slice. Base ingestion (`ingest_row` /
`ingest_from_replay`) and correction ingestion (`ingest_correction`) are
**deliberately separate code paths** that never touch each other's state —
see [§5](#5-bitemporal-versioning).

## 4. The query engine

```mermaid
sequenceDiagram
    participant U as Caller
    participant QE as QueryExecutor
    participant SP as sql_parser
    participant PD as pushdown
    participant CS as ChunkStore
    participant CH as Chunk(s)
    participant VS as versioning

    U->>QE: execute(sql)
    QE->>SP: extract_as_of(sql)
    SP-->>QE: (remaining_sql, as_of)
    QE->>SP: parse_where_clause(remaining_sql)
    SP-->>QE: PredicateNode tree
    QE->>PD: evaluate_pushdown(predicate, labels, as_of, as_of_index=store)
    PD->>CS: labels_visible_as_of(as_of)  [O(log n) bisect]
    CS-->>PD: candidate labels
    PD-->>QE: PushdownResult (candidates, skipped)
    QE->>CH: decompress() each candidate
    QE->>VS: attach_effective_transaction_time + resolve_versions
    VS-->>QE: latest-wins-per-(symbol,timestamp) rows
    QE-->>U: QueryResult
```

1. **`sql_parser.py`** — a hand-rolled, bounded SQL subset built on `sqlglot`.
   `parse_where_clause` lowers the WHERE clause into the project's own IR
   (below); `extract_as_of` separately locates a `TIMESTAMP AS OF '<ts>'`
   clause (the same syntax and placement Delta Lake and SQL Server use) and
   strips it before the WHERE clause is parsed, so the two concerns never
   interfere with each other. It's tokenizer-based, not a full parse: raw
   lexical analysis only, anchored to the exact token position immediately
   after the table reference, matching the clause's own documented placement.
   Comments are discarded before tokenization ever produces output and a
   string literal — however it's quoted or escaped — always comes through as
   one opaque token, so neither can be mistaken for the keyword sequence
   being matched; an earlier regex-based version could be defeated by a SQL
   comment containing the word "where" (see
   [methodology.md §5](methodology.md#5-a-second-audit-performance-loopholes-and-benchmark-methodology-honesty)).
   Unsupported SQL (subqueries, joins, `OR`, aggregates) is rejected
   explicitly rather than silently mishandled.

2. **`predicate.py`** — `PredicateNode` (`Comparison` / `And` / `In` /
   `AlwaysTrue`) implements three parallel evaluation modes that must always
   agree: `evaluate_against_label` (conservative, chunk-level),
   `evaluate_against_row` (exact, one row at a time — used by the naive
   baseline), and `evaluate_vectorized` (exact, one pandas boolean mask —
   used by the real query path). `_align_literal_to_reference` and
   `_coerce_row_operands`/`_coerce_column_literal` centralize the
   timezone-awareness alignment every comparison needs; keeping this logic in
   one place is what closed two real bugs a correctness audit found earlier
   in the project (see [methodology.md](methodology.md)).
   `evaluate_against_label` also takes `price_pruning_enabled: bool = True`:
   once a store has any correction chunk (`ChunkStore.has_corrections`),
   price-based pruning is disabled so a correction's own value can never
   hide it from candidacy — symbol and time pruning stay enabled either way.

3. **`pushdown.py`** — `evaluate_pushdown` partitions chunk IDs into
   candidates and provably-skippable chunks. Its contract:
   `evaluate_against_label` returning `False` **must** be a proof, never a
   guess. The `as_of_index` parameter is a `Protocol`-typed optional fast
   path (structurally satisfied by `ChunkStore`, not imported by name — this
   module still depends only on `list[ChunkLabel]`) that swaps the O(n)
   per-label scan for an O(log n) index lookup when both `as_of` and the
   index are supplied — by consulting the index as an O(1) membership test
   while still iterating labels in their original order, so results are
   identical (content *and* order) to the plain linear scan below regardless
   of which path ran; omitting `as_of_index` is byte-for-byte identical to
   the original linear scan, so every existing caller is unaffected.

4. **`QueryExecutor`** — parses, prunes, decompresses only the survivors,
   applies the vectorized predicate mask once over the concatenated frame
   (not once per chunk — a deliberate optimization, see inline comment in
   `executor.py`), and — only when an `AS OF` cutoff or a correction chunk is
   actually in play — resolves per-`(symbol, timestamp)` duplicates to the
   newest version visible as of that cutoff.

5. **`FullScanQueryEngine`** (`src/baselines/full_scan.py`) — the naive
   baseline: decompresses *every* chunk, never consults a label, applies the
   same predicate row-by-row. Every benchmark and most tests assert full
   row/value/dtype equality between this and `QueryExecutor`'s output before
   ever reporting a speedup number — pushdown pruning that returns the wrong
   rows faster is not a feature.

## 5. Bitemporal versioning

### The problem

Market data gets revised — busted trades, corrected prints, restated bars.
`ChunkBuilder._validate_row` rejects any out-of-order valid timestamp the same
way it would reject a bug, because the store had exactly one timeline. A
correction has no way to exist without either mutating history (silently
invalidating anything that already queried it) or being rejected outright.

### The model

```mermaid
graph LR
    subgraph "Valid time (existing)"
        VT["time_start / time_end<br/>when the event happened"]
    end
    subgraph "Transaction time (new)"
        TT["transaction_time_start / _end<br/>when the system learned it"]
    end
    VT -.orthogonal axis.-> TT
    Q["AS OF query"] -->|"asks: what was known<br/>at transaction time T?"| TT
```

Grounded in Snodgrass's 1985 taxonomy of database time (the paper that coined
the valid-time/transaction-time split and the `AS OF` clause itself, via
TQuel) and the same model TDSQL, Delta Lake, Apache Hudi, and XTDB all
converge on independently. `time_start`/`time_end` already existed for valid
time; `transaction_time_start`/`_end` is the new, orthogonal axis.

### Ingestion: two separate paths, deliberately

`ChunkStore.ingest_correction(timestamp, symbol, values, transaction_time)` is
**not** a variant of `ingest_row` — it never touches `self._builders`, so the
live per-symbol builder's non-decreasing causality contract is completely
undisturbed. It builds a throwaway single-row `ChunkBuilder` (reusing the
exact same validation, price-zone computation, and byte-packing logic
finalized chunks use — no parallel implementation to silently diverge from
it), tags the resulting label with the caller-supplied transaction time, and
appends it. The one causality rule enforced: `transaction_time >= timestamp`
— you cannot learn about an event before it happens.

### Resolution: latest-transaction-time-wins (`src/query/versioning.py`)

```python
def attach_effective_transaction_time(frame, chunk) -> pd.DataFrame:
    # base chunk: each row's own timestamp is its transaction time
    # correction chunk: the label's per-row correction_transaction_times
    ...

def resolve_versions(df, txn_col="_txn_time") -> pd.DataFrame:
    # group by (symbol, timestamp); keep the row with the max txn_col
```

This subsumes the recommendation's original "supersedes pointer" idea without
needing one — it's strictly more general (handles overlapping/partial
corrections that don't align to prior chunk boundaries) and requires no
bookkeeping. **A validating property**: with zero corrections ever ingested,
`AS OF T` is provably identical to a plain `timestamp <= T` filter — the
bitemporal machinery collapses to a no-op, and this equivalence is asserted
directly in `tests/test_bitemporal.py`.

### Compaction (`ChunkStore.compact_corrections`)

Each `ingest_correction` call produces its own single-row chunk — cheap to
write, but a store with heavy correction volume accumulates many tiny chunks.
`compact_corrections()` merges same-symbol correction chunks into fewer,
larger ones, mirroring Delta Lake's `OPTIMIZE`, Apache Hudi's compaction table
service, and TDSQL's asynchronous migration-to-historical-table: writes stay
cheap and never block, consolidation happens later via an explicit, idempotent
maintenance call. It replays merged rows through a fresh `ChunkBuilder` in
valid-timestamp order (required, since corrections can target any past
timestamp in any ingestion order), so a merge naturally respects the
configured `ChunkBoundary` — or an explicit coarser `boundary` override, since
correction chunks don't need base data's fine pruning granularity. Compaction
is provably safe: it only ever changes zone-map *granularity*, never the
per-row `correction_transaction_times` that row-level resolution actually
relies on — the same invariant every other zone-map field in this codebase
already leans on.

## 6. The transaction-time index

A one-dimensional threshold query (`transaction_time <= as_of`) only needs a
sorted list and `bisect`, not a two-dimensional structure like the R-tree
TDSQL uses for interval-overlap queries (`BETWEEN`) this project doesn't
support. `ChunkStore` maintains two parallel lists sorted by
`transaction_time_start` (or `time_start` for base chunks), updated on insert
(`bisect.bisect_right` + `list.insert`) and on batched removal (a single
filtering pass across both structures — removing many chunks per call, not
one at a time, was itself a real O(n²) bug found and fixed by profiling; see
[methodology.md](methodology.md)). `labels_visible_as_of`/`labels_not_visible_as_of`
answer "which labels are/aren't visible as of T" in O(log n) plus a cheap
slice.

## 7. Design decisions and why

| Decision | Rationale |
|---|---|
| Raw `float64` storage, not XOR/Gorilla compression | An earlier iteration of this project used delta/XOR compression; it was deliberately abandoned in favor of raw storage so predicate pushdown could operate on uncompressed zone maps without a decode step. Compression ratio is intentionally ~1.0x — a documented trade-off, not an oversight. |
| `AS OF` via a nullable `transaction_time_start`/`_end` pair, not a separate correction table | Keeps one code path (`ChunkLabel`, one pushdown contract) instead of two parallel storage models to keep in sync. |
| `bisect`-based sorted index, not an R-tree | The system only needs a 1D threshold query today; an R-tree would be solving a 2D problem (interval overlap) this project doesn't have. |
| Corrections replay through `ChunkBuilder`, not a bespoke merge writer | Reuses the exact same validation, price-zone, and byte-packing logic finalized chunks already use — eliminates a whole class of "two implementations silently diverge" bugs. |
| `compact_corrections()` is explicit and idempotent, not automatic | Matches Delta Lake/Hudi/TDSQL: writes stay cheap and predictable; consolidation is a deliberate, separately-timed maintenance operation. |
| Chunk boundary granularity is user-configurable (`MonthlyBoundary`, `DurationBoundary`, or a coarser override for compaction) | A direct, measured trade-off exists between pruning precision and chunk-management overhead — see `benchmarks/bench_chunk_granularity.py` and the Q5 selectivity-crossover discussion in [methodology.md](methodology.md). |
| `extract_as_of` locates its clause via `sqlglot`'s tokenizer, not a full parse | A full re-parse under a different dialect (to natively recognize the syntax) once corrupted quoted identifiers and escaped-quote literals elsewhere in the same query; a later plain-text regex fix over the raw SQL could then be defeated by a SQL comment containing the word "where". Tokenizing is the narrowest tool that's actually immune to both failure modes — comments and string literals never survive as matchable text. |

## 8. Module reference

| Path | Responsibility |
|---|---|
| `src/replay/data_loader.py` | Load and validate a merged OHLCV CSV |
| `src/replay/replay_harness.py` | Strict sequential, no-lookahead row iterator — the sole ingestion-facing view of a DataFrame |
| `src/store/chunk_boundary.py` | `ChunkBoundary` (Monthly / Duration) — decides when a chunk window closes |
| `src/store/chunk_builder.py` | Accumulates one symbol's rows into a finalized `Chunk` |
| `src/store/chunk.py` | Raw column-byte storage, `decompress()`, `column_arrays()` |
| `src/store/label.py` | `ChunkLabel` — the immutable zone map |
| `src/store/chunk_store.py` | Ingestion orchestration, chunk indexing, compaction, transaction-time index |
| `src/query/schema.py` | Canonical column schema, SQL aliases, typed literal parsing |
| `src/query/predicate.py` | Predicate-tree IR and its three evaluation modes |
| `src/query/sql_parser.py` | SQL → predicate tree; `AS OF` extraction |
| `src/query/pushdown.py` | Conservative chunk-candidate selection |
| `src/query/versioning.py` | Bitemporal row-level version resolution |
| `src/query/executor.py` | End-to-end query execution |
| `src/baselines/full_scan.py` | Naive baseline for correctness/speed comparison |
| `src/pipeline.py` | CLI: ingest a CSV, run queries, optionally benchmark |
| `benchmarks/` | The full benchmark suite — see [methodology.md](methodology.md) |
| `benchmarks/report.py` | Shared table rendering and number formatting every `bench_*.py` script uses for its final summary |
| `charts/` | Chart generation from benchmark JSON output |
