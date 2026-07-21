# PitDB — Consolidated Benchmark Results

Generated: 2026-07-21T15:44:39+00:00
Git commit: 543cfda
Environment: tuned (performance governor, turbo enabled, SMT on, swap on, background
apps at normal load, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=1 PYTHONHASHSEED=0`, no core pinning/isolation
used for these runs)

**Benchmark 3 (Scale Sweep) was excluded from this run by explicit request — not
executed.** Everything below is real, unedited terminal output from actually
running the other four benchmarks, each via its own direct invocation
(`benchmarks/run_all.py --memory`, `benchmarks/bench_bitemporal.py`,
`benchmarks/bench_chunk_granularity.py`), plus a suite-level summary built from
the resulting JSON files (not a separate re-run).

All three runs completed with exit code 0. Every correctness check in every
benchmark below is a real `PASSED`, not a placeholder.

---

## Suite Summary

```text
================================================================================
 PITDB BENCHMARK SUITE SUMMARY
================================================================================

Environment
  Git commit       : 543cfda
  Python           : 3.11.15
  Operating system : Linux 7.0.0-27-generic (x86_64)
  CPU              : x86_64 (12 logical / 10 physical cores)
  Random seed      : seed=1 (synthetic benchmarks 4-5, data generation); N/A (real market data, benchmarks 1-2)
  Note             : Benchmark 3 (scale sweep) excluded from this run by request -- not executed.

┌─────┬───────────────────────────────┬─────────────┬───────────────────────────────────────┐
│ No. │ Benchmark                     │ Correctness │ Primary result                        │
├─────┼───────────────────────────────┼─────────────┼───────────────────────────────────────┤
│ 1   │ Fixed queries and selectivity │ PASSED      │ Max speedup: 111.41x                  │
│ 2   │ Memory footprint              │ PASSED      │ Reduction: 63.14%                     │
│ 3   │ Scale sweep                   │ N/A         │ Excluded from this run (not executed) │
│ 4   │ Corrections and compaction    │ PASSED      │ AS OF speedup: 4.20x                  │
│ 5   │ Chunk-granularity trade-off   │ PASSED      │ Default: 1 calendar month             │
└─────┴───────────────────────────────┴─────────────┴───────────────────────────────────────┘

Validation
  Benchmarks passed        : 4/4
  Benchmarks excluded      : 1
  Correctness failures     : 0
  Performance claims valid : YES

Generated artifacts
  CSV results    : N/A (this suite exports JSON only)
  JSON results   : /home/abc/Desktop/project/benchmarks/results
  Charts         : /home/abc/Desktop/project/charts/output (not regenerated in this run)
  Plain-text log : See benchmarks_consolidated.md -- per-benchmark sections above

Final status
  ALL RUN BENCHMARKS COMPLETED SUCCESSFULLY (3/5 executed, 1 excluded by request)

================================================================================
```

---

## Benchmarks 1 & 2 — Fixed Queries, Selectivity Sweep, and Memory Footprint

Command: `benchmarks/run_all.py --memory`

```text
================================================================================
 PITDB BENCHMARK 1/5
 Fixed Queries and Selectivity Sweep
================================================================================

Question
  How does zone-map pushdown performance change as query selectivity narrows?

Configuration
  Dataset             : all_symbols_daily.csv
  Symbols             : 10
  Rows                : 5,010
  Chunks              : 240
  Chunk granularity   : monthly
  Warmup runs         : 3
  Measured runs       : 15
  Timing statistic    : outlier-trimmed mean
  Confidence interval : 95%
  Random seed         : N/A (real market data, no synthetic generation)

--------------------------------------------------------------------------------
[SETUP] Loading dataset
--------------------------------------------------------------------------------
  Status           : COMPLETE
  Rows loaded      : 5,010
  Store build time : 715.931 ms
  Total chunks     : 240

--------------------------------------------------------------------------------
[FIXED QUERIES]
--------------------------------------------------------------------------------

Query 1/5 — Single symbol, single day (Q1)

  Predicate
    Symbol           : AAPL
    Valid-time range : 2024-03-01 (single day)
    Price predicate  : none
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'

  Correctness validation
    Full-scan rows : 1
    Pushdown rows  : 1
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 240
    Chunks scanned     : 1
    Chunks pruned      : 239
    Pruning rate       : 99.58%
    Rows returned      : 1
    Actual selectivity : 0.02%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 53.159ms     │ 51.437 – 54.882 ms      │
    │ Zone-map pushdown │ 4.251ms      │ 4.202 – 4.300 ms        │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 12.51x

  Interpretation
    Observation : High pruning was achieved for this narrow query.
    Evidence    : 239/240 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

Query 2/5 — Single symbol, one-month range (Q2)

  Predicate
    Symbol           : AAPL
    Valid-time range : 2024-03-01 → 2024-03-31
    Price predicate  : none
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE symbol = 'AAPL' AND date >= '2024-03-01' AND date < '2024-04-01'

  Correctness validation
    Full-scan rows : 20
    Pushdown rows  : 20
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 240
    Chunks scanned     : 1
    Chunks pruned      : 239
    Pruning rate       : 99.58%
    Rows returned      : 20
    Actual selectivity : 0.40%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 140.649ms    │ 113.618 – 167.680 ms    │
    │ Zone-map pushdown │ 1.262ms      │ 1.239 – 1.285 ms        │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 111.41x

  Interpretation
    Observation : High pruning was achieved for this narrow query.
    Evidence    : 239/240 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

Query 3/5 — Multi-symbol, one-week range (Q3)

  Predicate
    Symbol           : AAPL, MSFT
    Valid-time range : 2024-03-01 → 2024-03-07
    Price predicate  : none
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE symbol IN ('AAPL', 'MSFT') AND date >= '2024-03-01' AND date < '2024-03-08'

  Correctness validation
    Full-scan rows : 10
    Pushdown rows  : 10
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 240
    Chunks scanned     : 2
    Chunks pruned      : 238
    Pruning rate       : 99.17%
    Rows returned      : 10
    Actual selectivity : 0.20%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 106.657ms    │ 71.124 – 142.189 ms     │
    │ Zone-map pushdown │ 5.719ms      │ 5.650 – 5.788 ms        │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 18.65x

  Interpretation
    Observation : High pruning was achieved for this narrow query.
    Evidence    : 238/240 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

Query 4/5 — Single symbol, price + date range (Q4)

  Predicate
    Symbol           : AAPL
    Valid-time range : 2024-01-01 → (open-ended)
    Price predicate  : > 200
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE symbol = 'AAPL' AND price > 200 AND date >= '2024-01-01'

  Correctness validation
    Full-scan rows : 140
    Pushdown rows  : 140
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 240
    Chunks scanned     : 7
    Chunks pruned      : 233
    Pruning rate       : 97.08%
    Rows returned      : 140
    Actual selectivity : 2.79%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 97.069ms     │ 66.009 – 128.129 ms     │
    │ Zone-map pushdown │ 2.315ms      │ 2.281 – 2.348 ms        │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 41.94x

  Interpretation
    Observation : High pruning was achieved for this narrow query.
    Evidence    : 233/240 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

Query 5/5 — Broad query (~50% of the table) (Q5)

  Predicate
    Symbol           : (all symbols)
    Valid-time range : 2024-01-01 → (open-ended)
    Price predicate  : none
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE date >= '2024-01-01'

  Correctness validation
    Full-scan rows : 2,510
    Pushdown rows  : 2,510
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 240
    Chunks scanned     : 120
    Chunks pruned      : 120
    Pruning rate       : 50.00%
    Rows returned      : 2,510
    Actual selectivity : 50.10%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 58.185ms     │ 53.925 – 62.446 ms      │
    │ Zone-map pushdown │ 18.853ms     │ 18.703 – 19.004 ms      │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 3.09x

  Interpretation
    Observation : Moderate pruning was achieved for this query.
    Evidence    : 120/240 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
[SELECTIVITY SWEEP]
--------------------------------------------------------------------------------

┌─────────────┬─────────────┬──────────────┬───────────┬──────────┬─────────┐
│ Target sel. │ Actual sel. │ Pruning rate │ Full scan │ Pushdown │ Speedup │
├─────────────┼─────────────┼──────────────┼───────────┼──────────┼─────────┤
│ 100.00%     │ 100.00%     │ 0.00%        │ 53.468ms  │ 37.758ms │ 1.42x   │
│ 50.00%      │ 50.10%      │ 50.00%       │ 54.100ms  │ 18.702ms │ 2.89x   │
│ 10.00%      │ 9.78%       │ 87.50%       │ 54.268ms  │ 5.400ms  │ 10.05x  │
│ 1.00%       │ 1.00%       │ 95.83%       │ 54.131ms  │ 2.438ms  │ 22.21x  │
│ 0.10%       │ 0.20%       │ 95.83%       │ 98.999ms  │ 2.464ms  │ 40.18x  │
└─────────────┴─────────────┴──────────────┴───────────┴──────────┴─────────┘

--------------------------------------------------------------------------------
[PRICE-PREDICATE CONTRIBUTION]
--------------------------------------------------------------------------------

┌───────────────────────┬───────────────┬──────────────┬───────────────┐
│ Predicate             │ Chunks pruned │ Pruning rate │ Speedup       │
├───────────────────────┼───────────────┼──────────────┼───────────────┤
│ Symbol + date         │ 228/240       │ 95.00%       │ N/A (untimed) │
│ Symbol + date + price │ 233/240       │ 97.08%       │ N/A (untimed) │
└───────────────────────┴───────────────┴──────────────┴───────────────┘

--------------------------------------------------------------------------------
[SETUP] Loading dataset (Dataset B, intraday)
--------------------------------------------------------------------------------
  Status           : COMPLETE
  Dataset          : all_symbols_1m.csv
  Symbols          : 10
  Rows loaded      : 16,739
  Store build time : 3368.591 ms
  Total chunks     : 300

--------------------------------------------------------------------------------
[FIXED QUERIES — DATASET B (INTRADAY)]
--------------------------------------------------------------------------------

Query 1/3 — narrow_symbol_hour (narrow_symbol_hour)

  Predicate
    Symbol           : AAPL
    Valid-time range : 2026-07-10 (single day)
    Price predicate  : none
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE symbol = 'AAPL' AND date >= '2026-07-10 13:30:00+00:00' AND date <= '2026-07-10 14:29:00+00:00'

  Correctness validation
    Full-scan rows : 60
    Pushdown rows  : 60
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 300
    Chunks scanned     : 1
    Chunks pruned      : 299
    Pruning rate       : 99.67%
    Rows returned      : 60
    Actual selectivity : 0.36%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 138.752ms    │ 107.266 – 170.238 ms    │
    │ Zone-map pushdown │ 1.357ms      │ 1.334 – 1.380 ms        │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 102.26x

  Interpretation
    Observation : High pruning was achieved for this narrow query.
    Evidence    : 299/300 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

Query 2/3 — price_threshold (price_threshold)

  Predicate
    Symbol           : AAPL
    Valid-time range : (unconstrained)
    Price predicate  : > 321.735
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE symbol = 'AAPL' AND price > 321.7350

  Correctness validation
    Full-scan rows : 486
    Pushdown rows  : 486
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 300
    Chunks scanned     : 10
    Chunks pruned      : 290
    Pruning rate       : 96.67%
    Rows returned      : 486
    Actual selectivity : 2.90%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 245.243ms    │ 181.605 – 308.880 ms    │
    │ Zone-map pushdown │ 9.300ms      │ 9.256 – 9.344 ms        │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 26.37x

  Interpretation
    Observation : High pruning was achieved for this narrow query.
    Evidence    : 290/300 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

Query 3/3 — broad_full_range (broad_full_range)

  Predicate
    Symbol           : (all symbols)
    Valid-time range : 2026-07-10 → 2026-07-16
    Price predicate  : none
    AS OF cutoff     : current state
    Raw SQL          : SELECT * FROM data WHERE date >= '2026-07-10 13:30:00+00:00' AND date <= '2026-07-16 15:24:00+00:00'

  Correctness validation
    Full-scan rows : 16,739
    Pushdown rows  : 16,739
    Row equality   : PASS
    Value equality : PASS
    Dtype equality : PASS
    Overall status : PASSED

  Execution statistics
    Total chunks       : 300
    Chunks scanned     : 300
    Chunks pruned      : 0
    Pruning rate       : 0.00%
    Rows returned      : 16,739
    Actual selectivity : 100.00%

  Timing results
    ┌───────────────────┬──────────────┬─────────────────────────┐
    │ Engine            │ Mean latency │ 95% confidence interval │
    ├───────────────────┼──────────────┼─────────────────────────┤
    │ Full scan         │ 480.258ms    │ 473.735 – 486.781 ms    │
    │ Zone-map pushdown │ 145.245ms    │ 116.936 – 173.555 ms    │
    └───────────────────┴──────────────┴─────────────────────────┘

  Speedup
    Full scan / pushdown : 3.31x

  Interpretation
    Observation : Low pruning was achieved; this query needs most of the table, so a small speedup is expected here, not a regression.
    Evidence    : 0/300 chunks were pruned by the zone-map index.
    Limitation  : Result applies to this predicate and dataset only.

--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
[RESULT]
--------------------------------------------------------------------------------
  Correctness             : PASSED
  Narrowest-query speedup : 111.41x (Q2)
  Broadest-query speedup  : 3.09x (Q5)
  Main observation        : Speedup increased as query selectivity narrowed (1.42x at 100.00% selectivity -> 40.18x at 0.20% selectivity).
  Result artifact         : /home/abc/Desktop/project/benchmarks/results/queries.json

================================================================================
 BENCHMARK 1 COMPLETE
================================================================================

================================================================================
 PITDB BENCHMARK 2/5
 Memory-Footprint Analysis
================================================================================

Question
  How memory-efficient is PitDB's uncompressed columnar storage compared with
  equivalent in-memory representations?

Configuration
  Rows                   : 5,010
  Symbols                : 10
  Chunks                 : 240
  Numeric representation : float64
  Chunk granularity      : monthly
  Query repetitions      : 1 (RSS sampled continuously during a single execution)

--------------------------------------------------------------------------------
[SETUP]
--------------------------------------------------------------------------------
  Dataset generated     : YES
  Store constructed     : YES
  Memory profiler ready : YES

--------------------------------------------------------------------------------
[CORRECTNESS VALIDATION]
--------------------------------------------------------------------------------
  Full-scan rows : 1
  Pushdown rows  : 1
  Row equality   : PASS
  Value equality : PASS
  Dtype equality : PASS
  Overall status : PASSED

--------------------------------------------------------------------------------
[STORAGE REPRESENTATION]
 Directly comparable storage measurements
--------------------------------------------------------------------------------

┌─────────────────────────────┬──────────────┬───────────────┐
│ Representation              │ Total memory │ Bytes per row │
├─────────────────────────────┼──────────────┼───────────────┤
│ Raw DataFrame columns       │ 530.97 KiB   │ 108.5         │
│ PitDB packed column storage │ 195.70 KiB   │ 40.0          │
└─────────────────────────────┴──────────────┴───────────────┘

  Packed-storage ratio
    PitDB / raw DataFrame : 0.37x
    Storage reduction     : 63.14%

  Comparison status
    Valid comparison : YES
    Reason           : Both measure raw column storage.

--------------------------------------------------------------------------------
[INTERNAL OBJECT FOOTPRINT]
 Not directly comparable with packed column bytes
--------------------------------------------------------------------------------

┌─────────────────────────────┬────────────┐
│ Component                   │ Memory     │
├─────────────────────────────┼────────────┤
│ Packed numeric buffers      │ 203.44 KiB │
│ Chunk metadata              │ 20.22 KiB  │
│ Zone maps                   │ 101.25 KiB │
│ Labels and indexes          │ 980.28 KiB │
│ Python object overhead      │ 1.67 KiB   │
│ Complete PitDB object graph │ 1.28 MiB   │
└─────────────────────────────┴────────────┘

  Important
    The complete object-graph footprint includes Python bookkeeping and must
    not be interpreted as equivalent to packed persisted column bytes.

--------------------------------------------------------------------------------
[PEAK PROCESS MEMORY]
--------------------------------------------------------------------------------

┌────────────────────┬────────────┬────────────────────┐
│ Operation          │ Peak RSS   │ Increase from idle │
├────────────────────┼────────────┼────────────────────┤
│ Process idle       │ 100.30 MiB │ —                  │
│ Store construction │ 100.44 MiB │ 148.00 KiB         │
│ Full-scan query    │ 100.44 MiB │ 148.00 KiB         │
│ Pushdown query     │ 100.44 MiB │ 148.00 KiB         │
└────────────────────┴────────────┴────────────────────┘

--------------------------------------------------------------------------------
[RESULT]
--------------------------------------------------------------------------------
  Correctness           : PASSED
  Packed storage        : 195.70 KiB
  Raw DataFrame storage : 530.97 KiB
  Storage reduction     : 63.14%
  Full object footprint : 1.28 MiB
  Peak query RSS        : 100.44 MiB

  Interpretation
    Observation : Packed storage uses 63.14% less memory than the equivalent raw DataFrame representation.
    Valid claim : Packed storage is compared only with raw table storage.
    Limitation  : Python object overhead is reported separately.

  Result artifact : /home/abc/Desktop/project/benchmarks/results/memory.json

================================================================================
 BENCHMARK 2 COMPLETE
================================================================================

[EXPORT]  Result files
  queries      /home/abc/Desktop/project/benchmarks/results/queries.json
  hardware     /home/abc/Desktop/project/benchmarks/results/hardware.json
  memory       /home/abc/Desktop/project/benchmarks/results/memory.json
EXIT_CODE=0
```

---

## Benchmark 4 — Bitemporal Corrections and Compaction

Command: `benchmarks/bench_bitemporal.py`

```text
================================================================================
 PITDB BENCHMARK 4/5
 Bitemporal Corrections and Compaction
================================================================================

Question
  How do correction volume and compaction affect current-state and historical
  AS OF query performance?

Configuration
  Base rows               : 10,020
  Correction levels       : 5
  Historical AS OF cutoff : 2024-12-03T00:00:00+00:00
  Compaction policy       : monthly (same granularity as base store)
  Warmup runs             : 2
  Measured runs           : 15

--------------------------------------------------------------------------------
[CORRECTION LEVEL 1/5]
--------------------------------------------------------------------------------

Correction workload
  Corrections              : 0
  Corrected-row percentage : 0.00%
  Ingestion time           : 0.000 ms
  Ingestion throughput     : N/A (no corrections at this level)

Storage state before compaction
  Total chunks      : 480
  Base chunks       : 480
  Correction chunks : 0

Correctness validation
  Current-state query    : PASSED
  Historical AS OF query : N/A (no corrections exist yet)

Performance matrix

┌──────────────────┬───────────────────┬───────────┬──────────┬─────────┐
│ Query            │ Storage state     │ Full scan │ Pushdown │ Speedup │
├──────────────────┼───────────────────┼───────────┼──────────┼─────────┤
│ Current state    │ Before compaction │ 110.100ms │ 70.952ms │ 1.55x   │
│ Historical AS OF │ Before compaction │ N/A       │ N/A      │ N/A     │
└──────────────────┴───────────────────┴───────────┴──────────┴─────────┘

Pruning before compaction

┌──────────────────┬──────────────┬───────────────┬──────────────┐
│ Query            │ Total chunks │ Chunks pruned │ Pruning rate │
├──────────────────┼──────────────┼───────────────┼──────────────┤
│ Current state    │ 480          │ 0             │ 0.00%        │
│ Historical AS OF │ N/A          │ N/A           │ N/A          │
└──────────────────┴──────────────┴───────────────┴──────────────┘

--------------------------------------------------------------------------------
[COMPACTION]
--------------------------------------------------------------------------------
  Input chunks          : 480
  Output chunks         : 480
  Chunks merged         : 0
  Compaction runtime    : 0.038 ms
  Processing throughput : N/A (nothing to compact)
  Status                : COMPLETE

Storage state after compaction
  Total chunks      : 480
  Base chunks       : 480
  Correction chunks : 0

Correctness revalidation
  Current-state query    : PASSED
  Historical AS OF query : N/A (no corrections exist yet)

Performance matrix

┌──────────────────┬──────────────────┬───────────┬──────────┬─────────┐
│ Query            │ Storage state    │ Full scan │ Pushdown │ Speedup │
├──────────────────┼──────────────────┼───────────┼──────────┼─────────┤
│ Current state    │ After compaction │ 109.646ms │ 72.518ms │ 1.51x   │
│ Historical AS OF │ After compaction │ N/A       │ N/A      │ N/A     │
└──────────────────┴──────────────────┴───────────┴──────────┴─────────┘

--------------------------------------------------------------------------------
[CORRECTION LEVEL 2/5]
--------------------------------------------------------------------------------

Correction workload
  Corrections              : 100
  Corrected-row percentage : 1.00%
  Ingestion time           : 1.476 ms
  Ingestion throughput     : 67,749 corrections/second

Storage state before compaction
  Total chunks      : 580
  Base chunks       : 480
  Correction chunks : 100

Correctness validation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬───────────────────┬───────────┬───────────┬─────────┐
│ Query            │ Storage state     │ Full scan │ Pushdown  │ Speedup │
├──────────────────┼───────────────────┼───────────┼───────────┼─────────┤
│ Current state    │ Before compaction │ 588.108ms │ 454.992ms │ 1.29x   │
│ Historical AS OF │ Before compaction │ 322.360ms │ 582.805ms │ 0.55x   │
└──────────────────┴───────────────────┴───────────┴───────────┴─────────┘

Pruning before compaction

┌──────────────────┬──────────────┬───────────────┬──────────────┐
│ Query            │ Total chunks │ Chunks pruned │ Pruning rate │
├──────────────────┼──────────────┼───────────────┼──────────────┤
│ Current state    │ 580          │ 0             │ 0.00%        │
│ Historical AS OF │ 580          │ 100           │ 17.24%       │
└──────────────────┴──────────────┴───────────────┴──────────────┘

--------------------------------------------------------------------------------
[COMPACTION]
--------------------------------------------------------------------------------
  Input chunks          : 580
  Output chunks         : 572
  Chunks merged         : 8
  Compaction runtime    : 3.197 ms
  Processing throughput : 31,282 rows/second
  Status                : COMPLETE

Storage state after compaction
  Total chunks      : 572
  Base chunks       : 480
  Correction chunks : 92

Correctness revalidation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬──────────────────┬───────────┬───────────┬─────────┐
│ Query            │ Storage state    │ Full scan │ Pushdown  │ Speedup │
├──────────────────┼──────────────────┼───────────┼───────────┼─────────┤
│ Current state    │ After compaction │ 247.629ms │ 209.986ms │ 1.18x   │
│ Historical AS OF │ After compaction │ 342.354ms │ 176.545ms │ 1.94x   │
└──────────────────┴──────────────────┴───────────┴───────────┴─────────┘

--------------------------------------------------------------------------------
[CORRECTION LEVEL 3/5]
--------------------------------------------------------------------------------

Correction workload
  Corrections              : 501
  Corrected-row percentage : 5.00%
  Ingestion time           : 24.567 ms
  Ingestion throughput     : 20,393 corrections/second

Storage state before compaction
  Total chunks      : 981
  Base chunks       : 480
  Correction chunks : 501

Correctness validation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬───────────────────┬────────────┬────────────┬─────────┐
│ Query            │ Storage state     │ Full scan  │ Pushdown   │ Speedup │
├──────────────────┼───────────────────┼────────────┼────────────┼─────────┤
│ Current state    │ Before compaction │ 1303.788ms │ 1188.016ms │ 1.10x   │
│ Historical AS OF │ Before compaction │ 1306.196ms │ 596.168ms  │ 2.19x   │
└──────────────────┴───────────────────┴────────────┴────────────┴─────────┘

Pruning before compaction

┌──────────────────┬──────────────┬───────────────┬──────────────┐
│ Query            │ Total chunks │ Chunks pruned │ Pruning rate │
├──────────────────┼──────────────┼───────────────┼──────────────┤
│ Current state    │ 981          │ 0             │ 0.00%        │
│ Historical AS OF │ 981          │ 501           │ 51.07%       │
└──────────────────┴──────────────┴───────────────┴──────────────┘

--------------------------------------------------------------------------------
[COMPACTION]
--------------------------------------------------------------------------------
  Input chunks          : 981
  Output chunks         : 772
  Chunks merged         : 209
  Compaction runtime    : 30.864 ms
  Processing throughput : 16,232 rows/second
  Status                : COMPLETE

Storage state after compaction
  Total chunks      : 772
  Base chunks       : 480
  Correction chunks : 292

Correctness revalidation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬──────────────────┬────────────┬───────────┬─────────┐
│ Query            │ Storage state    │ Full scan  │ Pushdown  │ Speedup │
├──────────────────┼──────────────────┼────────────┼───────────┼─────────┤
│ Current state    │ After compaction │ 1061.197ms │ 939.410ms │ 1.13x   │
│ Historical AS OF │ After compaction │ 1065.589ms │ 592.621ms │ 1.80x   │
└──────────────────┴──────────────────┴────────────┴───────────┴─────────┘

--------------------------------------------------------------------------------
[CORRECTION LEVEL 4/5]
--------------------------------------------------------------------------------

Correction workload
  Corrections              : 2,004
  Corrected-row percentage : 20.00%
  Ingestion time           : 96.207 ms
  Ingestion throughput     : 20,830 corrections/second

Storage state before compaction
  Total chunks      : 2,484
  Base chunks       : 480
  Correction chunks : 2,004

Correctness validation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬───────────────────┬────────────┬───────────┬─────────┐
│ Query            │ Storage state     │ Full scan  │ Pushdown  │ Speedup │
├──────────────────┼───────────────────┼────────────┼───────────┼─────────┤
│ Current state    │ Before compaction │ 1007.468ms │ 907.859ms │ 1.11x   │
│ Historical AS OF │ Before compaction │ 1526.618ms │ 178.520ms │ 8.55x   │
└──────────────────┴───────────────────┴────────────┴───────────┴─────────┘

Pruning before compaction

┌──────────────────┬──────────────┬───────────────┬──────────────┐
│ Query            │ Total chunks │ Chunks pruned │ Pruning rate │
├──────────────────┼──────────────┼───────────────┼──────────────┤
│ Current state    │ 2,484        │ 0             │ 0.00%        │
│ Historical AS OF │ 2,484        │ 2,004         │ 80.68%       │
└──────────────────┴──────────────┴───────────────┴──────────────┘

--------------------------------------------------------------------------------
[COMPACTION]
--------------------------------------------------------------------------------
  Input chunks          : 2,484
  Output chunks         : 944
  Chunks merged         : 1,540
  Compaction runtime    : 24.488 ms
  Processing throughput : 81,835 rows/second
  Status                : COMPLETE

Storage state after compaction
  Total chunks      : 944
  Base chunks       : 480
  Correction chunks : 464

Correctness revalidation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬──────────────────┬───────────┬───────────┬─────────┐
│ Query            │ Storage state    │ Full scan │ Pushdown  │ Speedup │
├──────────────────┼──────────────────┼───────────┼───────────┼─────────┤
│ Current state    │ After compaction │ 403.506ms │ 360.437ms │ 1.12x   │
│ Historical AS OF │ After compaction │ 434.184ms │ 186.023ms │ 2.33x   │
└──────────────────┴──────────────────┴───────────┴───────────┴─────────┘

--------------------------------------------------------------------------------
[CORRECTION LEVEL 5/5]
--------------------------------------------------------------------------------

Correction workload
  Corrections              : 5,010
  Corrected-row percentage : 50.00%
  Ingestion time           : 80.034 ms
  Ingestion throughput     : 62,598 corrections/second

Storage state before compaction
  Total chunks      : 5,490
  Base chunks       : 480
  Correction chunks : 5,010

Correctness validation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬───────────────────┬────────────┬────────────┬─────────┐
│ Query            │ Storage state     │ Full scan  │ Pushdown   │ Speedup │
├──────────────────┼───────────────────┼────────────┼────────────┼─────────┤
│ Current state    │ Before compaction │ 2258.909ms │ 2424.460ms │ 0.93x   │
│ Historical AS OF │ Before compaction │ 2500.189ms │ 177.380ms  │ 14.10x  │
└──────────────────┴───────────────────┴────────────┴────────────┴─────────┘

Pruning before compaction

┌──────────────────┬──────────────┬───────────────┬──────────────┐
│ Query            │ Total chunks │ Chunks pruned │ Pruning rate │
├──────────────────┼──────────────┼───────────────┼──────────────┤
│ Current state    │ 5,490        │ 0             │ 0.00%        │
│ Historical AS OF │ 5,490        │ 5,010         │ 91.26%       │
└──────────────────┴──────────────┴───────────────┴──────────────┘

--------------------------------------------------------------------------------
[COMPACTION]
--------------------------------------------------------------------------------
  Input chunks          : 5,490
  Output chunks         : 951
  Chunks merged         : 4,539
  Compaction runtime    : 54.715 ms
  Processing throughput : 91,566 rows/second
  Status                : COMPLETE

Storage state after compaction
  Total chunks      : 951
  Base chunks       : 480
  Correction chunks : 471

Correctness revalidation
  Current-state query    : PASSED
  Historical AS OF query : PASSED

Performance matrix

┌──────────────────┬──────────────────┬───────────┬───────────┬─────────┐
│ Query            │ Storage state    │ Full scan │ Pushdown  │ Speedup │
├──────────────────┼──────────────────┼───────────┼───────────┼─────────┤
│ Current state    │ After compaction │ 393.405ms │ 436.012ms │ 0.90x   │
│ Historical AS OF │ After compaction │ 751.506ms │ 179.026ms │ 4.20x   │
└──────────────────┴──────────────────┴───────────┴───────────┴─────────┘

--------------------------------------------------------------------------------
[CORRECTION SWEEP SUMMARY]
--------------------------------------------------------------------------------

┌─────────────┬─────────┬──────────────────────────────────────┬─────────────────────────────────────────┬────────────────────┐
│ Corrections │ Percent │ Current-state speedup (before/after) │ Historical AS OF speedup (before/after) │ Compaction runtime │
├─────────────┼─────────┼──────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
│ 0           │ 0.00%   │ 1.55x / 1.51x                        │ -- / --                                 │ 0.038 ms           │
│ 100         │ 1.00%   │ 1.29x / 1.18x                        │ 0.55x / 1.94x                           │ 3.197 ms           │
│ 501         │ 5.00%   │ 1.10x / 1.13x                        │ 2.19x / 1.80x                           │ 30.864 ms          │
│ 2,004       │ 20.00%  │ 1.11x / 1.12x                        │ 8.55x / 2.33x                           │ 24.488 ms          │
│ 5,010       │ 50.00%  │ 0.93x / 0.90x                        │ 14.10x / 4.20x                          │ 54.715 ms          │
└─────────────┴─────────┴──────────────────────────────────────┴─────────────────────────────────────────┴────────────────────┘

--------------------------------------------------------------------------------
[RESULT]
--------------------------------------------------------------------------------
  Correctness            : PASSED AT ALL CORRECTION LEVELS
  Current-state trend    : Pre-compaction speedup fell from 1.55x at 0% corrected to 0.93x at 50% corrected.
  Historical AS OF trend : Pre-compaction speedup grew from 0.55x at 1% corrected to 14.10x at 50% corrected.
  Compaction trend       : Runtime grew from 3.197 ms at 1% corrected to 54.715 ms at 50% corrected.
  Result artifact        : /home/abc/Desktop/project/benchmarks/results/bitemporal_sweep.json

  Interpretation
    Historical AS OF queries can reject correction chunks whose transaction
    times occur after the requested cutoff.

  Limitation
    Performance conclusions apply only to the tested correction distributions
    and compaction policy.

================================================================================
 BENCHMARK 4 COMPLETE
================================================================================
EXIT_CODE=0
```

---

## Benchmark 5 — Chunk-Granularity Trade-Off

Command: `benchmarks/bench_chunk_granularity.py`

```text
================================================================================
 PITDB BENCHMARK 5/5
 Chunk-Granularity Trade-Off
================================================================================

Question
  How does chunk granularity affect pruning precision, chunk overhead, and
  query performance?

Configuration
  Rows                 : 5,010
  Symbols              : 10
  Granularities tested : Weekly, Monthly, Quarterly
  Query                : date range + price range (SYN000003)
  Warmup runs          : 3
  Measured runs        : 15

--------------------------------------------------------------------------------
[GRANULARITY 1/3 — WEEKLY]
--------------------------------------------------------------------------------

Store layout
  Chunk window           : 7 days
  Total chunks           : 1,010
  Average rows per chunk : 5.0
  Minimum rows per chunk : 1
  Maximum rows per chunk : 5
  Metadata size          : 607.41 KiB

Correctness validation
  Full-scan rows : 11
  Pushdown rows  : 11
  Row equality   : PASS
  Value equality : PASS
  Dtype equality : PASS
  Overall status : PASSED

Pruning
  Chunks scanned : 3
  Chunks pruned  : 1,007
  Pruning rate   : 99.70%
  Rows scanned   : 15
  Rows returned  : 11

Timing
  ┌───────────────────┬──────────────┬─────────────────────────┐
  │ Engine            │ Mean latency │ 95% confidence interval │
  ├───────────────────┼──────────────┼─────────────────────────┤
  │ Full scan         │ 157.900ms    │ 153.479 – 162.321 ms    │
  │ Zone-map pushdown │ 2.420ms      │ 2.380 – 2.460 ms        │
  └───────────────────┴──────────────┴─────────────────────────┘

  Speedup : 65.24x

--------------------------------------------------------------------------------
[GRANULARITY 2/3 — MONTHLY]
--------------------------------------------------------------------------------

Store layout
  Chunk window           : 1 calendar month
  Total chunks           : 240
  Average rows per chunk : 20.9
  Minimum rows per chunk : 2
  Maximum rows per chunk : 23
  Metadata size          : 144.82 KiB

Correctness validation
  Full-scan rows : 11
  Pushdown rows  : 11
  Row equality   : PASS
  Value equality : PASS
  Dtype equality : PASS
  Overall status : PASSED

Pruning
  Chunks scanned : 2
  Chunks pruned  : 238
  Pruning rate   : 99.17%
  Rows scanned   : 45
  Rows returned  : 11

Timing
  ┌───────────────────┬──────────────┬─────────────────────────┐
  │ Engine            │ Mean latency │ 95% confidence interval │
  ├───────────────────┼──────────────┼─────────────────────────┤
  │ Full scan         │ 51.363ms     │ 50.847 – 51.878 ms      │
  │ Zone-map pushdown │ 1.703ms      │ 1.686 – 1.720 ms        │
  └───────────────────┴──────────────┴─────────────────────────┘

  Speedup : 30.16x

--------------------------------------------------------------------------------
[GRANULARITY 3/3 — QUARTERLY]
--------------------------------------------------------------------------------

Store layout
  Chunk window           : 90 days
  Total chunks           : 80
  Average rows per chunk : 62.6
  Minimum rows per chunk : 47
  Maximum rows per chunk : 65
  Metadata size          : 48.25 KiB

Correctness validation
  Full-scan rows : 11
  Pushdown rows  : 11
  Row equality   : PASS
  Value equality : PASS
  Dtype equality : PASS
  Overall status : PASSED

Pruning
  Chunks scanned : 2
  Chunks pruned  : 78
  Pruning rate   : 97.50%
  Rows scanned   : 130
  Rows returned  : 11

Timing
  ┌───────────────────┬──────────────┬─────────────────────────┐
  │ Engine            │ Mean latency │ 95% confidence interval │
  ├───────────────────┼──────────────┼─────────────────────────┤
  │ Full scan         │ 31.260ms     │ 31.153 – 31.367 ms      │
  │ Zone-map pushdown │ 1.671ms      │ 1.658 – 1.684 ms        │
  └───────────────────┴──────────────┴─────────────────────────┘

  Speedup : 18.70x

--------------------------------------------------------------------------------
[TRADE-OFF SUMMARY]
--------------------------------------------------------------------------------

┌─────────────┬────────┬────────────┬──────────────┬──────────┬─────────┐
│ Granularity │ Chunks │ Rows/chunk │ Pruning rate │ Pushdown │ Speedup │
├─────────────┼────────┼────────────┼──────────────┼──────────┼─────────┤
│ Weekly      │ 1,010  │ 5.0        │ 99.70%       │ 2.420ms  │ 65.24x  │
│ Monthly     │ 240    │ 20.9       │ 99.17%       │ 1.703ms  │ 30.16x  │
│ Quarterly   │ 80     │ 62.6       │ 97.50%       │ 1.671ms  │ 18.70x  │
└─────────────┴────────┴────────────┴──────────────┴──────────┴─────────┘

Trade-off interpretation
  Finest pruning          : Weekly
  Fewest managed chunks   : Quarterly
  Lowest measured latency : Quarterly
  Selected default        : Monthly
  Selection rationale     : Monthly pruned 99.17% of chunks with 30.16x speedup at 240 chunks, vs. Weekly: 99.70% pruned, 65.24x speedup, 1,010 chunks; Quarterly: 97.50% pruned, 18.70x speedup, 80 chunks

  Limitation
    The best granularity depends on query shape, dataset distribution, and
    storage workload. No universal optimum is claimed.

--------------------------------------------------------------------------------
[RESULT]
--------------------------------------------------------------------------------
  Correctness          : PASSED FOR ALL GRANULARITIES
  Tested granularities : 3
  Recommended default  : Monthly
  Result artifact      : /home/abc/Desktop/project/benchmarks/results/chunk_granularity_sweep.json

================================================================================
 BENCHMARK 5 COMPLETE
================================================================================
EXIT_CODE=0
```

---

## Not included

**Benchmark 3 (Scale Sweep)** — excluded from this run by explicit request. No
`benchmarks/results/scale_sweep.json` exists as of this writing; nothing about
Benchmark 3 above is real data.
