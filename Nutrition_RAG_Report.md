# Phase 0 Benchmark Report — Nutrition RAG

Evaluated naive baselines against 100 lookup queries spanning common
foodservice ingredients. Each query has expected keywords; a result
is correct if any of its top-K matches contains ALL keywords (case-insensitive).

## Headline metrics

| Metric | Target | usda_api_direct | sql_fts5_local |
|--------|--------|--------|--------|
| Top-1 accuracy | ≥ 0.90 | 75.0% | 95.0% |
| Top-5 recall | — | 76.0% | 95.0% |
| Top-10 recall | ≥ 0.95 | 77.0% | 96.0% |
| MRR | — | 0.757 | 0.951 |
| Median match rank | 1 | 1 | 1 |
| Latency p50 (ms) | < 1000 | 4.2 | 861.9 |
| Latency p95 (ms) | < 3000 | 10935.6 | 1094.1 |
| Queries w/ no match | 0 | 23 | 4 |

## Scope decision

**Recommendation: SCOPE DOWN.** At least one baseline already hits the targets (top-1 95.0%, top-10 96.0%). The full RAG pipeline would buy little. Ship the better baseline as a thin LLM-wrapped service.

## Failing queries (top 10 examples)


### usda_api_direct — queries with NO match in top 10 (23)
- Q001 `chicken breast raw` (expected: ['chicken', 'breast', 'raw'])
- Q007 `tuna canned` (expected: ['tuna', 'canned'])
- Q008 `pork chop raw` (expected: ['pork'])
- Q009 `shrimp raw` (expected: ['shrimp', 'raw'])
- Q014 `milk whole` (expected: ['milk', 'whole'])
- Q016 `butter unsalted` (expected: ['butter'])
- Q023 `strawberries raw` (expected: ['strawberr'])
- Q026 `avocado raw` (expected: ['avocado', 'raw'])
- Q029 `pineapple raw` (expected: ['pineapple'])
- Q032 `spinach raw` (expected: ['spinach', 'raw'])

### sql_fts5_local — queries with NO match in top 10 (4)
- Q010 `bacon raw` (expected: ['bacon'])
- Q044 `oatmeal cooked` (expected: ['oat'])
- Q074 `black pepper ground` (expected: ['pepper', 'black'])
- Q094 `corn on the cob` (expected: ['corn'])
