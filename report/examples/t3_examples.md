# Task 3 qualitative examples

Answers marked * are correct. Counts are answer-group sizes among the revealed rollouts.

## Selection disagreement (MATH)

Rule: the first 3 eval questions (split order) where the rollouts selected by self-certainty + RWC and by self-consistency differ in correctness at N = 32, order 0.

- `test:2460` gold `154`: SC → `58` (wrong); self-cert. + RWC → `154` (correct). Pool: `58` ×11, `154`* ×11, `NO_ANSWER` ×3, `50` ×1, `346` ×1, `90` ×1, `13` ×1, `80` ×1, `98` ×1, `45` ×1
- `test:1903` gold `80`: SC → `75` (wrong); self-cert. + RWC → `80` (correct). Pool: `75` ×7, `80`* ×7, `60` ×3, `55` ×3, `100` ×2, `90` ×2, `180` ×2, `120` ×2, `NO_ANSWER` ×1, `20` ×1, `30` ×1, `140` ×1
- `test:2744` gold `2`: SC → `0` (wrong); self-cert. + RWC → `2` (correct). Pool: `0` ×16, `2`* ×15, `\frac{7 + \sqrt{29}}{2}` ×1

## Selection disagreement (MathVista)

Rule: first 3 eval questions (split order) where VisConf-highest and SC disagree in correctness at N=32, order 0.

- `379`: visconf → `B` (wrong, T=116, V=0.055, C=0.389, Z=0.021); sc → `A` (correct, T=257, V=0.040, C=0.485, Z=0.019)
- `358`: visconf → `D` (wrong, T=277, V=0.051, C=0.457, Z=0.023); sc → `A` (correct, T=782, V=0.034, C=0.527, Z=0.018)
- `821`: visconf → `B` (wrong, T=118, V=0.089, C=0.495, Z=0.044); sc → `A` (correct, T=112, V=0.075, C=0.448, Z=0.034)

## Revision (live, 50 MATH questions)

Rule: every question where the revised answer differs from the draft answer.

- `test:2892` gold `-71`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `0` (wrong)
- `test:3285` gold `0`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `1688` (wrong)
- `test:29` gold `16`: draft `-16` (wrong) → revised `16` (correct)
- `test:2948` gold `1+\sqrt 2,\ 1-\sqrt 2,\ 2+\sqrt 3,\ 2-\sqrt 3`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `0 \text{ and } 3` (wrong)
- `test:2358` gold `2 \sqrt{5}`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `2` (wrong)
- `test:4516` gold `\frac{25 \sqrt{10}}{4}`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `4\sqrt{10}` (wrong)
- `test:3409` gold `3`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `3` (correct)

## Premature stopping (MATH, agreement rule, τ = 0.625)

Rule: the first 3 distinct questions in `adaptive.json` → `incorrect_early_stops.examples_first10` (eval split order, first listed order of each), i.e. early stops with a wrong answer that fixed N = 32 self-consistency gets right.

- `test:2460` order 7, gold `154`: stopped at n = 4 with `58` [`58` ×3, `154`* ×1]; at N = 32 SC gives `154` [`154`* ×11, `58` ×11, `NO_ANSWER` ×3, `346` ×1, `50` ×1, `45` ×1, `98` ×1, `13` ×1, `90` ×1, `80` ×1]
- `test:534` order 6, gold `56`: stopped at n = 4 with `48` [`48` ×3, `56`* ×1]; at N = 32 SC gives `56` [`56`* ×23, `48` ×4, `50` ×4, `44` ×1]
- `test:3409` order 4, gold `3`: stopped at n = 8 with `0` [`0` ×5, `3`* ×2, `9` ×1]; at N = 32 SC gives `3` [`3`* ×21, `0` ×6, `9` ×4, `7` ×1]

## Premature stopping (MathVista, agreement rule, τ = 0.5)

Rule: the first 3 distinct questions in `adaptive.json` → `incorrect_early_stops.examples_first10` (eval split order, first listed order of each), i.e. early stops with a wrong answer that fixed N = 32 self-consistency gets right.

- `358` order 4, gold `A`: stopped at n = 4 with `D` [`D` ×2, `A`* ×1, `E` ×1]; at N = 32 SC gives `A` [`A`* ×14, `D` ×7, `E` ×5, `C` ×3, `B` ×3]
- `821` order 4, gold `A`: stopped at n = 4 with `B` [`B` ×3, `A`* ×1]; at N = 32 SC gives `A` [`A`* ×21, `B` ×8, `NO_ANSWER` ×3]
- `434` order 0, gold `A`: stopped at n = 4 with `C` [`C` ×2, `NO_ANSWER` ×1, `A`* ×1]; at N = 32 SC gives `A` [`A`* ×17, `C` ×7, `B` ×4, `NO_ANSWER` ×2, `E` ×1, `D` ×1]
