# Task 3 qualitative examples

Answers marked * are correct. Counts are answer-group sizes among the revealed rollouts.

## Selection disagreement (MATH)

Rule: the first 3 eval questions (split order) where the rollouts selected by self-certainty + RWC and by self-consistency differ in correctness at N = 32, order 0.

- `test:2460` gold `154`: SC → `58` (wrong); self-cert. + RWC → `154` (correct). Pool: `58` ×11, `154`* ×11, `NO_ANSWER` ×3, `50` ×1, `90` ×1, `45` ×1, `98` ×1, `80` ×1, `13` ×1, `346` ×1
- `test:2744` gold `2`: SC → `0` (wrong); self-cert. + RWC → `2` (correct). Pool: `0` ×16, `2`* ×15, `\frac{7 + \sqrt{29}}{2}` ×1
- `test:2206` gold `512`: SC → `384` (wrong); self-cert. + RWC → `512` (correct). Pool: `384` ×3, `512`* ×2, `NO_ANSWER` ×2, `6` ×2, `10` ×2, `192` ×1, `\frac{4}{35}` ×1, `\frac{64}{3}` ×1, `\frac{8}{189}` ×1, `\frac{32}{3}` ×1, `\frac{512}{3}` ×1, `945` ×1, `140` ×1, `\frac{2}{35}` ×1, `\frac{2}{175}` ×1, `\frac{10}{3}` ×1, `\frac{60}{11}` ×1, `\frac{1536}{7}` ×1, `\frac{9}{5}` ×1, `682` ×1, `1228.8` ×1, `1600` ×1, `\frac{2560}{63}` ×1, `21` ×1, `12288000000` ×1, `\frac{2}{7}` ×1

## Selection disagreement (MathVista)

Rule: first 3 eval questions (split order) where VisConf-highest and SC disagree in correctness at N=32, order 0.

- `379`: visconf → `B` (wrong, T=116, V=0.055, C=0.389, Z=0.021); sc → `A` (correct, T=257, V=0.040, C=0.485, Z=0.019)
- `358`: visconf → `D` (wrong, T=277, V=0.051, C=0.457, Z=0.023); sc → `A` (correct, T=499, V=0.039, C=0.464, Z=0.018)
- `821`: visconf → `B` (wrong, T=118, V=0.089, C=0.495, Z=0.044); sc → `A` (correct, T=146, V=0.088, C=0.401, Z=0.035)

## Revision (live, 50 MATH questions)

Rule: every question where the revised answer differs from the draft answer.

- `test:2892` gold `-71`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `0` (wrong)
- `test:3285` gold `0`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `1688` (wrong)
- `test:29` gold `16`: draft `-16` (wrong) → revised `16` (correct)
- `test:2948` gold `1+\sqrt 2,\ 1-\sqrt 2,\ 2+\sqrt 3,\ 2-\sqrt 3`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `0 \text{ and } 3` (wrong)
- `test:2358` gold `2 \sqrt{5}`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `2` (wrong)
- `test:4516` gold `\frac{25 \sqrt{10}}{4}`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `4\sqrt{10}` (wrong)
- `test:3409` gold `3`: draft `none` (draft truncated at 1,024 tokens) (wrong) → revised `3` (correct)

## Premature stopping (MATH, agreement rule, τ = 0.5)

Rule: the first 3 entries of `adaptive.json` → `incorrect_early_stops.examples_first10` (eval split order), i.e. early stops with a wrong answer that fixed N = 32 self-consistency gets right.

- `test:3504` order 0, gold `216`: stopped at n = 4 with `120` [`120` ×2, `NO_ANSWER` ×1, `36` ×1]; at N = 32 SC gives `216` [`216`* ×13, `120` ×4, `NO_ANSWER` ×2, `36` ×2, `24` ×1, `360` ×1, `72` ×1, `1944` ×1, `144` ×1, `1152` ×1, `1728` ×1, `576` ×1, `15120` ×1, `1440` ×1, `168` ×1]
- `test:3504` order 2, gold `216`: stopped at n = 4 with `120` [`120` ×2, `24` ×1, `1728` ×1]; at N = 32 SC gives `216` [`216`* ×13, `120` ×4, `36` ×2, `NO_ANSWER` ×2, `24` ×1, `1728` ×1, `1152` ×1, `72` ×1, `1440` ×1, `15120` ×1, `1944` ×1, `144` ×1, `360` ×1, `576` ×1, `168` ×1]
- `test:2732` order 3, gold `\frac{266664}{5}`: stopped at n = 4 with `54166.47` [`54166.47` ×2, `11111` ×1, `54166` ×1]; at N = 32 SC gives `53332.8` [`53332.8`* ×7, `2222.2` ×3, `54166.47` ×2, `53760` ×2, `11111` ×1, `54166` ×1, `53327.8` ×1, `48484.8` ×1, `53352.05` ×1, `54040.8` ×1, `6399336` ×1, `1170328` ×1, `52943.6` ×1, `48332.8` ×1, `52920` ×1, `53333` ×1, `24` ×1, `58332.8` ×1, `52889.8` ×1, `53200` ×1, `54166.466666\ldots` ×1, `52853.63333333333` ×1]

## Premature stopping (MathVista, agreement rule, τ = 0.625)

Rule: the first 3 entries of `adaptive.json` → `incorrect_early_stops.examples_first10` (eval split order), i.e. early stops with a wrong answer that fixed N = 32 self-consistency gets right.

- `847` order 8, gold `A`: stopped at n = 4 with `C` [`C` ×4]; at N = 32 SC gives `A` [`A`* ×21, `C` ×6, `B` ×5]
- `612` order 8, gold `4`: stopped at n = 4 with `3` [`3` ×3, `5` ×1]; at N = 32 SC gives `4` [`4`* ×15, `3` ×10, `5` ×7]
- `542` order 1, gold `2`: stopped at n = 4 with `3` [`3` ×3, `5` ×1]; at N = 32 SC gives `2` [`2`* ×12, `3` ×11, `5` ×3, `1` ×2, `7` ×1, `NO_ANSWER` ×1, `10` ×1, `6` ×1]
