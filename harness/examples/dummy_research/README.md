Dummy aerodrome-weather task used to prove the harness before AeroWF is mounted.

The process is linear in lagged wind and runway-relative headwind, so:

- **model axis** (persistence → Ridge) should improve validation and transfer
- **representation / physics / data** alone cannot help persistence, which ignores extra columns and weights

That is the ablation lock working, not a harness bug. Point `domain.pipeline_root` at a real AeroWF tree when it is ready.
