"""Portfolio state and drawdown-stage logic.

Modules:
  - portfolio_state.py   — PortfolioState class, schema, trade-frontmatter emit.
  - dd_state_machine.py  — drawdown stages with cushion and cooling-off.
  - dd_multiplier.py     — drawdown stage -> position-sizing multiplier.

The JSON persistence layer (atomic save/load, file lock) is not part of this excerpt.
"""
