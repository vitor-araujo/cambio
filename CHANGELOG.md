# Changelog

All notable changes to cambio are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- New cambio brand mark, responsive browser favicons, and Apple touch icon.
- Backend-free portfolio mode with synthetic market data, local interactions,
  responsive case-study framing, and a dedicated static build.
- GitHub Pages deployment workflow and host-agnostic launch instructions.

### Changed

- Rewritten project README with clearer positioning, setup instructions,
  execution-policy explanation, validation context, and reference sections.
- Relative asset routing so the static build works from repository subpaths.

### Fixed

- Portfolio navigation now imports the brand mark through Vite's asset pipeline
  and falls back to a text mark instead of showing a broken-image glyph.

## [0.8.0] - 2026-07-20

### Added

- Adaptive execution planner with a bounded 3–4 day cadence, a declining
  opportunity hurdle, and a mandatory tranche when the cadence limit is due.
- Short-window execution quality based on the current USD/BRL percentile over
  the last 20 sessions, kept separate from the medium-term directional model.
- Walk-forward cadence evaluation with pre-2025 calibration, a 2025+ holdout,
  bootstrap confidence intervals, and spread-aware reporting.
- Institutional execution desk UI with an actionable order ticket, countdown,
  market chart, execution ledger, broker handoff, risk controls, and undoable
  execution marking.
- Developer-focused terminal interface with a compact live status line,
  material-state updates, and a readable non-interactive mode.
- Execution-policy tests covering starter fills, opportunity windows, forced
  cadence, sizing bounds, quality ranking, and journal mark/undo behavior.

### Changed

- Recurring tranche sizing now defaults to 25–50% of the available USD balance.
- Live quote polling is decoupled from slower full-market refreshes, and watch
  intervals can be configured down to one minute.
- Journal reads are bounded and aggregated; execution metadata now includes the
  trigger, rationale, cadence, quality score, and current hurdle.
- Threshold updates are validated and saved atomically by the API.

### Fixed

- A non-zero DCA tranche can no longer be mislabeled as `wait` merely because
  its size is below 40%.
- Empty optional market series no longer crash backtests.
- Polling and watch output no longer flood the HTTP or terminal logs.

### Validation

- The 3–4 day selector was evaluated over 284 historical windows. It improved
  on always waiting until day four, but did not improve on always executing on
  day three across the full sample. Treat cadence as disciplined execution,
  not guaranteed alpha or a promise of the best available rate.

[0.8.0]: https://github.com/vitor-araujo/cambio/compare/v0.7.7...v0.8.0
