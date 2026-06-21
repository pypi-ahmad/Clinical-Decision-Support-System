# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `docs/HANDBOOK.md` &mdash; zero-to-hero walkthrough of every major subsystem (definition &rarr; why &rarr; how &rarr; code &rarr; outputs).
- `docs/ARCHITECTURE.md` &mdash; module dependency graph, request lifecycle, and SQLite schema.
- `docs/SECURITY.md` &mdash; threat model, defense-in-depth layers, and operator hardening checklist.
- `docs/DEVELOPMENT.md` &mdash; lint, type-check, test, CI, and "add a new backend / node / provider" recipes.

### Changed

- `README.md` &mdash; added "What is MediScan OCR?" elevator, "Why use it?" benefit table, and a documentation navigation block. Existing API reference and env-var tables preserved.
- `README.quickstart.md` &mdash; now points at `docs/HANDBOOK.md` for the full reference and at the per-subsystem docs for deep dives.
- `TEST_REPORT.md` &mdash; refreshed to reflect the post-eval-harness test inventory (241 pass / 8 skip on default markers, ~249 total).

### Fixed

- `TEST_REPORT.md` no longer claims 34 tests / 97% coverage; that was the state of the suite before the eval harness and security hardening landed.

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).
