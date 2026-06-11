# ShanghaiTech/SIST RAG Codex Multiagent Guide

This directory contains project-local Codex subagent configuration for the
ShanghaiTech/SIST official-source RAG repository. It supplements the root
`AGENTS.md`; it does not replace the root project rules.

## Skills Discovery

Relevant globally installed skills from `~/.codex/skills/` include:

- `grill-with-docs` — Stress-test plans against `CONTEXT.md`, terminology, and
  documented decisions.
- `experiment-plan` — Design retrieval/generation evaluation plans,
  comparisons, and acceptance criteria.
- `python-testing` — Build focused pytest coverage and verification strategy.
- `python-patterns` — Keep Python implementation idiomatic and maintainable.
- `benchmark` — Design and interpret retrieval or answer-quality comparisons.
- `run-validation` — Execute validation checks before completion.
- `verification-loop` — Coordinate build, test, lint, and typecheck loops.
- `documentation-lookup` — Verify library and API behavior against docs.
- `security-review` — Check security and data-handling risks.
- `git-workflow` — Keep commits, branches, and diffs clean.

Project-local skills, if added later, should live under `.agents/skills/`.
Each skill should provide:

- `SKILL.md` — Detailed instructions and workflow.
- `agents/openai.yaml` — Optional Codex interface metadata when the skill
  provides it.

Prefer skills that support research planning, Python testing, documentation
review, and verification. Do not invent or list unavailable skills.

## Default Workflow

- Use subagents only when the user explicitly asks for multiagent or parallel
  work.
- Keep `max_depth = 1`; child agents should not recursively fan out work.
- Each custom agent must read the root `AGENTS.md` and `CONTEXT.md` before
  making claims or edits.
- The parent agent owns orchestration, final decisions, and integration.
- Remote SSH, large artifact generation, and final Git operations stay with the
  parent agent.
- Preserve the official-source boundary. Do not add hosted LLM APIs or
  non-official web sources to the implementation plan.
- Treat generated crawl, merge, RAG, and evaluation artifacts as append-only
  unless the parent agent explicitly assigns cleanup.

## Roles

- `corpus_mapper`: read-only mapping of crawler, parser, merge, ingest, index,
  retrieval, generation, data, config, and documentation paths.
- `evaluation_planner`: read-only retrieval/generation evaluation planning using
  the `dense` before-optimization, `hybrid` after-optimization, and `bm25`
  diagnostic terminology from `CONTEXT.md`.
- `implementation_worker`: scoped code or config edits after the parent agent
  assigns a specific ownership boundary.
- `reviewer`: read-only correctness, official-source, data-contract, security,
  and missing-test review.
- `docs_guard`: read-only documentation and retrieval terminology drift checks.

## Documentation Flow

`docs_guard` reports stale wording and suggested replacements. The parent agent
performs final documentation edits so `README.md`, root `AGENTS.md`,
`CONTEXT.md`, `doc/*.md`, and evaluation specs stay synchronized.
