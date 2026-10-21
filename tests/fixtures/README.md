# Stage Fixtures

This directory holds small, deterministic fixtures for stage-level verification. It is not a full test suite: each stage ticket adds what it needs, and the independent verifier runs it through the stage runner.

T0 only defines this layout and the fixture-to-workdir mapping in `docs/agents/verification.md`. Stage tickets T3–T8 add the actual fixtures; a stage ticket is not complete until its fixture exists.

## Layout

```
tests/fixtures/
├── audio/          # 16 kHz mono WAV, short (e.g. 30 s); transcribe input
├── video/          # short MP4 with an audio track (e.g. 5 s); ingest --source
├── transcript/     # transcript JSON (language, segments, words)
├── segments/       # sentence-level segments JSON
├── translation/    # translation JSON
├── srt/            # source / target / bilingual SRT
└── expected/       # invariants and expectations, not byte-for-byte snapshots
```

## Rules

- Keep every file small. A fixture should be seconds, not minutes; no full films.
- Use synthetic audio, public-domain media, or short clips you have the right to commit.
- Put a short `README.md` beside a fixture that needs context: provenance, duration, language, expected words/timestamps.
- Do not commit secrets, API keys, cookies, model weights, or personal media.
- Prefer named, stable fixtures over ad-hoc files; verification commands must reference a named fixture.
- `expected/` contains invariants (schema, ordering, timestamp bounds, counts), not exact model prose. Do not snapshot LLM output.
- Large binary fixtures are out of scope; if one is unavoidable, stop and ask before adding it.

## Ownership

The ticket that needs a fixture adds it and documents how it was produced. The verifier treats fixtures as read-only; if a fixture is wrong, report it on the ticket instead of editing it in place.