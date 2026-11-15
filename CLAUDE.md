Read README.md first. This repository holds Rust solvers only: no game server, UI or opponent model.

## Commands

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace -- --ignored   # needs the real artifacts named in each test
```

## Rules

- One crate per solver in `crates/`. A crate is a library plus, when it builds artifacts, one
  binary in `src/main.rs` whose subcommands are the build, solve and certify steps.
- The workspace owns lints, dependency versions and the toolchain. A crate inherits them. A new
  dependency needs a reason a reviewer can check; prefer `std`.
- Write the shortest correct code. No trait without two implementations, no generic without two
  instantiations, no option nobody passes, no dead code, no commented-out code.
- Library code returns `Result` with a crate error enum for bad input or bad artifacts. It panics
  only on a broken internal invariant, through `expect("<the invariant>")` or `assert!`.
- `unsafe` appears only for memory maps, with a `// SAFETY:` comment.
- Every public item has a doc comment that says what it computes, in the game's terms.
- Each numerical port proves parity with the Python reference through a golden fixture in
  `crates/<crate>/tests/data/`. `tests/data/README.md` names the source repository, commit and
  command that produced each fixture. A test that needs a large local artifact is `#[ignore]` and
  names the path in its message.
- Bit-identical output where the reference is deterministic. Otherwise the test states the
  tolerance and why it holds.
