"""Adapt the pinned Bend 2.0.40 CPU emitter for DTH's packed binary64 operations."""

import re
import struct
from pathlib import Path

BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
COMPARE = {"is_eq": "==", "is_lt": "<", "is_le": "<=", "is_gt": ">", "is_ge": ">="}
OPERATIONS = {
    name: (f"dth_f64($0, $1) {op} dth_f64($2, $3)", ["F64", "F64"], "F64")
    for name, op in BINARY.items()
}
OPERATIONS.update(
    {
        name: (f"dth_f64($0, $1) {op} dth_f64($2, $3)", ["F64", "F64"], "Bool")
        for name, op in COMPARE.items()
    }
)
OPERATIONS.update(
    {
        "neg": ("-dth_f64($0, $1)", ["F64"], "F64"),
        "abs": ("fabs(dth_f64($0, $1))", ["F64"], "F64"),
        "pow": ("pow(dth_f64($0, $1), dth_f64($2, $3))", ["F64", "F64"], "F64"),
        "from_nat": ("(double)($0)", ["Nat"], "F64"),
    }
)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one occurrence in pinned Bend 2.0.40: {old}")
    return text.replace(old, new)


def patch(backend: Path) -> None:
    """Add packed float64 operations and C entry points to the pinned emitter."""
    base = backend / "base.bend"
    compiler = backend / "comp.ts"
    definitions = base.read_text()
    source = compiler.read_text()
    definitions += "\n\ntype F64 is Data:\n  F64{lo: U32, hi: U32}\n"
    entries = []
    for name, (expr, args, result) in OPERATIONS.items():
        params = "\n".join(
            f"  for {arg}: {kind}" for arg, kind in zip("ab", args, strict=False)
        )
        definitions += f"\nlaw F64.{name}:\n{params}\n  {result}\n"
        c = (
            f'["dth_f64_lo({expr})", "dth_f64_hi({expr})"]'
            if result == "F64"
            else f'"({expr})"'
        )
        entries.append(f"  f64_{name}: {{ C: {c} }},")
    source = replace_once(
        source, "  f32_neg: { C:", "\n".join(entries) + "\n  f32_neg: { C:"
    )
    source = replace_once(
        source,
        """  const as = args.map((v, i) => val_own(val_to(sc, v, lays[i]))[0])
    .map((w, i) => op_bare(m.args[i]) ? w : emit_alias(sc, w, "a"));""",
        """  const as = args.flatMap((v, i) => {
    const ws = val_own(val_to(sc, v, lays[i]));
    return (op.startsWith("f64_") ? ws : ws.slice(0, 1))
      .map((w) => op_bare(m.args[i]) ? w : emit_alias(sc, w, "a"));
  });""",
    )
    source = replace_once(
        source,
        "INLINE f32 f32_unbox(u64 x) {",
        """// F64 values use two unboxed words in the packed U32 array layout.
INLINE double dth_f64(u32 lo, u32 hi) {
  union { u64 u; double f; } v = { ((u64)hi << 32) | lo };
  return v.f;
}

INLINE u32 dth_f64_lo(double f) {
  union { double f; u64 u; } v = { f };
  return (u32)v.u;
}

INLINE u32 dth_f64_hi(double f) {
  union { double f; u64 u; } v = { f };
  return (u32)(v.u >> 32);
}

INLINE f32 f32_unbox(u64 x) {""",
    )
    source = replace_once(
        source,
        "    fids: new Set<Name>(),",
        "    fids: new Set<Name>(),\n    native: new Set<Name>(),",
    )
    source = replace_once(
        source, "([CF]ID)\\(([\\w./~-]+)", "([CF]ID|SPIN)\\(([\\w./~-]+)"
    )
    source = replace_once(
        source,
        '    return FL.js ? JSON.stringify(p === "CID"',
        '''    if (p === "SPIN") return name_id("spin_", q);
    return FL.js ? JSON.stringify(p === "CID"''',
    )
    source = replace_once(
        source,
        """      if (p === "FID" && fun_runs(tld_of(q))) {
        src_add(q);
        FL.fids.add(q);
      }""",
        """      if (p === "FID" && fun_runs(tld_of(q))) {
        src_add(q);
        FL.fids.add(q);
      }
      if (p === "SPIN") {
        if (!fun_runs(tld_of(q))) die("SPIN requires a filled def: " + q);
        src_add(q);
        FL.native.add(q);
      }""",
    )
    source = replace_once(
        source,
        """  let was;
  do {
    was = FL.facts.size;""",
        """  for (const k of FL.native) {
    if (!flat_of(k)) die("SPIN requires a flat CPU def: " + k);
    emit_native(scope_new(), k);
  }
  let was;
  do {
    was = FL.facts.size;""",
    )
    source = replace_once(
        source,
        """  const live = reach(["main", ...FL.fids].map(seg_fid));""",
        """  const live = reach(["main", ...FL.fids].map(seg_fid)
    .concat([...FL.native].map((k) => name_id("spin_", k))));""",
    )
    base.write_text(definitions)
    compiler.write_text(source)


TOKEN = re.compile(
    r'#[^\n]*|"(?:\\.|[^"\\])*"|(?<!\w)-?\d+\.\d+(?:[eE][+-]?\d+)?(?!\w)'
)


def lower_literals(source: str) -> str:
    """Write decimal literals as two exact binary64 words for Bend's parser."""

    def lower(match: re.Match[str]) -> str:
        token = match[0]
        if token.startswith(("#", '"')):
            return token
        lo, hi = struct.unpack("<II", struct.pack("<d", float(token)))
        return f"F64{{{lo}, {hi}}}"

    return TOKEN.sub(lower, source)
