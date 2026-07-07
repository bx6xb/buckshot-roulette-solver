# Task: port the solver core to Rust

Self-contained brief for an AI agent (or human). The Python engine is the
reference implementation — this task is a faithful, much faster port.
**Do not commit or push anything — the repo owner commits himself.**

## Goal

Port the search core of [ai_engine.py](ai_engine.py) to Rust as a PyO3
extension module (suggested name `bsr_core`), keeping Python as the oracle:

1. `ai_engine.py` tries `import bsr_core` and uses it; if the import fails
   it falls back to the existing pure-Python solver. Public API
   (`run_ai(...)`) and all plan texts stay EXACTLY as they are (English).
2. Differential tests: random + curated positions must produce the same
   best action and win probability (tolerance 1e-9) in both engines.
3. The overlay (`overlay.py`) must not change at all.

## What the engine is

Exact expectimax over belief states ("worlds" = all shell sequences
consistent with known info, uniform). No minimax: the dealer is a FAITHFUL
port of `DealerIntelligence.gd` from the game's decompiled source —
clone https://github.com/thecatontheceiling/buckshotroulette and port from
the actual GDScript, endless/Double-or-Nothing mode. Every quirk matters
and is already replicated in the Python code (read its comments, they cite
decompile line numbers):

- item choice = first eligible item in table order -> modelled as uniform
  pick among eligible item INSTANCES; dealer's own items always take
  priority over items stealable via adrenaline;
- FigureOutShell inference incl. burner-phone memory subtraction;
- beer resets his shell knowledge but NOT his target (stale-target bug);
- leftover-handsaw coin flip; count-weighted coin flips (live>blank ->
  shoot player, blank>live -> self, equal -> 50/50);
- expired medicine: never at exactly 1 HP, once per turn, blocked by cigs;
- last-shell rule; cuffs are a 3-state cycle (CUFFED -> COOLDOWN -> OFF),
  no re-cuff until a real turn, ALL cuffs removed at every reload;
- the player ALWAYS acts first after a reload; barrel regrows at turn end;
  saw survives a beer eject but not a fired shot;
- burner phone: player version never reveals the last shell of an 8-load
  (result 7 remapped to 6); dealer version is uniform over future slots.

Values are `(win_probability, tiebreak)` compared lexicographically
(P_EPS = 1e-9). Tiebreak = item economy (ITEM_W weights, enemy items at
0.6) + 0.5*HP − 0.01 per player action. The reload frontier uses the
formula in `_reload_value` — reproduce it bit-compatibly first, improve it
second (see below).

## Performance targets (current Python numbers)

- typical positions: <0.1 s  -> expect instant
- 4 unknown shells + 8v8 items: ~10 s exact -> expect well under 1 s
- fresh 8-shell load + 8v8 items: exceeds RAM in Python (fallback mode,
  ~15 s estimate) -> in Rust try to solve exactly (compact bit-packed
  states); keep the memo cap + iterative-deepening fallback as a safety
  net (its clamped cutoff (0.03..0.97) guarantees p==1.0 is a proven win).

## Follow-up tasks once the port is verified

1. **Honest reload tail**: replace the hand-tuned `_reload_value` constants
   with a real expectation over next loads. Generator from the decompile
   (RoundManager.GenerateRandomBatches): total = randi(2,8),
   live = max(1, total/2) (integer div, so blank >= live, each <= 4),
   items dealt = randi(2,5) EACH side, random kinds, capped by 8 slots.
2. **Slot-cap fix**: held items near the 8-slot cap are worth slightly less
   (full slots block the free item deal at reload).

## Build & ship

- maturin + PyO3, Python 3.13, Windows x64. Rust toolchain (stable-msvc)
  is installed; VS Build Tools (C++ workload) should be too — VERIFY the
  linker first with a hello-world `cargo build` before starting. Beware:
  running cargo from Git Bash can shadow MSVC `link.exe` with GNU
  coreutils `link` — build from cmd/PowerShell if linking fails oddly.
- PyInstaller must bundle the `.pyd` (build.py); CI: add a Rust toolchain +
  `maturin build --release` step to .github/workflows/build.yml before the
  PyInstaller step.
- Run the existing suite: `python -m unittest discover -s tests`
  (19 tests, all must stay green).
