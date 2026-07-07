# Buckshot Roulette Solver

### [⬇ Download the latest build](https://github.com/bx6xb/buckshot-roulette-solver/releases/latest)

---

![main](./assets/main.gif)

A screen overlay for Buckshot Roulette. It reads the game state through
computer vision (live/blank shell counts, items, HP) and works out the
mathematically optimal move with an exact game-tree solver — not a guess,
the actual best play given everything currently known about the round.

The opponent model is not an abstract "perfect dealer": it is a faithful
port of the real dealer AI from the game's decompiled source
(`DealerIntelligence.gd`), including its risky self-shots when blanks
outnumber lives, its pill habits and its item quirks. The solver exploits
all of it, maximises your chance to win the round, and among equally
winning lines prefers the one that spends the fewest items.

## Usage

**The Tasks panel is your next move(s) — not a script for the whole
round.** The solver does look ahead through everything (every dealer
reply, every possible shell, item combos), but it only prints steps whose
outcome is already certain. The moment a step's result depends on
something not yet known — a magnifying-glass reveal, the phone, pills,
a shot on an unknown shell, the dealer's move — the plan stops there and
the last task tells you what to do: update the overlay with what actually
happened and press **AI** again.

So playing with the overlay is a loop:

1. Press **AI** and do the step(s) it shows, in order.
2. When the game reveals something new — a shell type, an HP change, the
   dealer's move, fresh items — mark it in the overlay.
3. Press **AI** again. Repeat until the round is over.

A one-step plan is normal: it means the best follow-up depends on what
that step reveals. The solver never assumes anything you haven't marked.

Hover over the overlay to bring it to full opacity and interact with it —
move your mouse away and it fades back to stay out of your way. The **AI**
button, tasks and **X** (close) button are always fully visible regardless of
hover state.

**Top bar**
- **AI** (top-left) — computes and shows the solver's recommended plan for
  the current turn in the Tasks panel below it.
- **X** (top-right) — closes the overlay.

**Player / Enemy panels**
- Left-click an item icon to increase its count, right-click to decrease.
  Same for the HEALTH / ENEMY HP bar.

**Status toggles** (above the shell chamber)
- **SAW** — turn on once you've used the saw on the current shot (doubles
  its damage), so the solver accounts for that on the next shot.
- **D CUFF** — three-state toggle, click to cycle. Set to **CUFFED** the
  moment you handcuff the dealer (his next turn will be skipped). After
  the shot that skipped his turn, set it to **NO CUFF** — you are on your
  bonus turn, but re-cuffing is not allowed yet (the game forbids cuffing
  twice in a row). Once the dealer has played a real turn, click back to
  off. The plan steps remind you when to switch.

**Shells**
- **LIVE** / **BLANK** — left-click to increase the known count in the
  chamber, right-click to decrease. Raising either adds a slot to the
  numbered row above.
- The numbered slots represent the shells left to fire, in order (leftmost
  = next shot). Left-click a slot to cycle it through
  unknown → live → blank → unknown — use this for shells you've checked
  with the magnifying glass, or a specific shell the Phone told you about.
  As shells get fired during the round, the remaining ones shift left.

**Bottom buttons**
- **SCAN ROUNDS** — reads live/blank shell counts via computer vision.
  Only works at the moment the game actually reveals the shells at the
  start of a round.
- **SCAN ITEMS** — reads items on both sides via computer vision. Works
  whenever the main table view (both players' items + the shotgun) is on
  screen.
- **MAX HP** — sets the round's max HP (2–4). Left-click / right-click to
  cycle up/down.
- **↺** — resets the whole overlay: tasks, shells, items, HP, toggles,
  everything back to empty.

> Detection isn't perfect 100% of the time. If a scan reads something
> wrong, take a screenshot of that moment in-game and open an
> [issue](https://github.com/bx6xb/buckshot-roulette-solver/issues) with it.

**Windows only** — the click-through/transparency layer is built on the
Windows API, so it won't run on macOS or Linux.

## Building it yourself

```bash
pip install -r requirements.txt
pip install pyinstaller

# optional but recommended: the Rust search core (needs a Rust toolchain).
# Without it the solver still works, just slower (pure Python fallback).
pip install maturin
maturin build --release --manifest-path bsr_core/Cargo.toml --out wheels
pip install --no-index --find-links wheels bsr_core

python build.py
```

The exe will be in `dist/BuckshotOverlay.exe`.

## License

MIT — see [LICENSE](LICENSE).
