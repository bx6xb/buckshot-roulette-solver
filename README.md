# Buckshot Roulette Solver

### [⬇ Download the latest build](releases/latest)

---

A screen overlay for Buckshot Roulette. It reads the game state through
computer vision (live/blank shell counts, items, HP) and works out the
mathematically optimal move with a full adversarial search — not a guess,
the actual best play given everything currently known about the round.

![main](./assets/main.gif)

## Usage

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
- **D CUFF** — turn on the moment you handcuff the dealer, so the solver
  knows you get two actions this turn instead of one. Turn it back off as
  soon as you've used the first of those two actions — otherwise the
  solver will keep assuming you have an extra action you no longer have.

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
> [issue](issues) with it.

**Windows only** — the click-through/transparency layer is built on the
Windows API, so it won't run on macOS or Linux.

## Building it yourself

```bash
pip install -r requirements.txt
pip install pyinstaller
python build.py
```

The exe will be in `dist/BuckshotOverlay.exe`.

## License

MIT — see [LICENSE](LICENSE).
