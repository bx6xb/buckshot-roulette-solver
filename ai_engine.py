"""Buckshot Roulette AI Engine — exact solver against the REAL dealer AI.

The engine solves the current shell load exactly (no depth limits, no
heuristic mid-round evaluations): the game tree is finite because every
action consumes either an item or a shell, so plain memoized recursion
terminates on its own.  The only estimated leaf is the reload frontier
(shells run out and a fresh, unknown load begins).

The dealer is NOT modelled as an adversarial minimax opponent.  It is a
faithful port of DealerIntelligence.gd from the decompiled game
(github.com/thecatontheceiling/buckshotroulette), endless/Double-or-Nothing
mode, including its quirks:

  * item choice is "first eligible item in table order" — table order is
    unobservable from outside, so it is modelled as a uniform pick among
    eligible item instances (the dealer's own items always precede items
    stolen through adrenaline, matching the array build order in the code);
  * FigureOutShell(): the dealer deduces the current shell when only one
    type remains among shells unknown to him (burner-phone memory counts);
  * beer resets his shell knowledge (endless mode) but NOT his shooting
    target — the "stale target" bug where he can shoot himself on an
    unknown shell after glass->blank->beer is reproduced on purpose;
  * with an unknown shell his shoot decision is a count-weighted coin flip
    (more lives -> shoots player, more blanks -> shoots himself, equal ->
    true 50/50), and the leftover-handsaw coin flip can pre-commit the
    target before that;
  * expired medicine is never used at exactly 1 HP, and never twice within
    one turn;
  * handcuffs follow the real 3-state cycle: cuffed (skip pending) ->
    cooldown (skip consumed, re-cuffing blocked) -> off after a real turn;
    all cuffs are removed when a new shell load begins.

Values are (win_probability, tiebreak) pairs compared lexicographically:
maximise the chance to win the current round first, then prefer lines that
keep more/better items (they carry through reloads and across the
Double-or-Nothing screen), keep more HP, and need fewer steps.
"""

from __future__ import annotations

import sys
from itertools import combinations
from typing import Dict, Iterable, List, NamedTuple, Tuple

# ── Optional Rust core ────────────────────────────────────────────────────────
# bsr_core is a faithful PyO3 port of the search below (same best actions,
# same win probabilities to 1e-9 — enforced by tests/test_rust_parity.py).
# When the compiled module is importable the search runs in Rust; otherwise
# this file's pure-Python solver does the exact same job, only slower.
try:
    import bsr_core as _bsr
except ImportError:                       # wheel not built/installed
    _bsr = None

# ── Item indices (must match ITEMS_CONF order in overlay.py) ─────────────────
GLASS, PILLS, PHONE, CUFFS, ADRENALINE, SAW, CIGGS, BEER, INVERTER = range(9)

LIVE, BLANK = 1, 2

# Cuff cycle, identical for both sides (mirrors dealerCuffed/aboutToBreakFree
# and playerCuffed/playerAboutToBreakFree in RoundManager.gd).
CUFF_NONE, CUFF_ACTIVE, CUFF_COOLDOWN = 0, 1, 2

DEFAULT_MAX_HP = 4
MAX_PLAN_STEPS = 14

# Lexicographic value tuning ---------------------------------------------------
P_EPS     = 1e-9   # win probabilities closer than this count as a tie
STEP_COST = 0.01   # tiebreak penalty per player action (prefer shorter plans)
HP_TB     = 0.5    # tiebreak weight of one player HP
ENEMY_ITEM_TB = 0.6  # denying dealer items is worth this fraction of ours

# Search-size guards ------------------------------------------------------------
# The solver is exact whenever the position fits in memory.  Gigantic openings
# (a fresh 8-shell load with both inventories full) can exceed any RAM, so the
# search falls back to iterative deepening over CONSUMED SHELLS with a clamped
# heuristic at the horizon.  The heuristic can never produce 1.0 or 0.0, so a
# reported "100% guaranteed win" is a proven one in both modes.
EXACT_MEMO_CAP    = 900_000
FALLBACK_MEMO_CAP = 350_000
# Skip the doomed exact attempt when the position is obviously enormous.
FALLBACK_GATE_WORLDS = 40
FALLBACK_GATE_ITEMS  = 10

_memo_cap = EXACT_MEMO_CAP


class _SearchOverflow(Exception):
    """Raised when the memo table outgrows the current cap."""

# Item weights used by the tiebreak economy and the reload heuristic.
# Index order: GLASS PILLS PHONE CUFFS ADRENALINE SAW CIGGS BEER INVERTER.
ITEM_W = (1.8, 0.7, 1.0, 2.2, 1.6, 2.2, 1.2, 1.2, 1.6)

Value = Tuple[float, float]          # (win probability, tiebreak score)
Branch = Tuple[float, "State", str]  # (probability, next state, phase)

# Phases returned by action application:
#   "player"  — same player keeps acting (extra turn / mid-turn item)
#   "dealer"  — turn passes to the dealer side
#   "win"     — dealer HP reached 0
#   "loss"    — player HP reached 0
#   "reload"  — shells ran out, a fresh load begins (heuristic frontier)


class State(NamedTuple):
    php: int
    ehp: int
    max_hp: int
    pitems: Tuple[int, ...]
    eitems: Tuple[int, ...]
    worlds: Tuple[Tuple[int, ...], ...]  # equiprobable candidate sequences
    mem: Tuple[bool, ...]                # dealer's burner-phone knowledge
    saw: bool
    e_cuff: int
    p_cuff: int


# ── Belief helpers ────────────────────────────────────────────────────────────
def _init_worlds(shells: List[int], live: int, blank: int) -> Tuple[Tuple[int, ...], ...]:
    """All shell sequences consistent with the marked shells and counts."""
    if len(shells) != live + blank:
        return ()
    if any(sh not in (0, 1, 2) for sh in shells):
        return ()

    unknown_idx = [i for i, sh in enumerate(shells) if sh == 0]
    known_live  = sum(1 for sh in shells if sh == LIVE)
    known_blank = sum(1 for sh in shells if sh == BLANK)
    need_live   = live - known_live
    need_blank  = blank - known_blank

    if need_live < 0 or need_blank < 0 or need_live + need_blank != len(unknown_idx):
        return ()

    worlds: List[Tuple[int, ...]] = []
    for live_choice in combinations(range(len(unknown_idx)), need_live):
        live_set = set(live_choice)
        world = list(shells)
        for pos, shell_idx in enumerate(unknown_idx):
            world[shell_idx] = LIVE if pos in live_set else BLANK
        worlds.append(tuple(world))
    return tuple(sorted(worlds))


def _shell_count(s: State) -> int:
    return len(s.worlds[0]) if s.worlds else 0


def _counts(worlds) -> Tuple[int, int]:
    """(live, blank) remaining — identical across all candidate worlds."""
    w = worlds[0]
    lives = sum(1 for sh in w if sh == LIVE)
    return lives, len(w) - lives


def _split_at(worlds, idx) -> Dict[int, Tuple[Tuple[int, ...], ...]]:
    """Partition worlds by the shell type at position idx."""
    out: Dict[int, list] = {}
    for w in worlds:
        out.setdefault(w[idx], []).append(w)
    return {t: tuple(ws) for t, ws in out.items()}


def _consume_front(s: State, worlds, saw: bool) -> State:
    """Remove the chambered shell (worlds already filtered to one type at
    position 0, so slicing keeps them sorted).  saw is passed explicitly:
    every fired shot resets it, a beer eject preserves it."""
    return State(s.php, s.ehp, s.max_hp, s.pitems, s.eitems,
                 tuple(w[1:] for w in worlds), s.mem[1:],
                 saw, s.e_cuff, s.p_cuff)


def _toggle_front(s: State) -> State:
    return State(s.php, s.ehp, s.max_hp, s.pitems, s.eitems,
                 tuple(sorted((3 - w[0],) + w[1:] for w in s.worlds)),
                 s.mem, s.saw, s.e_cuff, s.p_cuff)


def _probs(s: State) -> Tuple[float, float]:
    """(P(current is live), P(current is blank)) under the player's belief."""
    if not s.worlds or not s.worlds[0]:
        return 0.0, 0.0
    total = len(s.worlds)
    live = sum(1 for w in s.worlds if w[0] == LIVE)
    return live / total, 1.0 - live / total


def _known_at(s: State, idx: int) -> bool:
    first = s.worlds[0][idx]
    return all(w[idx] == first for w in s.worlds)


# ── Value helpers ─────────────────────────────────────────────────────────────
def _better(a: Value, b: Value) -> bool:
    if a[0] > b[0] + P_EPS:
        return True
    if b[0] > a[0] + P_EPS:
        return False
    return a[1] > b[1]


def _econ_tb(s: State) -> float:
    own = sum(w * c for w, c in zip(ITEM_W, s.pitems))
    foe = sum(w * c for w, c in zip(ITEM_W, s.eitems))
    return own - ENEMY_ITEM_TB * foe


def _win_value(s: State) -> Value:
    return 1.0, _econ_tb(s) + HP_TB * s.php


def _loss_value(s: State) -> Value:
    return 0.0, -HP_TB * s.ehp


def _reload_value(s: State) -> Value:
    """Heuristic frontier: shells ran out, a fresh unknown load begins.

    Cuffs are removed and the barrel regrows at every reload
    (RoundManager.StartRound -> RemoveAllCuffsRoutine), items persist and
    new ones are dealt, HP persists, and the player always acts first.
    """
    own = sum(w * c for w, c in zip(ITEM_W, s.pitems))
    foe = sum(w * c for w, c in zip(ITEM_W, s.eitems))
    p = 0.5 + 0.055 * (s.php - s.ehp) + 0.012 * (own - foe) + 0.02
    p = min(0.97, max(0.03, p))
    return p, _econ_tb(s) + HP_TB * s.php


def _cutoff_value(s: State, dealer_to_act: bool) -> Value:
    """Horizon estimate for the depth-limited fallback mode.  Clamped away
    from 0.0/1.0 so proven wins/losses stay unmistakable."""
    p, tb = _reload_value(s)
    if dealer_to_act:
        p -= 0.05
    if s.e_cuff == CUFF_ACTIVE:
        p += 0.04
    if s.p_cuff == CUFF_ACTIVE:
        p -= 0.04
    return min(0.97, max(0.03, p)), tb


# ── Shared shot / turn-passing mechanics ─────────────────────────────────────
def _after_shot_phase(s: State) -> str:
    """Phase after a turn-ending shot resolved on state s (health checked
    by the caller).  Health takes precedence over the empty-gun check."""
    return "reload" if _shell_count(s) == 0 else "dealer"


def _phase_value(phase: str, s: State, memo: dict, d: int | None) -> Value:
    if phase == "player":
        return _solve_player(s, memo, d)
    if phase == "dealer":
        return _pass_to_dealer(s, memo, d)
    if phase == "win":
        return _win_value(s)
    if phase == "loss":
        return _loss_value(s)
    return _reload_value(s)


def _pass_to_dealer(s: State, memo: dict, d: int | None) -> Value:
    # EndTurn(false): cuffed dealer skips exactly one turn, breaks free at
    # the start of the next one (DealerCheckHandCuffs two-visit mechanic).
    if s.e_cuff == CUFF_ACTIVE:
        return _pass_to_player(s._replace(e_cuff=CUFF_COOLDOWN), memo, d)
    if s.e_cuff == CUFF_COOLDOWN:
        s = s._replace(e_cuff=CUFF_NONE)
    return _dealer_turn(s, memo, d)


def _pass_to_player(s: State, memo: dict, d: int | None) -> Value:
    if s.p_cuff == CUFF_ACTIVE:
        return _pass_to_dealer(s._replace(p_cuff=CUFF_COOLDOWN), memo, d)
    if s.p_cuff == CUFF_COOLDOWN:
        s = s._replace(p_cuff=CUFF_NONE)
    return _solve_player(s, memo, d)


# ── Player actions ────────────────────────────────────────────────────────────
def _item_usable_by_player(s: State, idx: int) -> bool:
    """Legality + safe dominance pruning for one player-side item use."""
    n = _shell_count(s)
    if idx == GLASS:
        return not _known_at(s, 0)
    if idx == PILLS:
        return s.php < s.max_hp          # at full HP pills are a pure loss
    if idx == CIGGS:
        return s.php < s.max_hp
    if idx == PHONE:
        return n >= 2 and any(not _known_at(s, i) for i in range(1, n))
    if idx == CUFFS:
        # Blocked while the dealer is cuffed OR on cuff cooldown (the game
        # keeps dealerCuffed true until he breaks free).  Cuffing with one
        # shell left is a pure waste: cuffs vanish at the reload.
        return s.e_cuff == CUFF_NONE and n >= 2
    if idx == SAW:
        return not s.saw
    if idx == BEER:
        return True                       # ejecting the last shell is legal
    if idx == INVERTER:
        return True
    return False


def _adrenaline_targets(s: State) -> List[int]:
    if s.pitems[ADRENALINE] <= 0:
        return []
    return [
        idx for idx in range(9)
        if idx != ADRENALINE and s.eitems[idx] > 0 and _item_usable_by_player(s, idx)
    ]


def _player_actions(s: State) -> List[Tuple[str, int]]:
    p_live, p_blank = _probs(s)
    acts: List[Tuple[str, int]] = [("shoot_opp", -1)]
    if p_live < 1.0 - P_EPS:
        # Self-shot on a confirmed live shell is strictly dominated.
        acts.append(("shoot_self", -1))
    for idx in range(9):
        if s.pitems[idx] <= 0 or idx == ADRENALINE:
            continue
        if _item_usable_by_player(s, idx):
            acts.append(("item", idx))
    for t in _adrenaline_targets(s):
        acts.append(("steal", t))
    return acts


def _dec(items: Tuple[int, ...], idx: int) -> Tuple[int, ...]:
    return items[:idx] + (items[idx] - 1,) + items[idx + 1:]


def _apply_item_effect(s: State, idx: int) -> List[Tuple[float, State, str]]:
    """Effect of an item used by the player (item already paid for)."""
    total = len(s.worlds)

    if idx == GLASS:
        return [
            (len(ws) / total, s._replace(worlds=ws), "player")
            for ws in _split_at(s.worlds, 0).values()
        ]

    if idx == PILLS:
        good = s._replace(php=min(s.max_hp, s.php + 2))
        bad  = s._replace(php=s.php - 1)
        out: List[Tuple[float, State, str]] = [(0.5, good, "player")]
        out.append((0.5, bad, "loss" if bad.php <= 0 else "player"))
        return out

    if idx == CIGGS:
        return [(1.0, s._replace(php=min(s.max_hp, s.php + 1)), "player")]

    if idx == PHONE:
        # BurnerPhone.gd: randi_range(1, n-1); with a full 8-shell load the
        # result 7 is remapped to 6 (position 8 is never revealed, position
        # 7 is twice as likely).
        n = _shell_count(s)
        weights: Dict[int, float] = {}
        base = 1.0 / (n - 1)
        for r in range(1, n):
            pos = r - 1 if (r == 7 and n == 8) else r
            weights[pos] = weights.get(pos, 0.0) + base
        merged: Dict[State, float] = {}
        for pos, wprob in weights.items():
            for ws in _split_at(s.worlds, pos).values():
                ns = s._replace(worlds=ws)
                merged[ns] = merged.get(ns, 0.0) + wprob * len(ws) / total
        return [(p, ns, "player") for ns, p in merged.items()]

    if idx == CUFFS:
        return [(1.0, s._replace(e_cuff=CUFF_ACTIVE), "player")]

    if idx == SAW:
        return [(1.0, s._replace(saw=True), "player")]

    if idx == BEER:
        out = []
        for ws in _split_at(s.worlds, 0).values():
            ns = _consume_front(s, ws, s.saw)   # saw survives a beer eject
            phase = "reload" if _shell_count(ns) == 0 else "player"
            out.append((len(ws) / total, ns, phase))
        return out

    if idx == INVERTER:
        return [(1.0, _toggle_front(s), "player")]

    return [(1.0, s, "player")]


def _apply_player(s: State, act: str, idx: int) -> List[Branch]:
    if act == "shoot_opp":
        dmg = 2 if s.saw else 1
        total = len(s.worlds)
        out: List[Branch] = []
        for stype, ws in _split_at(s.worlds, 0).items():
            ns = _consume_front(s, ws, False)
            prob = len(ws) / total
            if stype == LIVE:
                ns = ns._replace(ehp=max(0, ns.ehp - dmg))
                if ns.ehp == 0:
                    out.append((prob, ns, "win"))
                    continue
            out.append((prob, ns, _after_shot_phase(ns)))
        return out

    if act == "shoot_self":
        dmg = 2 if s.saw else 1
        total = len(s.worlds)
        out = []
        for stype, ws in _split_at(s.worlds, 0).items():
            ns = _consume_front(s, ws, False)
            prob = len(ws) / total
            if stype == LIVE:
                ns = ns._replace(php=max(0, ns.php - dmg))
                if ns.php == 0:
                    out.append((prob, ns, "loss"))
                    continue
                out.append((prob, ns, _after_shot_phase(ns)))
            else:
                # Blank into yourself: free extra turn.
                phase = "reload" if _shell_count(ns) == 0 else "player"
                out.append((prob, ns, phase))
        return out

    if act == "item":
        return _apply_item_effect(s._replace(pitems=_dec(s.pitems, idx)), idx)

    # act == "steal": adrenaline + immediate use of the dealer's item.
    ns = s._replace(pitems=_dec(s.pitems, ADRENALINE), eitems=_dec(s.eitems, idx))
    return _apply_item_effect(ns, idx)


# ── Player search ─────────────────────────────────────────────────────────────
_ITEM_ORDER = {GLASS: 2, PHONE: 3, INVERTER: 4, SAW: 5, CUFFS: 6,
               BEER: 8, CIGGS: 9, PILLS: 10}


def _ordered_actions(s: State) -> List[Tuple[str, int]]:
    """Try likely-strong actions first so the upper-bound cut fires early."""
    p_live, p_blank = _probs(s)

    def order(a: Tuple[str, int]) -> int:
        act, idx = a
        if act == "shoot_opp":
            return 0 if p_live > 0.6 else 12
        if act == "shoot_self":
            return 1 if p_blank > 0.8 else 13
        base = _ITEM_ORDER.get(idx, 7)
        return base + 2 if act == "steal" else base

    acts = _player_actions(s)
    acts.sort(key=order)
    return acts


def _best_action(s: State, memo: dict, d: int | None) -> Tuple[str, int, Value]:
    """Exact max over player actions.  An action is abandoned early once
    even 100% wins on its remaining probability mass cannot beat the best
    found so far (sound: only strictly worse actions are dropped)."""
    n0 = _shell_count(s)
    best_v: Value | None = None
    best_a = ("shoot_opp", -1)
    for act, idx in _ordered_actions(s):
        p = tb = 0.0
        rem = 1.0
        abandoned = False
        for prob, ns, phase in _apply_player(s, act, idx):
            dd = d if (d is None or _shell_count(ns) == n0) else d - 1
            v = _phase_value(phase, ns, memo, dd)
            p += prob * v[0]
            tb += prob * v[1]
            rem -= prob
            if best_v is not None and p + rem < best_v[0] - P_EPS:
                abandoned = True
                break
        if abandoned:
            continue
        v = (p, tb)
        if best_v is None or _better(v, best_v):
            best_v, best_a = v, (act, idx)
    return best_a[0], best_a[1], best_v


def _action_values(s: State, memo: dict,
                   d: int | None = None) -> List[Tuple[str, int, Value]]:
    """Un-pruned per-action values (diagnostics/tests)."""
    n0 = _shell_count(s)
    out = []
    for act, idx in _player_actions(s):
        p = tb = 0.0
        for prob, ns, phase in _apply_player(s, act, idx):
            dd = d if (d is None or _shell_count(ns) == n0) else d - 1
            v = _phase_value(phase, ns, memo, dd)
            p  += prob * v[0]
            tb += prob * v[1]
        out.append((act, idx, (p, tb)))
    return out


def _solve_player(s: State, memo: dict, d: int | None) -> Value:
    if s.php <= 0:
        return _loss_value(s)
    if s.ehp <= 0:
        return _win_value(s)
    if _shell_count(s) == 0:
        return _reload_value(s)
    if d is not None and d <= 0:
        return _cutoff_value(s, dealer_to_act=False)

    key = ("P", s, d)
    hit = memo.get(key)
    if hit is not None:
        return hit

    _act, _idx, best = _best_action(s, memo, d)
    val = (best[0], best[1] - STEP_COST)
    if len(memo) >= _memo_cap:
        raise _SearchOverflow
    memo[key] = val
    return val


# ── Dealer policy (faithful port of DealerIntelligence.gd, endless mode) ─────
def _dealer_figures_out(world: Tuple[int, ...], mem: Tuple[bool, ...]) -> bool:
    """FigureOutShell(): can the dealer deduce the chambered shell?"""
    if mem[0]:
        return True
    lives = sum(1 for sh in world if sh == LIVE)
    blanks = len(world) - lives
    if lives == 0 or blanks == 0:
        return True
    for i, known in enumerate(mem):
        if known:
            if world[i] == LIVE:
                lives -= 1
            else:
                blanks -= 1
    return lives == 0 or blanks == 0


def _dealer_turn(s: State, memo: dict, d: int | None) -> Value:
    """One full dealer turn (BeginDealerTurn: fresh per-turn locals)."""
    return _dealer_iter(s, memo, False, 0, 0, False, d)


def _dealer_iter(s: State, memo: dict, knows: bool, known: int,
                 target: int, used_med: bool, d: int | None) -> Value:
    """One DealerChoice() iteration (the game recurses here after each item).

    Memoized together with the turn-local variables: different item orders
    inside one dealer turn converge to the same (state, locals) pairs.
    """
    if s.php <= 0:
        return _loss_value(s)
    if s.ehp <= 0:
        return _win_value(s)
    if _shell_count(s) == 0:
        return _reload_value(s)
    if d is not None and d <= 0:
        return _cutoff_value(s, dealer_to_act=True)

    key = ("D", s, knows, known, target, used_med, d)
    hit = memo.get(key)
    if hit is not None:
        return hit
    val = _dealer_iter_inner(s, memo, knows, known, target, used_med, d)
    if len(memo) >= _memo_cap:
        raise _SearchOverflow
    memo[key] = val
    return val


def _dealer_iter_inner(s: State, memo: dict, knows: bool, known: int,
                       target: int, used_med: bool, d: int | None) -> Value:
    # FigureOutShell — endless-mode inference (DealerIntelligence.gd:96-104).
    if not knows:
        groups: Dict[Tuple[bool, int], list] = {}
        for w in s.worlds:
            if _dealer_figures_out(w, s.mem):
                groups.setdefault((True, w[0]), []).append(w)
            else:
                groups.setdefault((False, 0), []).append(w)
        if len(groups) > 1 or (True, LIVE) in groups or (True, BLANK) in groups:
            total = len(s.worlds)
            p = tb = 0.0
            for (fos, val), ws in groups.items():
                ns = s._replace(worlds=tuple(sorted(ws)))
                if fos:
                    tgt = 2 if val == LIVE else 1     # 2=player, 1=self
                    v = _dealer_items(ns, memo, True, val, tgt, used_med, d)
                else:
                    v = _dealer_items(ns, memo, False, 0, target, used_med, d)
                p  += len(ws) / total * v[0]
                tb += len(ws) / total * v[1]
            return p, tb
    return _dealer_items(s, memo, knows, known, target, used_med, d)


def _dealer_items(s: State, memo: dict, knows: bool, known: int,
                  target: int, used_med: bool, d: int | None) -> Value:
    n = _shell_count(s)

    # Last-shell rule (DealerIntelligence.gd:106-112): with one shell left
    # the dealer always knows it.
    if n == 1 and not knows:
        total = len(s.worlds)
        p = tb = 0.0
        for stype, ws in _split_at(s.worlds, 0).items():
            ns = s._replace(worlds=ws)
            tgt = 2 if stype == LIVE else 1
            v = _dealer_items(ns, memo, True, stype, tgt, used_med, d)
            p  += len(ws) / total * v[0]
            tb += len(ws) / total * v[1]
        return p, tb

    own = _dealer_eligible(s, s.eitems, knows, known, used_med)
    if own:
        total = sum(c for _, c in own)
        p = tb = 0.0
        for idx, cnt in own:
            ns = s._replace(eitems=_dec(s.eitems, idx))
            v = _dealer_use(ns, memo, idx, knows, known, target, used_med, d)
            p  += cnt / total * v[0]
            tb += cnt / total * v[1]
        return p, tb

    if s.eitems[ADRENALINE] > 0:
        steal = _dealer_eligible(s, s.pitems, knows, known, used_med)
        if steal:
            total = sum(c for _, c in steal)
            p = tb = 0.0
            for idx, cnt in steal:
                ns = s._replace(
                    eitems=_dec(s.eitems, ADRENALINE),
                    pitems=_dec(s.pitems, idx),
                )
                v = _dealer_use(ns, memo, idx, knows, known, target, used_med, d)
                p  += cnt / total * v[0]
                tb += cnt / total * v[1]
            return p, tb

    return _dealer_bonus_saw(s, memo, knows, known, target, used_med, d)


def _dealer_eligible(s: State, items: Tuple[int, ...], knows: bool,
                     known: int, used_med: bool) -> List[Tuple[int, int]]:
    """(item, count) pairs whose trigger condition currently passes,
    evaluated with the DEALER's knowledge (DealerIntelligence.gd:151-201)."""
    n = _shell_count(s)
    has_cigs = s.eitems[CIGGS] > 0
    out: List[Tuple[int, int]] = []
    for idx in range(9):
        cnt = items[idx]
        if cnt <= 0:
            continue
        ok = False
        if idx == GLASS:
            ok = not knows and n != 1
        elif idx == CIGGS:
            ok = s.ehp < s.max_hp
        elif idx == PILLS:
            ok = (s.ehp < s.max_hp and not has_cigs and not used_med
                  and s.ehp != 1)
        elif idx == BEER:
            ok = known != LIVE and n != 1         # unknown shells qualify too
        elif idx == CUFFS:
            ok = s.p_cuff == CUFF_NONE and n != 1
        elif idx == SAW:
            ok = not s.saw and known == LIVE
        elif idx == PHONE:
            ok = n > 2
        elif idx == INVERTER:
            ok = knows and known == BLANK
        if ok:
            out.append((idx, cnt))
    return out


def _dealer_use(s: State, memo: dict, idx: int, knows: bool, known: int,
                target: int, used_med: bool, d: int | None) -> Value:
    """Apply one dealer item (already paid for) and continue the turn."""
    total = len(s.worlds)

    if idx == GLASS:
        p = tb = 0.0
        for stype, ws in _split_at(s.worlds, 0).items():
            ns = s._replace(worlds=ws)
            tgt = 2 if stype == LIVE else 1
            v = _dealer_iter(ns, memo, True, stype, tgt, used_med, d)
            p  += len(ws) / total * v[0]
            tb += len(ws) / total * v[1]
        return p, tb

    if idx == CIGGS:
        ns = s._replace(ehp=min(s.max_hp, s.ehp + 1))
        return _dealer_iter(ns, memo, knows, known, target, used_med, d)

    if idx == PILLS:
        good = s._replace(ehp=min(s.max_hp, s.ehp + 2))
        bad  = s._replace(ehp=s.ehp - 1)
        v_good = _dealer_iter(good, memo, knows, known, target, True, d)
        v_bad = _win_value(bad) if bad.ehp <= 0 else \
            _dealer_iter(bad, memo, knows, known, target, True, d)
        return (0.5 * (v_good[0] + v_bad[0]), 0.5 * (v_good[1] + v_bad[1]))

    if idx == BEER:
        # Ejects the chambered shell; in endless mode the dealer FORGETS
        # what he knew about it, but his shooting target is NOT cleared
        # (the stale-target quirk, DealerIntelligence.gd:170-176).
        dd = None if d is None else d - 1
        p = tb = 0.0
        for stype, ws in _split_at(s.worlds, 0).items():
            ns = _consume_front(s, ws, s.saw)
            prob = len(ws) / total
            if _shell_count(ns) == 0:            # unreachable: beer needs n != 1
                v = _reload_value(ns)
            else:
                v = _dealer_iter(ns, memo, False, 0, target, used_med, dd)
            p  += prob * v[0]
            tb += prob * v[1]
        return p, tb

    if idx == CUFFS:
        ns = s._replace(p_cuff=CUFF_ACTIVE)
        return _dealer_iter(ns, memo, knows, known, target, used_med, d)

    if idx == SAW:
        ns = s._replace(saw=True)
        return _dealer_iter(ns, memo, knows, known, target, used_med, d)

    if idx == PHONE:
        # DealerIntelligence.gd:187-194 — uniform future position.
        n = _shell_count(s)
        merged: Dict[Tuple[bool, ...], float] = {}
        base = 1.0 / (n - 1)
        for pos in range(1, n):
            mem = s.mem[:pos] + (True,) + s.mem[pos + 1:]
            merged[mem] = merged.get(mem, 0.0) + base
        p = tb = 0.0
        for mem, prob in merged.items():
            v = _dealer_iter(s._replace(mem=mem), memo, knows, known, target,
                             used_med, d)
            p  += prob * v[0]
            tb += prob * v[1]
        return p, tb

    if idx == INVERTER:
        ns = _toggle_front(s)
        return _dealer_iter(ns, memo, True, LIVE, 2, used_med, d)

    return _dealer_iter(s, memo, knows, known, target, used_med, d)


def _weighted_flip(worlds) -> List[Tuple[float, int]]:
    """CoinFlip() in endless mode: [(prob, result)]; 1 -> player, 0 -> self."""
    lives, blanks = _counts(worlds)
    if lives > blanks:
        return [(1.0, 1)]
    if lives < blanks:
        return [(1.0, 0)]
    return [(0.5, 0), (0.5, 1)]


def _dealer_bonus_saw(s: State, memo: dict, knows: bool, known: int,
                      target: int, used_med: bool, d: int | None) -> Value:
    """Leftover-handsaw coin flip (DealerIntelligence.gd:203-215)."""
    has_saw = s.eitems[SAW] > 0 or (s.eitems[ADRENALINE] > 0 and s.pitems[SAW] > 0)
    if has_saw and not s.saw and known != BLANK:
        p = tb = 0.0
        for prob, res in _weighted_flip(s.worlds):
            if res == 0:
                v = _dealer_shoot(s, memo, knows, known, 1, d)
            else:
                if s.eitems[SAW] > 0:
                    ns = s._replace(eitems=_dec(s.eitems, SAW))
                else:
                    ns = s._replace(
                        eitems=_dec(s.eitems, ADRENALINE),
                        pitems=_dec(s.pitems, SAW),
                    )
                ns = ns._replace(saw=True)
                v = _dealer_iter(ns, memo, knows, known, 2, used_med, d)
            p  += prob * v[0]
            tb += prob * v[1]
        return p, tb
    return _dealer_shoot(s, memo, knows, known, target, d)


def _dealer_shoot(s: State, memo: dict, knows: bool, known: int,
                  target: int, d: int | None) -> Value:
    """Shoot(dealerTarget) — or the count-weighted random choice."""
    if target == 0:
        p = tb = 0.0
        for prob, res in _weighted_flip(s.worlds):
            v = _dealer_shoot(s, memo, knows, known, 1 if res == 0 else 2, d)
            p  += prob * v[0]
            tb += prob * v[1]
        return p, tb

    dmg = 2 if s.saw else 1
    total = len(s.worlds)
    dd = None if d is None else d - 1
    p = tb = 0.0
    for stype, ws in _split_at(s.worlds, 0).items():
        prob = len(ws) / total

        if target == 1:                          # dealer shoots himself
            if stype == BLANK:
                # Free extra turn; the barrel does NOT regrow on this path
                # (EndDealerTurn -> BeginDealerTurn skips RoundManager.EndTurn).
                ns = _consume_front(s, ws, s.saw)
                if _shell_count(ns) == 0:
                    v = _reload_value(ns)
                else:
                    v = _dealer_turn(ns, memo, dd)
                p, tb = p + prob * v[0], tb + prob * v[1]
                continue
            ns = _consume_front(s, ws, False)
            ns = ns._replace(ehp=max(0, ns.ehp - dmg))
            if ns.ehp == 0:
                v = _win_value(ns)
            elif _shell_count(ns) == 0:
                v = _reload_value(ns)
            else:
                v = _pass_to_player(ns, memo, dd)
        else:                                    # dealer shoots the player
            ns = _consume_front(s, ws, False)
            if stype == LIVE:
                ns = ns._replace(php=max(0, ns.php - dmg))
            if ns.php == 0:
                v = _loss_value(ns)
            elif _shell_count(ns) == 0:
                v = _reload_value(ns)
            else:
                v = _pass_to_player(ns, memo, dd)
        p, tb = p + prob * v[0], tb + prob * v[1]
    return p, tb


# ── Plan builder ──────────────────────────────────────────────────────────────
def _rust_state_args(s: State) -> tuple:
    """State -> plain-Python arguments for the bsr_core.Solver methods."""
    return (s.php, s.ehp, s.max_hp, list(s.pitems), list(s.eitems),
            [list(w) for w in s.worlds], list(s.mem),
            s.saw, s.e_cuff, s.p_cuff)


def _search_best_action(s: State, memo, d: int | None) -> Tuple[str, int, Value]:
    """_best_action, dispatched to the Rust solver when `memo` is its
    handle (the Solver keeps its own memo and horizon between calls)."""
    if _bsr is not None and isinstance(memo, _bsr.Solver):
        hit = memo.best_action(*_rust_state_args(s))
        if hit is None:
            raise _SearchOverflow
        act, idx, p, tb = hit
        return act, idx, (p, tb)
    act, idx, val = _best_action(s, memo, d)
    return act, idx, val


def _plan_root_search(state: State):
    """Exact solve when it fits in memory; otherwise iterative deepening
    over consumed shells.  Returns (act, idx, value, memo, horizon) where
    horizon is None for an exact result and memo is either the Python dict
    or the Rust Solver handle."""
    if _bsr is not None and _shell_count(state) <= 8:
        solver = _bsr.Solver()
        hit = solver.root_search(*_rust_state_args(state))
        if hit is None:
            return None
        act, idx, p, tb, horizon = hit
        return act, idx, (p, tb), solver, horizon

    global _memo_cap
    huge = (len(state.worlds) >= FALLBACK_GATE_WORLDS
            and sum(state.pitems) + sum(state.eitems) >= FALLBACK_GATE_ITEMS)
    if not huge:
        _memo_cap = EXACT_MEMO_CAP
        memo: dict = {}
        try:
            act, idx, val = _best_action(state, memo, None)
            return act, idx, val, memo, None
        except _SearchOverflow:
            pass

    _memo_cap = FALLBACK_MEMO_CAP
    result = None
    for h in range(1, _shell_count(state) + 1):
        memo = {}
        try:
            act, idx, val = _best_action(state, memo, h)
            result = (act, idx, val, memo, h)
        except _SearchOverflow:
            break
    return result


def _shell_hint(s: State) -> str:
    p_live, _ = _probs(s)
    if p_live >= 0.99:
        return " (confirmed LIVE)"
    if p_live <= 0.01:
        return " (confirmed BLANK)"
    return f" ({int(round(p_live * 100))}% LIVE)"


def _fmt_action(act: str, idx: int, s: State) -> str:
    p_live, p_blank = _probs(s)
    dmg = 2 if s.saw else 1
    cuff = " — dealer's turn SKIPPED by cuffs (set D CUFF: COOLDOWN)" \
        if s.e_cuff == CUFF_ACTIVE else ""

    if act == "shoot_opp":
        if p_live >= 0.99:
            return f"Shoot DEALER — confirmed LIVE ({dmg} dmg){cuff}"
        if p_live <= 0.01:
            return f"Shoot DEALER — confirmed BLANK (end turn){cuff}"
        return f"Shoot DEALER — {int(round(p_live * 100))}% LIVE ({dmg} dmg if hit){cuff}"

    if act == "shoot_self":
        if p_blank >= 0.99:
            return f"Shoot YOURSELF — confirmed BLANK (free extra turn){cuff}"
        return f"Shoot YOURSELF — {int(round(p_blank * 100))}% BLANK (risky free turn){cuff}"

    hint = _shell_hint(s)
    if act == "steal":
        steal_labels = {
            GLASS:    "Use ADRENALINE — steal dealer's MAGNIFYING GLASS to reveal current shell",
            PILLS:    "Use ADRENALINE — steal dealer's PILLS (50/50: +2 HP or -1 HP)",
            PHONE:    "Use ADRENALINE — steal dealer's PHONE to reveal a random future shell",
            CUFFS:    "Use ADRENALINE — steal dealer's HANDCUFFS to skip their turn (set D CUFF: CUFFED)",
            SAW:      f"Use ADRENALINE — steal dealer's HACKSAW for 2 dmg{hint}",
            CIGGS:    "Use ADRENALINE — steal dealer's CIGARETTES to restore 1 HP",
            BEER:     f"Use ADRENALINE — steal dealer's BEER to eject current shell{hint}",
            INVERTER: f"Use ADRENALINE — steal dealer's INVERTER to flip current shell{hint}",
        }
        return steal_labels.get(idx, "Use ADRENALINE")

    labels = {
        GLASS:      "Use MAGNIFYING GLASS — reveal current shell",
        PILLS:      "Use PILLS — 50/50: +2 HP or -1 HP",
        PHONE:      "Use PHONE — reveal a random future shell",
        CUFFS:      "Use HANDCUFFS — dealer loses their next turn (set D CUFF: CUFFED)",
        SAW:        f"Use HACKSAW — next shot deals 2 dmg{hint}",
        CIGGS:      "Use CIGARETTES — restore 1 HP",
        BEER:       f"Use BEER — eject current shell{hint}",
        INVERTER:   f"Use INVERTER — flip current shell{hint}",
    }
    return labels.get(idx, f"Use item {idx}")


def _update_prompt_for_branch(idx: int) -> str:
    if idx == PHONE:
        return "Mark the revealed future shell in overlay — press AI again"
    if idx == GLASS:
        return "Mark the revealed current shell in overlay if needed — press AI again"
    if idx == BEER:
        return "Update the shell sequence in overlay — press AI again"
    if idx == PILLS:
        return "Update HP in overlay with the result — press AI again"
    return "Update overlay with the new info — press AI again"


def build_plan(state: State) -> List[str]:
    root = _plan_root_search(state)
    if root is None:
        return ["Position too complex to search — make your move and press AI again"]
    act, idx, val, memo, horizon = root

    tasks: List[str] = []
    cur = state

    for step in range(MAX_PLAN_STEPS):
        if cur.php <= 0 or cur.ehp <= 0 or _shell_count(cur) == 0:
            break

        if step > 0:
            try:
                act, idx, val = _search_best_action(cur, memo, horizon)
            except _SearchOverflow:
                break
        tasks.append(_fmt_action(act, idx, cur))

        outcomes = _apply_player(cur, act, idx)

        def _returns_to_player(nxt: State, phase: str) -> bool:
            if phase == "player":
                return True
            # A cuffed dealer's turn is skipped and control bounces back.
            return (phase == "dealer" and nxt.e_cuff == CUFF_ACTIVE
                    and nxt.p_cuff == CUFF_NONE and _shell_count(nxt) > 0)

        if len(outcomes) > 1:
            if act in ("item", "steal"):
                tasks.append(_update_prompt_for_branch(idx))
            else:
                if all(_returns_to_player(nxt, phase) for _, nxt, phase in outcomes):
                    tasks.append(
                        "After that shot update overlay — press AI again for your next turn"
                    )
            break

        _, nxt, phase = outcomes[0]
        if phase == "player":
            cur = nxt
            continue
        if _returns_to_player(nxt, phase):
            cur = nxt._replace(e_cuff=CUFF_COOLDOWN)
            continue
        break

    if not tasks:
        tasks.append("No beneficial action found — assess the situation manually")
    return tasks


# ── Public API ────────────────────────────────────────────────────────────────
def run_ai(player_hp: int, enemy_hp: int,
           player_items: list, enemy_items: list,
           shells: list, live_count: int, blank_count: int,
           max_hp: int = DEFAULT_MAX_HP,
           e_cuffed: bool = False,
           saw_active: bool = False,
           cuff_state: int | None = None) -> List[str]:
    """cuff_state: 0 = dealer free, 1 = CUFFED (his next turn is skipped),
    2 = COOLDOWN (skip already consumed, re-cuffing not allowed yet).
    Falls back to the legacy boolean e_cuffed when omitted."""
    if player_hp <= 0:
        return ["You are dead — game over"]
    if enemy_hp <= 0:
        return ["Dealer HP is 0 — round should already be over"]
    if not shells:
        return ["No shells loaded — update shell count first"]
    if live_count + blank_count == 0:
        return ["Shell counts are both 0 — update LIVE / BLANK counts"]

    worlds = _init_worlds(list(shells), live_count, blank_count)
    if not worlds:
        return ["Shell knowledge is inconsistent with LIVE / BLANK counts"]

    if cuff_state is None:
        cuff_state = CUFF_ACTIVE if e_cuffed else CUFF_NONE

    state = State(
        php=player_hp, ehp=enemy_hp, max_hp=max_hp,
        pitems=tuple(player_items), eitems=tuple(enemy_items),
        worlds=worlds,
        mem=(False,) * len(shells),
        saw=saw_active,
        e_cuff=cuff_state,
        p_cuff=CUFF_NONE,
    )

    sys.setrecursionlimit(20000)
    return build_plan(state)
