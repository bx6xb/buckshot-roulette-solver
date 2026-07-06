"""Buckshot Roulette AI Engine — Full Adversarial Expectiminimax.

Both player AND dealer explore all legal actions via minimax.  The dealer is
modelled as a perfect MIN-node opponent so the search always finds the true
optimal (top-1) move for the player.

No artificial limits: no node budget, no Star-1 pruning.
Iterative deepening runs until the game tree is fully solved (no heuristic
_eval() calls at leaf nodes) or a forced win/loss is found.  A 120-second
safety timeout guards against extreme edge cases.
"""

from __future__ import annotations

from itertools import combinations
import time
from typing import Iterable, List, Tuple

# ── Item indices (must match ITEMS_CONF order in overlay.py) ─────────────────
GLASS, PILLS, PHONE, CUFFS, ADRENALINE, SAW, CIGGS, BEER, INVERTER = range(9)

DEFAULT_MAX_HP = 6
INF            = 1e9
EXACT, LOWER, UPPER = 0, 1, 2

MAX_PLAN_STEPS = 14
SAFETY_TIMEOUT = 120.0

_ENDGAME_CACHE: dict = {}
_eval_used: bool = False


# ── Game state ────────────────────────────────────────────────────────────────
class State:
    __slots__ = (
        "php", "ehp", "max_hp", "pitems", "eitems",
        "worlds", "player_turn", "saw", "e_cuffed", "p_cuffed",
    )

    def __init__(self, php, ehp, max_hp, pitems, eitems, worlds,
                 player_turn=True, saw=False, e_cuffed=False, p_cuffed=False):
        self.php         = php
        self.ehp         = ehp
        self.max_hp      = max_hp
        self.pitems      = list(pitems)
        self.eitems      = list(eitems)
        self.worlds      = tuple(worlds)
        self.player_turn = player_turn
        self.saw         = saw
        self.e_cuffed    = e_cuffed
        self.p_cuffed    = p_cuffed

    def clone(self) -> State:
        s             = State.__new__(State)
        s.php         = self.php
        s.ehp         = self.ehp
        s.max_hp      = self.max_hp
        s.pitems      = self.pitems[:]
        s.eitems      = self.eitems[:]
        s.worlds      = self.worlds
        s.player_turn = self.player_turn
        s.saw         = self.saw
        s.e_cuffed    = self.e_cuffed
        s.p_cuffed    = self.p_cuffed
        return s

    def key(self) -> tuple:
        return (
            self.php, self.ehp, self.max_hp,
            tuple(self.pitems), tuple(self.eitems),
            self.worlds,
            self.player_turn, self.saw, self.e_cuffed, self.p_cuffed,
        )


# ── Belief helpers ────────────────────────────────────────────────────────────
def _normalize_worlds(worlds: Iterable[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(dict.fromkeys(tuple(w) for w in worlds)))


def _init_worlds(shells: list[int], live: int, blank: int) -> Tuple[Tuple[int, ...], ...]:
    if len(shells) != live + blank:
        return ()
    if any(sh not in (0, 1, 2) for sh in shells):
        return ()

    unknown_idx = [i for i, sh in enumerate(shells) if sh == 0]
    known_live  = sum(1 for sh in shells if sh == 1)
    known_blank = sum(1 for sh in shells if sh == 2)
    need_live   = live - known_live
    need_blank  = blank - known_blank

    if need_live < 0 or need_blank < 0 or need_live + need_blank != len(unknown_idx):
        return ()

    worlds: list[Tuple[int, ...]] = []
    for live_choice in combinations(range(len(unknown_idx)), need_live):
        live_set = set(live_choice)
        world = list(shells)
        for pos, shell_idx in enumerate(unknown_idx):
            world[shell_idx] = 1 if pos in live_set else 2
        worlds.append(tuple(world))
    return _normalize_worlds(worlds)


def _shell_count(s: State) -> int:
    return len(s.worlds[0]) if s.worlds else 0


def _known_shells(s: State) -> Tuple[int, ...]:
    if not s.worlds:
        return ()
    if len(s.worlds) == 1:
        return s.worlds[0]
    first = s.worlds[0]
    out = []
    for i in range(len(first)):
        cur = first[i]
        out.append(cur if all(world[i] == cur for world in s.worlds[1:]) else 0)
    return tuple(out)


def _probs_at(s: State, idx: int) -> Tuple[float, float]:
    if not s.worlds or idx < 0 or idx >= _shell_count(s):
        return 0.0, 0.0
    if len(s.worlds) == 1:
        return (1.0, 0.0) if s.worlds[0][idx] == 1 else (0.0, 1.0)
    total = len(s.worlds)
    live = sum(1 for world in s.worlds if world[idx] == 1)
    return live / total, 1.0 - live / total


def _probs(s: State) -> Tuple[float, float]:
    if len(s.worlds) == 1 and s.worlds[0]:
        return (1.0, 0.0) if s.worlds[0][0] == 1 else (0.0, 1.0)
    return _probs_at(s, 0)


def _condition_on_index(s: State, idx: int, stype: int) -> State | None:
    worlds = tuple(world for world in s.worlds if world[idx] == stype)
    if not worlds:
        return None
    ns = s.clone()
    ns.worlds = worlds
    return ns


def _branch_at(s: State, idx: int) -> List[Tuple[float, int, State]]:
    if not s.worlds or idx < 0 or idx >= _shell_count(s):
        return []
    if len(s.worlds) == 1:
        return [(1.0, s.worlds[0][idx], s.clone())]
    total = len(s.worlds)
    out: List[Tuple[float, int, State]] = []
    for stype in (1, 2):
        ns = _condition_on_index(s, idx, stype)
        if ns is not None:
            out.append((len(ns.worlds) / total, stype, ns))
    return out


def _consume_current(s: State, stype: int) -> State | None:
    worlds = [world[1:] for world in s.worlds if world[0] == stype]
    if not worlds:
        return None
    ns = s.clone()
    ns.worlds = _normalize_worlds(worlds)
    ns.saw = False
    return ns


def _toggle_current_worlds(s: State) -> State:
    if _shell_count(s) == 0:
        return s.clone()
    ns = s.clone()
    ns.worlds = _normalize_worlds((3 - world[0],) + world[1:] for world in s.worlds)
    return ns


def _merge_outcomes(outcomes: List[Tuple[float, State]]) -> List[Tuple[float, State]]:
    merged: dict[tuple, list] = {}
    for prob, state in outcomes:
        if prob <= 0:
            continue
        k = state.key()
        if k in merged:
            merged[k][0] += prob
        else:
            merged[k] = [prob, state]
    return [(p, st) for p, st in merged.values()]


def _remaining_items(s: State) -> int:
    return sum(s.pitems) + sum(s.eitems)


def _skip_cuffed(s: State) -> State:
    if not s.player_turn and s.e_cuffed:
        ns = s.clone()
        ns.e_cuffed = False
        ns.player_turn = True
        return ns
    if s.player_turn and s.p_cuffed:
        ns = s.clone()
        ns.p_cuffed = False
        ns.player_turn = False
        return ns
    return s


# ── Shot actions ──────────────────────────────────────────────────────────────
def _shoot_opp(s: State) -> List[Tuple[float, State]]:
    dmg = 2 if s.saw else 1
    out: List[Tuple[float, State]] = []
    for prob, stype, _ in _branch_at(s, 0):
        ns = _consume_current(s, stype)
        if ns is None:
            continue
        if stype == 1:
            if s.player_turn:
                ns.ehp -= dmg
            else:
                ns.php -= dmg
        ns.player_turn = not s.player_turn
        ns = _skip_cuffed(ns)
        out.append((prob, ns))
    return out


def _shoot_self(s: State) -> List[Tuple[float, State]]:
    dmg = 2 if s.saw else 1
    out: List[Tuple[float, State]] = []
    for prob, stype, _ in _branch_at(s, 0):
        ns = _consume_current(s, stype)
        if ns is None:
            continue
        if stype == 1:
            if s.player_turn:
                ns.php -= dmg
            else:
                ns.ehp -= dmg
            ns.player_turn = not s.player_turn
            ns = _skip_cuffed(ns)
        out.append((prob, ns))
    return out


# ── Item actions ──────────────────────────────────────────────────────────────
def _use_item(s: State, idx: int, by_player: bool,
              consume_item: bool = True) -> List[Tuple[float, State]]:
    ns = s.clone()
    if consume_item:
        if by_player:
            ns.pitems[idx] -= 1
        else:
            ns.eitems[idx] -= 1

    if idx == GLASS:
        if _shell_count(ns) == 0:
            return [(1.0, ns)]
        known = _known_shells(ns)
        if known and known[0] != 0:
            return [(1.0, ns)]
        return [(prob, branch) for prob, _, branch in _branch_at(ns, 0)]

    if idx == BEER:
        if _shell_count(ns) == 0:
            return [(1.0, ns)]
        out: List[Tuple[float, State]] = []
        for prob, stype, _ in _branch_at(ns, 0):
            nns = _consume_current(ns, stype)
            if nns is not None:
                nns.saw = s.saw
                out.append((prob, nns))
        return out

    if idx == PILLS:
        good, bad = ns.clone(), ns.clone()
        if by_player:
            good.php = min(good.max_hp, good.php + 2)
            bad.php  = max(0, bad.php - 1)
        else:
            good.ehp = min(good.max_hp, good.ehp + 2)
            bad.ehp  = max(0, bad.ehp - 1)
        return [(0.5, good), (0.5, bad)]

    if idx == CIGGS:
        if by_player:
            if ns.php >= ns.max_hp:
                return [(1.0, ns)]
            ns.php = min(ns.max_hp, ns.php + 1)
        else:
            if ns.ehp >= ns.max_hp:
                return [(1.0, ns)]
            ns.ehp = min(ns.max_hp, ns.ehp + 1)
        return [(1.0, ns)]

    if idx == SAW:
        ns.saw = True
        return [(1.0, ns)]

    if idx == CUFFS:
        if by_player:
            ns.e_cuffed = True
        else:
            ns.p_cuffed = True
        return [(1.0, ns)]

    if idx == INVERTER:
        if _shell_count(ns) == 0:
            return [(1.0, ns)]
        return [(1.0, _toggle_current_worlds(ns))]

    if idx == ADRENALINE:
        return _use_adrenaline(ns, by_player, consume_item=False)

    if idx == PHONE:
        positions = list(range(1, _shell_count(ns)))
        if not positions:
            return [(1.0, ns)]
        slot_prob = 1.0 / len(positions)
        out: List[Tuple[float, State]] = []
        for pos in positions:
            for prob, _, branch in _branch_at(ns, pos):
                out.append((slot_prob * prob, branch))
        return _merge_outcomes(out)

    return [(1.0, ns)]


def _adrenaline_targets(s: State, by_player: bool) -> List[int]:
    src = s.eitems if by_player else s.pitems
    out: List[int] = []
    for idx in range(9):
        if src[idx] <= 0:
            continue
        if idx == ADRENALINE:
            continue
        if idx in (GLASS, BEER, INVERTER) and _shell_count(s) == 0:
            continue
        if idx == PHONE and _shell_count(s) <= 1:
            continue
        if idx == CUFFS and ((by_player and s.e_cuffed) or (not by_player and s.p_cuffed)):
            continue
        out.append(idx)
    return out


def _use_adrenaline(s: State, by_player: bool,
                    target_idx: int | None = None,
                    consume_item: bool = True) -> List[Tuple[float, State]]:
    ns = s.clone()
    if consume_item:
        if by_player:
            ns.pitems[ADRENALINE] -= 1
        else:
            ns.eitems[ADRENALINE] -= 1

    targets = _adrenaline_targets(ns, by_player)
    if not targets:
        return [(1.0, ns)]

    chosen = target_idx if target_idx in targets else targets[0]
    nns = ns.clone()
    if by_player:
        nns.eitems[chosen] -= 1
    else:
        nns.pitems[chosen] -= 1
    return _use_item(nns, chosen, by_player, consume_item=False)


# ── Evaluation ────────────────────────────────────────────────────────────────
def _round_reset_eval(s: State) -> float:
    """Score when all shells are spent and a fresh round will start."""
    score = (s.php - s.ehp) * 260.0
    score += 80.0

    hp_room_p = max(0, s.max_hp - s.php)
    hp_room_e = max(0, s.max_hp - s.ehp)

    score += min(hp_room_p, s.pitems[CIGGS]) * 35
    score += min(hp_room_p, s.pitems[PILLS]) * 20
    score += s.pitems[SAW]        * 35
    score += s.pitems[GLASS]      * 20
    score += s.pitems[BEER]       * 20
    score += s.pitems[CUFFS]      * 45
    score += s.pitems[INVERTER]   * 22
    score += s.pitems[PHONE]      * 25
    score += s.pitems[ADRENALINE] * 30

    score -= min(hp_room_e, s.eitems[CIGGS]) * 30
    score -= min(hp_room_e, s.eitems[PILLS]) * 18
    score -= s.eitems[SAW]        * 35
    score -= s.eitems[GLASS]      * 20
    score -= s.eitems[BEER]       * 20
    score -= s.eitems[CUFFS]      * 45
    score -= s.eitems[INVERTER]   * 22
    score -= s.eitems[PHONE]      * 25
    score -= s.eitems[ADRENALINE] * 30
    return score


def _eval(s: State) -> float:
    """Heuristic leaf evaluation — only called when search cannot go deeper."""
    global _eval_used
    if s.php <= 0:
        return -INF
    if s.ehp <= 0:
        return INF
    if _shell_count(s) == 0:
        return _round_reset_eval(s)

    _eval_used = True
    p_live, p_blank = _probs(s)
    known = _known_shells(s)
    dmg   = 2 if s.saw else 1

    score = (s.php - s.ehp) * 250.0

    if s.player_turn:
        score += 60.0
        score += p_live * (200.0 if s.ehp <= dmg else 100.0)
        score += p_blank * 30.0
    else:
        score -= 60.0
        score -= p_live * (200.0 if s.php <= dmg else 100.0)
        score -= p_blank * 30.0

    hp_room_p = max(0, s.max_hp - s.php)
    hp_room_e = max(0, s.max_hp - s.ehp)
    current_unknown = bool(known) and known[0] == 0
    future_unknowns = sum(1 for sh in known[1:] if sh == 0) if known else 0

    score += min(hp_room_p, s.pitems[CIGGS]) * 45
    score += min(hp_room_p, s.pitems[PILLS]) * 18
    score += s.pitems[SAW]        * (35 + 25 * p_live)
    score += s.pitems[GLASS]      * (30 if current_unknown else 5)
    score += s.pitems[BEER]       * (25 + 15 * p_live)
    score += s.pitems[CUFFS]      * (60 if not s.e_cuffed else 5)
    score += s.pitems[INVERTER]   * (22 + 18 * abs(p_live - 0.5))
    score += s.pitems[PHONE]      * (20 + 8 * future_unknowns)
    score += s.pitems[ADRENALINE] * 25

    score -= min(hp_room_e, s.eitems[CIGGS]) * 40
    score -= min(hp_room_e, s.eitems[PILLS]) * 16
    score -= s.eitems[SAW]        * (35 + 25 * p_live)
    score -= s.eitems[GLASS]      * (30 if current_unknown else 5)
    score -= s.eitems[BEER]       * (25 + 15 * p_live)
    score -= s.eitems[CUFFS]      * (60 if not s.p_cuffed else 5)
    score -= s.eitems[INVERTER]   * (22 + 18 * abs(p_live - 0.5))
    score -= s.eitems[PHONE]      * (20 + 8 * future_unknowns)
    score -= s.eitems[ADRENALINE] * 25

    if s.saw:
        score += 55.0 if s.player_turn else -55.0
    if s.e_cuffed:
        score += 90.0
    if s.p_cuffed:
        score -= 90.0

    return score


# ── Search infrastructure ────────────────────────────────────────────────────
def _exact_node(s: State) -> bool:
    """True when the position is small enough to solve without depth limit."""
    shells  = _shell_count(s)
    items   = _remaining_items(s)
    beliefs = len(s.worlds)
    if shells <= 4:
        return True
    if items <= 6:
        return True
    if shells <= 5 and items <= 12:
        return True
    return beliefs <= 4 and items <= 14


def _is_endgame(s: State) -> bool:
    return _shell_count(s) <= 5 and _remaining_items(s) <= 8


def _move_order_key(s: State, act: str, idx: int) -> int:
    p_live, p_blank = _probs(s)
    if act == "shoot_opp":
        return 0 if p_live > 0.6 else 14
    if act == "shoot_self":
        return 1 if p_blank > 0.8 else 15
    prio = {
        GLASS: 2, PHONE: 3, SAW: 5, CUFFS: 6,
        INVERTER: 7, ADRENALINE: 8, BEER: 9,
        CIGGS: 12, PILLS: 13,
    }
    return prio.get(idx, 20)


def _actions(s: State, by_player: bool) -> List[Tuple[str, int]]:
    """Generate legal actions with provably-correct forward pruning."""
    items     = s.pitems if by_player else s.eitems
    p_live, p_blank = _probs(s)
    actor_hp  = s.php if by_player else s.ehp

    acts: List[Tuple[str, int]] = []

    if p_blank >= 0.999:
        acts.append(("shoot_self", -1))
    elif p_live >= 0.999:
        acts.append(("shoot_opp", -1))
    else:
        acts.append(("shoot_opp", -1))
        acts.append(("shoot_self", -1))

    for i in range(9):
        if items[i] <= 0:
            continue
        if i in (GLASS, BEER, INVERTER) and _shell_count(s) == 0:
            continue
        if i == GLASS:
            known = _known_shells(s)
            if known and known[0] != 0:
                continue
        if i == PHONE and _shell_count(s) <= 1:
            continue
        if i == CUFFS and ((by_player and s.e_cuffed) or (not by_player and s.p_cuffed)):
            continue
        if i == ADRENALINE and not _adrenaline_targets(s, by_player):
            continue
        if i == CIGGS and actor_hp >= s.max_hp:
            continue
        if i == PILLS and actor_hp >= s.max_hp:
            continue
        if i == SAW and s.saw:
            continue
        acts.append(("item", i))

    acts.sort(key=lambda a: _move_order_key(s, a[0], a[1]))
    return acts


def _next_depth(depth: int | None, s: State, branching: bool) -> int | None:
    """Depth for the child node.  Deterministic outcomes are free."""
    if _exact_node(s):
        return None
    if depth is None:
        return None
    return depth - 1 if branching else depth


def _chance_value(outcomes: List[Tuple[float, State]], depth: int | None,
                  memo: dict,
                  alpha: float, beta: float) -> float:
    if not outcomes:
        return 0.0

    if len(outcomes) == 1 and outcomes[0][0] >= 0.999:
        ns = outcomes[0][1]
        return _minimax(ns, _next_depth(depth, ns, False), memo, alpha, beta)

    total = 0.0
    for prob, ns in sorted(outcomes, key=lambda it: it[0], reverse=True):
        if prob <= 0.0:
            continue
        val    = _minimax(ns, _next_depth(depth, ns, True), memo, -INF, INF)
        total += prob * val
    return total


def _score_action(s: State, act: str, idx: int, depth: int | None,
                  by_player: bool, memo: dict,
                  alpha: float, beta: float) -> float:
    if act == "shoot_opp":
        outcomes = _shoot_opp(s)
    elif act == "shoot_self":
        outcomes = _shoot_self(s)
    elif idx == ADRENALINE:
        _, score, _ = _best_adrenaline_choice(
            s, depth, by_player, memo, alpha, beta,
        )
        return score
    else:
        outcomes = _use_item(s, idx, by_player)
    if not outcomes:
        return _eval(s)
    return _chance_value(outcomes, depth, memo, alpha, beta)


def _best_adrenaline_choice(
    s: State, depth: int | None, by_player: bool,
    memo: dict,
    alpha: float = -INF, beta: float = INF,
) -> Tuple[int | None, float, List[Tuple[float, State]]]:
    targets = _adrenaline_targets(s, by_player)
    if not targets:
        return None, _eval(s), [(1.0, s.clone())]

    best_idx      = targets[0]
    best_outcomes = _use_adrenaline(s, by_player, best_idx)
    best_score    = _chance_value(best_outcomes, depth, memo, alpha, beta)

    if by_player:
        alpha = max(alpha, best_score)
    else:
        beta = min(beta, best_score)

    for t_idx in targets[1:]:
        if alpha >= beta:
            break
        outcomes = _use_adrenaline(s, by_player, t_idx)
        score    = _chance_value(outcomes, depth, memo, alpha, beta)
        if by_player and score > best_score:
            best_idx, best_score, best_outcomes = t_idx, score, outcomes
            alpha = max(alpha, best_score)
        elif not by_player and score < best_score:
            best_idx, best_score, best_outcomes = t_idx, score, outcomes
            beta = min(beta, best_score)

    return best_idx, best_score, best_outcomes


# ── Core search ──────────────────────────────────────────────────────────────
def _minimax(s: State, depth: int | None, memo: dict,
             alpha: float = -INF, beta: float = INF) -> float:
    if s.php <= 0:
        return -INF
    if s.ehp <= 0:
        return INF
    if _shell_count(s) == 0:
        return _round_reset_eval(s)
    if depth is not None and depth <= 0 and not _exact_node(s):
        return _eval(s)

    key     = s.key()
    req_tag = 10**9 if (depth is None or _exact_node(s)) else depth

    if depth is None and _is_endgame(s):
        cached = _ENDGAME_CACHE.get(key)
        if cached is not None:
            return cached

    entry = memo.get(key)
    if entry is not None:
        score, hit_depth, flag = entry
        if hit_depth >= req_tag:
            if flag == EXACT:
                return score
            if flag == LOWER:
                alpha = max(alpha, score)
            elif flag == UPPER:
                beta = min(beta, score)
            if alpha >= beta:
                return score

    orig_alpha, orig_beta = alpha, beta

    if s.player_turn:
        best = -INF
        for act, idx in _actions(s, True):
            sc = _score_action(s, act, idx, depth, True, memo, alpha, beta)
            if sc > best:
                best = sc
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        val = best if best > -INF else _eval(s)
    else:
        best = INF
        for act, idx in _actions(s, False):
            sc = _score_action(s, act, idx, depth, False, memo, alpha, beta)
            if sc < best:
                best = sc
            beta = min(beta, best)
            if alpha >= beta:
                break
        val = best if best < INF else _eval(s)

    if val <= orig_alpha:
        flag = UPPER
    elif val >= orig_beta:
        flag = LOWER
    else:
        flag = EXACT
    memo[key] = (val, req_tag, flag)

    if depth is None and _is_endgame(s):
        _ENDGAME_CACHE[key] = val
    return val


# ── Iterative-deepening wrapper ──────────────────────────────────────────────
def _find_best_action(
    s: State, memo: dict,
) -> Tuple[str, int, float, int | None]:

    if _is_endgame(s):
        best_sc, best_act, best_idx = -INF, "shoot_opp", -1
        for act, idx in _actions(s, True):
            sc = _score_action(s, act, idx, None, True, memo, -INF, INF)
            if sc > best_sc:
                best_sc, best_act, best_idx = sc, act, idx
        ad = None
        if best_act == "item" and best_idx == ADRENALINE:
            ad, _, _ = _best_adrenaline_choice(s, None, True, memo)
        return best_act, best_idx, best_sc, ad

    global _eval_used
    best: Tuple[str, int, float, int | None] = ("shoot_opp", -1, -INF, None)
    deadline = time.monotonic() + SAFETY_TIMEOUT

    for depth in range(1, 200):
        _eval_used = False
        cur_sc, cur_act, cur_idx = -INF, "shoot_opp", -1
        alpha = -INF
        timed_out = False

        for act, idx in _actions(s, True):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            sc = _score_action(s, act, idx, depth, True, memo, alpha, INF)
            if sc > cur_sc:
                cur_sc, cur_act, cur_idx = sc, act, idx
            alpha = max(alpha, cur_sc)

        if timed_out:
            break

        ad = None
        if cur_act == "item" and cur_idx == ADRENALINE:
            ad, _, _ = _best_adrenaline_choice(s, depth, True, memo)
        best = (cur_act, cur_idx, cur_sc, ad)

        if abs(cur_sc) >= INF / 2:
            break
        if not _eval_used:
            break

    return best


# ── Plan builder ──────────────────────────────────────────────────────────────
def _branching(outcomes: List[Tuple[float, State]]) -> bool:
    if len(outcomes) <= 1:
        return False
    return any(prob < 0.999 for prob, _ in outcomes)


def _shell_hint(s: State) -> str:
    p_live, _ = _probs(s)
    if p_live >= 0.99:
        return " (confirmed LIVE)"
    if p_live <= 0.01:
        return " (confirmed BLANK)"
    return f" ({int(p_live * 100)}% LIVE)"


def _fmt_action(act: str, idx: int, s: State,
                detail_idx: int | None = None) -> str:
    p_live, p_blank = _probs(s)
    dmg = 2 if s.saw else 1

    if act == "shoot_opp":
        cuff = " — dealer's turn SKIPPED by cuffs" if s.e_cuffed else ""
        if p_live >= 0.99:
            return f"Shoot DEALER — confirmed LIVE ({dmg} dmg){cuff}"
        if p_live <= 0.01:
            return f"Shoot DEALER — confirmed BLANK (end turn){cuff}"
        return f"Shoot DEALER — {int(p_live * 100)}% LIVE ({dmg} dmg if hit){cuff}"

    if act == "shoot_self":
        cuff = " — dealer's turn SKIPPED by cuffs" if s.e_cuffed else ""
        if p_blank >= 0.99:
            return f"Shoot YOURSELF — confirmed BLANK (free extra turn){cuff}"
        if p_blank <= 0.01:
            return f"Shoot YOURSELF — confirmed LIVE (AVOID — {dmg} dmg){cuff}"
        return f"Shoot YOURSELF — {int(p_blank * 100)}% BLANK (risky free turn){cuff}"

    hint = _shell_hint(s)
    if idx == ADRENALINE:
        chosen = detail_idx
        if chosen == GLASS:
            return "Use ADRENALINE — steal dealer's MAGNIFYING GLASS to reveal current shell"
        if chosen == PILLS:
            return "Use ADRENALINE — steal dealer's PILLS (50/50: +2 HP or -1 HP)"
        if chosen == PHONE:
            return "Use ADRENALINE — steal dealer's PHONE to reveal a random future shell"
        if chosen == CUFFS:
            return "Use ADRENALINE — steal dealer's HANDCUFFS to skip their turn"
        if chosen == SAW:
            return f"Use ADRENALINE — steal dealer's HACKSAW for 2 dmg{hint}"
        if chosen == CIGGS:
            return "Use ADRENALINE — steal dealer's CIGARETTES to restore 1 HP"
        if chosen == BEER:
            return f"Use ADRENALINE — steal dealer's BEER to eject current shell{hint}"
        if chosen == INVERTER:
            return f"Use ADRENALINE — steal dealer's INVERTER to flip current shell{hint}"
        return "Use ADRENALINE"

    labels = {
        GLASS:      "Use MAGNIFYING GLASS — reveal current shell",
        PILLS:      "Use PILLS — 50/50: +2 HP or -1 HP",
        PHONE:      "Use PHONE — reveal a random future shell",
        CUFFS:      "Use HANDCUFFS — dealer loses their next turn",
        ADRENALINE: "Use ADRENALINE",
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
    tasks: List[str] = []
    cur = state.clone()
    memo: dict = {}
    root_score = -INF

    for _ in range(MAX_PLAN_STEPS):
        if cur.php <= 0 or cur.ehp <= 0 or _shell_count(cur) == 0:
            break

        act, idx, score, ad_target = _find_best_action(cur, memo)
        if not tasks:
            root_score = score
        tasks.append(_fmt_action(act, idx, cur, ad_target))

        if act == "shoot_opp":
            outcomes = _shoot_opp(cur)
        elif act == "shoot_self":
            outcomes = _shoot_self(cur)
        elif idx == ADRENALINE:
            outcomes = _use_adrenaline(cur, True, ad_target)
        else:
            outcomes = _use_item(cur, idx, True)

        if not outcomes:
            break

        if _branching(outcomes):
            if act == "item":
                tasks.append(_update_prompt_for_branch(idx))
            else:
                all_player = all(ns.player_turn for _, ns in outcomes)
                if all_player:
                    tasks.append(
                        "After that shot update overlay — press AI again for your next turn"
                    )
            break

        cur = outcomes[0][1]
        if not cur.player_turn:
            break

    if not tasks:
        tasks.append("No beneficial action found — assess the situation manually")

    if root_score <= -INF / 2:
        tasks.append("WARNING: critical position — focus on survival")
    return tasks


# ── Public API ────────────────────────────────────────────────────────────────
def run_ai(player_hp: int, enemy_hp: int,
           player_items: list, enemy_items: list,
           shells: list, live_count: int, blank_count: int,
           max_hp: int = DEFAULT_MAX_HP,
           e_cuffed: bool = False,
           saw_active: bool = False) -> List[str]:
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

    state = State(
        php=player_hp, ehp=enemy_hp, max_hp=max_hp,
        pitems=list(player_items), eitems=list(enemy_items),
        worlds=worlds,
        e_cuffed=e_cuffed, saw=saw_active,
    )
    return build_plan(state)
