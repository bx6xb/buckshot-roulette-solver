//! bsr_core — Rust port of the search core in ai_engine.py.
//!
//! This is a FAITHFUL port: every function mirrors its Python counterpart
//! (named in the doc comments) operation-for-operation, including the order
//! of floating-point additions, dict-insertion-order iteration and the
//! stable action sort, so both engines produce identical best actions and
//! win probabilities (differential tests hold them to 1e-9).
//!
//! The dealer model, its quirks and the reload heuristic are documented in
//! ai_engine.py — read that file first; comments here only mark deviations
//! in representation, never in behaviour.
//!
//! Representation: a belief state's candidate worlds are bit-packed.  A
//! world of `n` shells is an integer `enc` where bit `(n-1-i)` is 1 iff the
//! shell at position `i` is BLANK; ascending `enc` order equals Python's
//! lexicographic tuple order (LIVE=1 < BLANK=2), so the sorted-worlds
//! invariant holds for free.  A set of worlds is a 256-bit bitset
//! (`[u64; 4]`, n <= 8 so enc < 256).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// ── Item indices (must match ITEMS_CONF order in overlay.py) ─────────────────
const GLASS: usize = 0;
const PILLS: usize = 1;
const PHONE: usize = 2;
const CUFFS: usize = 3;
const ADRENALINE: usize = 4;
const SAW: usize = 5;
const CIGGS: usize = 6;
const BEER: usize = 7;
const INVERTER: usize = 8;

const LIVE: u8 = 1;
const BLANK: u8 = 2;

const CUFF_NONE: u8 = 0;
const CUFF_ACTIVE: u8 = 1;
const CUFF_COOLDOWN: u8 = 2;

// ── Lexicographic value tuning (mirror ai_engine.py) ─────────────────────────
const P_EPS: f64 = 1e-9;
const STEP_COST: f64 = 0.01;
const HP_TB: f64 = 0.5;
const ENEMY_ITEM_TB: f64 = 0.6;
const ITEM_W: [f64; 9] = [1.8, 0.7, 1.0, 2.2, 1.6, 2.2, 1.2, 1.2, 1.6];

// ── Search-size guards ────────────────────────────────────────────────────────
// Same two-mode scheme as Python (exact first, iterative deepening over
// consumed shells as the safety net) but with far higher caps: a memo entry
// here is ~90 bytes of packed structs instead of kilobytes of tuples, so the
// exact solver reaches positions Python must punt on, and the fallback digs
// much deeper horizons.  The gate thresholds are kept identical to Python.
const EXACT_MEMO_CAP: usize = 20_000_000;
const FALLBACK_MEMO_CAP: usize = 8_000_000;
const FALLBACK_GATE_WORLDS: u32 = 40;
const FALLBACK_GATE_ITEMS: u32 = 10;

type Value = (f64, f64); // (win probability, tiebreak score)

/// Raised (as an Err) when the memo table outgrows the current cap —
/// mirrors _SearchOverflow.
struct Ovf;

// ── World-set plumbing ────────────────────────────────────────────────────────
type WorldSet = [u64; 4];

/// BLANK_AT_BIT[b]: 256-bit mask of every enc whose bit `b` is 1, i.e. all
/// worlds whose shell at position `n-1-b` is BLANK.
const fn build_blank_masks() -> [[u64; 4]; 8] {
    let mut m = [[0u64; 4]; 8];
    let mut b = 0;
    while b < 8 {
        let mut k = 0;
        while k < 256 {
            if (k >> b) & 1 == 1 {
                m[b][k / 64] |= 1u64 << (k % 64);
            }
            k += 1;
        }
        b += 1;
    }
    m
}
static BLANK_AT_BIT: [[u64; 4]; 8] = build_blank_masks();

#[inline]
fn ws_and(a: &WorldSet, b: &WorldSet) -> WorldSet {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}

#[inline]
fn ws_andnot(a: &WorldSet, b: &WorldSet) -> WorldSet {
    [a[0] & !b[0], a[1] & !b[1], a[2] & !b[2], a[3] & !b[3]]
}

#[inline]
fn ws_is_empty(a: &WorldSet) -> bool {
    a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
}

#[inline]
fn ws_len(a: &WorldSet) -> u32 {
    a[0].count_ones() + a[1].count_ones() + a[2].count_ones() + a[3].count_ones()
}

#[inline]
fn ws_set_bit(a: &mut WorldSet, k: usize) {
    a[k / 64] |= 1u64 << (k % 64);
}

/// Lowest set bit == lexicographically smallest world (worlds[0] in Python).
#[inline]
fn ws_min_bit(a: &WorldSet) -> u32 {
    for w in 0..4 {
        if a[w] != 0 {
            return (w as u32) * 64 + a[w].trailing_zeros();
        }
    }
    u32::MAX
}

/// Iterate encodings in ascending order (== Python's sorted worlds order).
fn ws_iter(set: &WorldSet) -> impl Iterator<Item = u16> + '_ {
    (0..4).flat_map(move |w| {
        let mut bits = set[w];
        std::iter::from_fn(move || {
            if bits == 0 {
                None
            } else {
                let b = bits.trailing_zeros();
                bits &= bits - 1;
                Some((w as u16) * 64 + b as u16)
            }
        })
    })
}

/// Shell value (LIVE/BLANK) at position `i` of world `enc` with `n` shells.
#[inline]
fn shell_at(enc: u16, n: u8, i: u8) -> u8 {
    1 + ((enc >> (n - 1 - i)) & 1) as u8
}

// ── State ─────────────────────────────────────────────────────────────────────
/// Mirrors the State namedtuple; max_hp is search-constant and lives in Ctx.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct St {
    worlds: WorldSet,
    n: u8,   // shells remaining (len of every world)
    mem: u8, // dealer's burner-phone knowledge, bit i = position i known
    php: u8,
    ehp: u8,
    pitems: [u8; 9],
    eitems: [u8; 9],
    saw: bool,
    e_cuff: u8,
    p_cuff: u8,
}

/// Memo key: ("P", s, d) for player nodes, ("D", s, locals..., d) for dealer.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    st: St,
    dealer: bool,
    knows: bool,
    known: u8,
    target: u8,
    used_med: bool,
    d: i32, // -1 encodes Python's None (exact, no horizon)
}

struct Ctx {
    memo: FxHashMap<Key, Value>,
    cap: usize,
    max_hp: u8,
}

impl Ctx {
    fn new(cap: usize, max_hp: u8) -> Self {
        Ctx { memo: FxHashMap::default(), cap, max_hp }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Player,
    Dealer,
    Win,
    Loss,
    Reload,
}

#[derive(Clone, Copy, PartialEq)]
enum Act {
    ShootOpp,
    ShootSelf,
    Item,
    Steal,
}

impl Act {
    fn as_str(self) -> &'static str {
        match self {
            Act::ShootOpp => "shoot_opp",
            Act::ShootSelf => "shoot_self",
            Act::Item => "item",
            Act::Steal => "steal",
        }
    }
}

// ── Belief helpers ────────────────────────────────────────────────────────────
/// _split_at: partition worlds by shell type at `idx`, groups in Python's
/// dict-insertion order (the group holding the smallest world comes first).
struct Split {
    arr: [(u8, WorldSet); 2],
    len: usize,
}

impl Split {
    fn iter(&self) -> std::slice::Iter<'_, (u8, WorldSet)> {
        self.arr[..self.len].iter()
    }
}

fn split_at(set: &WorldSet, n: u8, idx: u8) -> Split {
    let m = &BLANK_AT_BIT[(n - 1 - idx) as usize];
    let blank = ws_and(set, m);
    let live = ws_andnot(set, m);
    if ws_is_empty(&blank) {
        return Split { arr: [(LIVE, live), (BLANK, blank)], len: 1 };
    }
    if ws_is_empty(&live) {
        return Split { arr: [(BLANK, blank), (LIVE, live)], len: 1 };
    }
    if ws_min_bit(&live) < ws_min_bit(&blank) {
        Split { arr: [(LIVE, live), (BLANK, blank)], len: 2 }
    } else {
        Split { arr: [(BLANK, blank), (LIVE, live)], len: 2 }
    }
}

/// _consume_front: remove the chambered shell (worlds already filtered to
/// one type at position 0).  `saw` is passed explicitly: every fired shot
/// resets it, a beer eject preserves it.
fn consume_front(s: &St, ws: &WorldSet, saw: bool) -> St {
    let keep: u16 = (1u16 << (s.n - 1)) - 1;
    let mut set = [0u64; 4];
    for enc in ws_iter(ws) {
        ws_set_bit(&mut set, (enc & keep) as usize);
    }
    St { worlds: set, n: s.n - 1, mem: s.mem >> 1, saw, ..*s }
}

/// _toggle_front: flip the chambered shell in every world.
fn toggle_front(s: &St) -> St {
    let flip: u16 = 1u16 << (s.n - 1);
    let mut set = [0u64; 4];
    for enc in ws_iter(&s.worlds) {
        ws_set_bit(&mut set, (enc ^ flip) as usize);
    }
    St { worlds: set, ..*s }
}

/// _probs: (P(current is live), P(current is blank)).
fn probs(s: &St) -> (f64, f64) {
    if s.n == 0 || ws_is_empty(&s.worlds) {
        return (0.0, 0.0);
    }
    let total = ws_len(&s.worlds) as f64;
    let live = ws_len(&ws_andnot(&s.worlds, &BLANK_AT_BIT[(s.n - 1) as usize])) as f64;
    (live / total, 1.0 - live / total)
}

/// _known_at: every world agrees on the shell at `idx`.
fn known_at(s: &St, idx: u8) -> bool {
    let sub = ws_and(&s.worlds, &BLANK_AT_BIT[(s.n - 1 - idx) as usize]);
    ws_is_empty(&sub) || sub == s.worlds
}

/// _counts: (live, blank) remaining, read from the FIRST world only — the
/// Python original does the same (a quirk that matters after an inverter on
/// an unknown shell), so this must not be "fixed".
fn counts(set: &WorldSet, n: u8) -> (i32, i32) {
    let blanks = ws_min_bit(set).count_ones() as i32;
    (n as i32 - blanks, blanks)
}

// ── Value helpers ─────────────────────────────────────────────────────────────
fn better(a: Value, b: Value) -> bool {
    if a.0 > b.0 + P_EPS {
        return true;
    }
    if b.0 > a.0 + P_EPS {
        return false;
    }
    a.1 > b.1
}

fn wsum(items: &[u8; 9]) -> f64 {
    // Python computes these as sum(w * c for ...), and CPython 3.12+
    // builtins.sum runs Neumaier compensated summation over floats —
    // replicate it exactly, bit-for-bit parity depends on it.
    let mut f = 0.0f64;
    let mut c = 0.0f64;
    for i in 0..9 {
        let x = ITEM_W[i] * items[i] as f64;
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    f + c
}

fn econ_tb(s: &St) -> f64 {
    wsum(&s.pitems) - ENEMY_ITEM_TB * wsum(&s.eitems)
}

fn win_value(s: &St) -> Value {
    (1.0, econ_tb(s) + HP_TB * s.php as f64)
}

fn loss_value(s: &St) -> Value {
    (0.0, (-HP_TB) * s.ehp as f64)
}

/// _reload_value: heuristic frontier at a fresh unknown load.
fn reload_value(s: &St) -> Value {
    let own = wsum(&s.pitems);
    let foe = wsum(&s.eitems);
    let p = 0.5 + 0.055 * f64::from(s.php as i32 - s.ehp as i32) + 0.012 * (own - foe) + 0.02;
    let p = p.max(0.03).min(0.97);
    (p, econ_tb(s) + HP_TB * s.php as f64)
}

/// _cutoff_value: horizon estimate for the depth-limited fallback mode.
fn cutoff_value(s: &St, dealer_to_act: bool) -> Value {
    let (mut p, tb) = reload_value(s);
    if dealer_to_act {
        p -= 0.05;
    }
    if s.e_cuff == CUFF_ACTIVE {
        p += 0.04;
    }
    if s.p_cuff == CUFF_ACTIVE {
        p -= 0.04;
    }
    (p.max(0.03).min(0.97), tb)
}

// ── Shared shot / turn-passing mechanics ─────────────────────────────────────
fn after_shot_phase(s: &St) -> Phase {
    if s.n == 0 { Phase::Reload } else { Phase::Dealer }
}

fn phase_value(phase: Phase, s: &St, ctx: &mut Ctx, d: i32) -> Result<Value, Ovf> {
    match phase {
        Phase::Player => solve_player(s, ctx, d),
        Phase::Dealer => pass_to_dealer(s, ctx, d),
        Phase::Win => Ok(win_value(s)),
        Phase::Loss => Ok(loss_value(s)),
        Phase::Reload => Ok(reload_value(s)),
    }
}

fn pass_to_dealer(s: &St, ctx: &mut Ctx, d: i32) -> Result<Value, Ovf> {
    if s.e_cuff == CUFF_ACTIVE {
        let mut ns = *s;
        ns.e_cuff = CUFF_COOLDOWN;
        return pass_to_player(&ns, ctx, d);
    }
    let mut s2 = *s;
    if s2.e_cuff == CUFF_COOLDOWN {
        s2.e_cuff = CUFF_NONE;
    }
    dealer_turn(&s2, ctx, d)
}

fn pass_to_player(s: &St, ctx: &mut Ctx, d: i32) -> Result<Value, Ovf> {
    if s.p_cuff == CUFF_ACTIVE {
        let mut ns = *s;
        ns.p_cuff = CUFF_COOLDOWN;
        return pass_to_dealer(&ns, ctx, d);
    }
    let mut s2 = *s;
    if s2.p_cuff == CUFF_COOLDOWN {
        s2.p_cuff = CUFF_NONE;
    }
    solve_player(&s2, ctx, d)
}

// ── Player actions ────────────────────────────────────────────────────────────
fn item_usable_by_player(s: &St, max_hp: u8, idx: usize) -> bool {
    let n = s.n;
    match idx {
        GLASS => !known_at(s, 0),
        PILLS => s.php < max_hp,
        CIGGS => s.php < max_hp,
        PHONE => n >= 2 && (1..n).any(|i| !known_at(s, i)),
        CUFFS => s.e_cuff == CUFF_NONE && n >= 2,
        SAW => !s.saw,
        BEER => true,
        INVERTER => true,
        _ => false,
    }
}

fn adrenaline_targets(s: &St, max_hp: u8) -> Vec<usize> {
    if s.pitems[ADRENALINE] == 0 {
        return Vec::new();
    }
    (0..9)
        .filter(|&idx| idx != ADRENALINE && s.eitems[idx] > 0 && item_usable_by_player(s, max_hp, idx))
        .collect()
}

fn player_actions(s: &St, max_hp: u8) -> Vec<(Act, i8)> {
    let (p_live, _p_blank) = probs(s);
    let mut acts: Vec<(Act, i8)> = vec![(Act::ShootOpp, -1)];
    if p_live < 1.0 - P_EPS {
        // Self-shot on a confirmed live shell is strictly dominated.
        acts.push((Act::ShootSelf, -1));
    }
    for idx in 0..9 {
        if s.pitems[idx] == 0 || idx == ADRENALINE {
            continue;
        }
        if item_usable_by_player(s, max_hp, idx) {
            acts.push((Act::Item, idx as i8));
        }
    }
    for t in adrenaline_targets(s, max_hp) {
        acts.push((Act::Steal, t as i8));
    }
    acts
}

/// _apply_item_effect: effect of an item used by the player (already paid).
fn apply_item_effect(s: &St, idx: usize, max_hp: u8) -> Vec<(f64, St, Phase)> {
    let total = ws_len(&s.worlds) as f64;

    match idx {
        GLASS => {
            let sp = split_at(&s.worlds, s.n, 0);
            sp.iter()
                .map(|(_t, ws)| {
                    let mut ns = *s;
                    ns.worlds = *ws;
                    (ws_len(ws) as f64 / total, ns, Phase::Player)
                })
                .collect()
        }

        PILLS => {
            let mut good = *s;
            good.php = (good.php + 2).min(max_hp);
            let mut bad = *s;
            bad.php -= 1;
            let mut out = vec![(0.5, good, Phase::Player)];
            out.push((0.5, bad, if bad.php == 0 { Phase::Loss } else { Phase::Player }));
            out
        }

        CIGGS => {
            let mut ns = *s;
            ns.php = (ns.php + 1).min(max_hp);
            vec![(1.0, ns, Phase::Player)]
        }

        PHONE => {
            // BurnerPhone.gd: randi_range(1, n-1); with a full 8-shell load
            // the result 7 is remapped to 6 (position 8 never revealed).
            let n = s.n;
            let base = 1.0 / (n as f64 - 1.0);
            let mut weights: Vec<(u8, f64)> = Vec::new(); // insertion order
            for r in 1..n {
                let pos = if r == 7 && n == 8 { 6 } else { r };
                match weights.iter_mut().find(|(p, _)| *p == pos) {
                    Some((_, w)) => *w += base,
                    None => weights.push((pos, base)),
                }
            }
            let mut merged: Vec<(WorldSet, f64)> = Vec::new(); // insertion order
            for &(pos, wprob) in &weights {
                let sp = split_at(&s.worlds, n, pos);
                for (_t, ws) in sp.iter() {
                    let add = wprob * ws_len(ws) as f64 / total;
                    match merged.iter_mut().find(|(w, _)| w == ws) {
                        Some((_, p)) => *p += add,
                        None => merged.push((*ws, add)),
                    }
                }
            }
            merged
                .into_iter()
                .map(|(ws, p)| {
                    let mut ns = *s;
                    ns.worlds = ws;
                    (p, ns, Phase::Player)
                })
                .collect()
        }

        CUFFS => {
            let mut ns = *s;
            ns.e_cuff = CUFF_ACTIVE;
            vec![(1.0, ns, Phase::Player)]
        }

        SAW => {
            let mut ns = *s;
            ns.saw = true;
            vec![(1.0, ns, Phase::Player)]
        }

        BEER => {
            let sp = split_at(&s.worlds, s.n, 0);
            sp.iter()
                .map(|(_t, ws)| {
                    let ns = consume_front(s, ws, s.saw); // saw survives a beer eject
                    let phase = if ns.n == 0 { Phase::Reload } else { Phase::Player };
                    (ws_len(ws) as f64 / total, ns, phase)
                })
                .collect()
        }

        INVERTER => vec![(1.0, toggle_front(s), Phase::Player)],

        _ => vec![(1.0, *s, Phase::Player)],
    }
}

/// _apply_player.
fn apply_player(s: &St, act: Act, idx: i8, max_hp: u8) -> Vec<(f64, St, Phase)> {
    match act {
        Act::ShootOpp => {
            let dmg = if s.saw { 2 } else { 1 };
            let total = ws_len(&s.worlds) as f64;
            let mut out = Vec::with_capacity(2);
            let sp = split_at(&s.worlds, s.n, 0);
            for (stype, ws) in sp.iter() {
                let mut ns = consume_front(s, ws, false);
                let prob = ws_len(ws) as f64 / total;
                if *stype == LIVE {
                    ns.ehp = ns.ehp.saturating_sub(dmg);
                    if ns.ehp == 0 {
                        out.push((prob, ns, Phase::Win));
                        continue;
                    }
                }
                out.push((prob, ns, after_shot_phase(&ns)));
            }
            out
        }

        Act::ShootSelf => {
            let dmg = if s.saw { 2 } else { 1 };
            let total = ws_len(&s.worlds) as f64;
            let mut out = Vec::with_capacity(2);
            let sp = split_at(&s.worlds, s.n, 0);
            for (stype, ws) in sp.iter() {
                let mut ns = consume_front(s, ws, false);
                let prob = ws_len(ws) as f64 / total;
                if *stype == LIVE {
                    ns.php = ns.php.saturating_sub(dmg);
                    if ns.php == 0 {
                        out.push((prob, ns, Phase::Loss));
                        continue;
                    }
                    out.push((prob, ns, after_shot_phase(&ns)));
                } else {
                    // Blank into yourself: free extra turn.
                    let phase = if ns.n == 0 { Phase::Reload } else { Phase::Player };
                    out.push((prob, ns, phase));
                }
            }
            out
        }

        Act::Item => {
            let mut ns = *s;
            ns.pitems[idx as usize] -= 1;
            apply_item_effect(&ns, idx as usize, max_hp)
        }

        Act::Steal => {
            let mut ns = *s;
            ns.pitems[ADRENALINE] -= 1;
            ns.eitems[idx as usize] -= 1;
            apply_item_effect(&ns, idx as usize, max_hp)
        }
    }
}

// ── Player search ─────────────────────────────────────────────────────────────
fn item_order(idx: i8) -> i32 {
    match idx as usize {
        GLASS => 2,
        PHONE => 3,
        INVERTER => 4,
        SAW => 5,
        CUFFS => 6,
        BEER => 8,
        CIGGS => 9,
        PILLS => 10,
        _ => 7,
    }
}

/// _ordered_actions: try likely-strong actions first (stable sort, so ties
/// keep the _player_actions order — this decides equal-value action picks).
fn ordered_actions(s: &St, max_hp: u8) -> Vec<(Act, i8)> {
    let (p_live, p_blank) = probs(s);
    let mut acts = player_actions(s, max_hp);
    acts.sort_by_key(|&(act, idx)| match act {
        Act::ShootOpp => if p_live > 0.6 { 0 } else { 12 },
        Act::ShootSelf => if p_blank > 0.8 { 1 } else { 13 },
        Act::Item => item_order(idx),
        Act::Steal => item_order(idx) + 2,
    });
    acts
}

/// _best_action: exact max over player actions with the sound
/// upper-bound abandonment cut.
fn search_best_action(s: &St, ctx: &mut Ctx, d: i32) -> Result<(Act, i8, Value), Ovf> {
    let n0 = s.n;
    let mut best_v: Option<Value> = None;
    let mut best_a = (Act::ShootOpp, -1i8);
    for (act, idx) in ordered_actions(s, ctx.max_hp) {
        let mut p = 0.0;
        let mut tb = 0.0;
        let mut rem = 1.0;
        let mut abandoned = false;
        for (prob, ns, phase) in apply_player(s, act, idx, ctx.max_hp) {
            let dd = if d < 0 || ns.n == n0 { d } else { d - 1 };
            let v = phase_value(phase, &ns, ctx, dd)?;
            p += prob * v.0;
            tb += prob * v.1;
            rem -= prob;
            if let Some(bv) = best_v {
                if p + rem < bv.0 - P_EPS {
                    abandoned = true;
                    break;
                }
            }
        }
        if abandoned {
            continue;
        }
        let v = (p, tb);
        if best_v.is_none() || better(v, best_v.unwrap()) {
            best_v = Some(v);
            best_a = (act, idx);
        }
    }
    Ok((best_a.0, best_a.1, best_v.unwrap()))
}

/// _solve_player.
fn solve_player(s: &St, ctx: &mut Ctx, d: i32) -> Result<Value, Ovf> {
    if s.php == 0 {
        return Ok(loss_value(s));
    }
    if s.ehp == 0 {
        return Ok(win_value(s));
    }
    if s.n == 0 {
        return Ok(reload_value(s));
    }
    if d == 0 {
        return Ok(cutoff_value(s, false));
    }

    let key = Key { st: *s, dealer: false, knows: false, known: 0, target: 0, used_med: false, d };
    if let Some(v) = ctx.memo.get(&key) {
        return Ok(*v);
    }

    let (_a, _i, best) = search_best_action(s, ctx, d)?;
    let val = (best.0, best.1 - STEP_COST);
    if ctx.memo.len() >= ctx.cap {
        return Err(Ovf);
    }
    ctx.memo.insert(key, val);
    Ok(val)
}

// ── Dealer policy (see DealerIntelligence.gd notes in ai_engine.py) ──────────
/// _dealer_figures_out: can the dealer deduce the chambered shell?
fn dealer_figures_out(enc: u16, mem: u8, n: u8) -> bool {
    if mem & 1 != 0 {
        return true;
    }
    let blanks_total = enc.count_ones() as i32;
    let mut lives = n as i32 - blanks_total;
    let mut blanks = blanks_total;
    if lives == 0 || blanks == 0 {
        return true;
    }
    for i in 0..n {
        if (mem >> i) & 1 != 0 {
            if shell_at(enc, n, i) == LIVE {
                lives -= 1;
            } else {
                blanks -= 1;
            }
        }
    }
    lives == 0 || blanks == 0
}

fn dealer_turn(s: &St, ctx: &mut Ctx, d: i32) -> Result<Value, Ovf> {
    dealer_iter(s, ctx, false, 0, 0, false, d)
}

/// _dealer_iter: one DealerChoice() iteration, memoized with turn locals.
fn dealer_iter(s: &St, ctx: &mut Ctx, knows: bool, known: u8, target: u8,
               used_med: bool, d: i32) -> Result<Value, Ovf> {
    if s.php == 0 {
        return Ok(loss_value(s));
    }
    if s.ehp == 0 {
        return Ok(win_value(s));
    }
    if s.n == 0 {
        return Ok(reload_value(s));
    }
    if d == 0 {
        return Ok(cutoff_value(s, true));
    }

    let key = Key { st: *s, dealer: true, knows, known, target, used_med, d };
    if let Some(v) = ctx.memo.get(&key) {
        return Ok(*v);
    }
    let val = dealer_iter_inner(s, ctx, knows, known, target, used_med, d)?;
    if ctx.memo.len() >= ctx.cap {
        return Err(Ovf);
    }
    ctx.memo.insert(key, val);
    Ok(val)
}

fn dealer_iter_inner(s: &St, ctx: &mut Ctx, knows: bool, known: u8, target: u8,
                     used_med: bool, d: i32) -> Result<Value, Ovf> {
    // FigureOutShell — endless-mode inference.
    if !knows {
        // Group worlds by (figures_out, chambered value) in first-seen order.
        let mut groups: Vec<((bool, u8), WorldSet)> = Vec::with_capacity(3);
        for enc in ws_iter(&s.worlds) {
            let key = if dealer_figures_out(enc, s.mem, s.n) {
                (true, shell_at(enc, s.n, 0))
            } else {
                (false, 0u8)
            };
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, set)) => ws_set_bit(set, enc as usize),
                None => {
                    let mut set = [0u64; 4];
                    ws_set_bit(&mut set, enc as usize);
                    groups.push((key, set));
                }
            }
        }
        if groups.len() > 1 || groups.iter().any(|((fos, _), _)| *fos) {
            let total = ws_len(&s.worlds) as f64;
            let mut p = 0.0;
            let mut tb = 0.0;
            for ((fos, val), ws) in &groups {
                let mut ns = *s;
                ns.worlds = *ws;
                let v = if *fos {
                    let tgt = if *val == LIVE { 2 } else { 1 }; // 2=player, 1=self
                    dealer_items(&ns, ctx, true, *val, tgt, used_med, d)?
                } else {
                    dealer_items(&ns, ctx, false, 0, target, used_med, d)?
                };
                let cnt = ws_len(ws) as f64;
                p += cnt / total * v.0;
                tb += cnt / total * v.1;
            }
            return Ok((p, tb));
        }
    }
    dealer_items(s, ctx, knows, known, target, used_med, d)
}

fn dealer_items(s: &St, ctx: &mut Ctx, knows: bool, known: u8, target: u8,
                used_med: bool, d: i32) -> Result<Value, Ovf> {
    let n = s.n;

    // Last-shell rule: with one shell left the dealer always knows it.
    if n == 1 && !knows {
        let total = ws_len(&s.worlds) as f64;
        let mut p = 0.0;
        let mut tb = 0.0;
        let sp = split_at(&s.worlds, n, 0);
        for (stype, ws) in sp.iter() {
            let mut ns = *s;
            ns.worlds = *ws;
            let tgt = if *stype == LIVE { 2 } else { 1 };
            let v = dealer_items(&ns, ctx, true, *stype, tgt, used_med, d)?;
            let cnt = ws_len(ws) as f64;
            p += cnt / total * v.0;
            tb += cnt / total * v.1;
        }
        return Ok((p, tb));
    }

    let own = dealer_eligible(s, &s.eitems, knows, known, used_med, ctx.max_hp);
    if !own.is_empty() {
        let total: u32 = own.iter().map(|&(_, c)| c as u32).sum();
        let totf = total as f64;
        let mut p = 0.0;
        let mut tb = 0.0;
        for &(idx, cnt) in &own {
            let mut ns = *s;
            ns.eitems[idx] -= 1;
            let v = dealer_use(&ns, ctx, idx, knows, known, target, used_med, d)?;
            p += cnt as f64 / totf * v.0;
            tb += cnt as f64 / totf * v.1;
        }
        return Ok((p, tb));
    }

    if s.eitems[ADRENALINE] > 0 {
        let steal = dealer_eligible(s, &s.pitems, knows, known, used_med, ctx.max_hp);
        if !steal.is_empty() {
            let total: u32 = steal.iter().map(|&(_, c)| c as u32).sum();
            let totf = total as f64;
            let mut p = 0.0;
            let mut tb = 0.0;
            for &(idx, cnt) in &steal {
                let mut ns = *s;
                ns.eitems[ADRENALINE] -= 1;
                ns.pitems[idx] -= 1;
                let v = dealer_use(&ns, ctx, idx, knows, known, target, used_med, d)?;
                p += cnt as f64 / totf * v.0;
                tb += cnt as f64 / totf * v.1;
            }
            return Ok((p, tb));
        }
    }

    dealer_bonus_saw(s, ctx, knows, known, target, used_med, d)
}

/// _dealer_eligible: (item, count) pairs whose trigger condition passes,
/// evaluated with the DEALER's knowledge.
fn dealer_eligible(s: &St, items: &[u8; 9], knows: bool, known: u8,
                   used_med: bool, max_hp: u8) -> Vec<(usize, u8)> {
    let n = s.n;
    let has_cigs = s.eitems[CIGGS] > 0;
    let mut out = Vec::new();
    for idx in 0..9 {
        let cnt = items[idx];
        if cnt == 0 {
            continue;
        }
        let ok = match idx {
            GLASS => !knows && n != 1,
            CIGGS => s.ehp < max_hp,
            PILLS => s.ehp < max_hp && !has_cigs && !used_med && s.ehp != 1,
            BEER => known != LIVE && n != 1, // unknown shells qualify too
            CUFFS => s.p_cuff == CUFF_NONE && n != 1,
            SAW => !s.saw && known == LIVE,
            PHONE => n > 2,
            INVERTER => knows && known == BLANK,
            _ => false,
        };
        if ok {
            out.push((idx, cnt));
        }
    }
    out
}

/// _dealer_use: apply one dealer item (already paid for), continue the turn.
fn dealer_use(s: &St, ctx: &mut Ctx, idx: usize, knows: bool, known: u8,
              target: u8, used_med: bool, d: i32) -> Result<Value, Ovf> {
    let total = ws_len(&s.worlds) as f64;

    match idx {
        GLASS => {
            let mut p = 0.0;
            let mut tb = 0.0;
            let sp = split_at(&s.worlds, s.n, 0);
            for (stype, ws) in sp.iter() {
                let mut ns = *s;
                ns.worlds = *ws;
                let tgt = if *stype == LIVE { 2 } else { 1 };
                let v = dealer_iter(&ns, ctx, true, *stype, tgt, used_med, d)?;
                let cnt = ws_len(ws) as f64;
                p += cnt / total * v.0;
                tb += cnt / total * v.1;
            }
            Ok((p, tb))
        }

        CIGGS => {
            let mut ns = *s;
            ns.ehp = (ns.ehp + 1).min(ctx.max_hp);
            dealer_iter(&ns, ctx, knows, known, target, used_med, d)
        }

        PILLS => {
            let mut good = *s;
            good.ehp = (good.ehp + 2).min(ctx.max_hp);
            let mut bad = *s;
            bad.ehp -= 1;
            let v_good = dealer_iter(&good, ctx, knows, known, target, true, d)?;
            let v_bad = if bad.ehp == 0 {
                win_value(&bad)
            } else {
                dealer_iter(&bad, ctx, knows, known, target, true, d)?
            };
            Ok((0.5 * (v_good.0 + v_bad.0), 0.5 * (v_good.1 + v_bad.1)))
        }

        BEER => {
            // Ejects the chambered shell; the dealer FORGETS what he knew
            // about it but his shooting target is NOT cleared (stale-target
            // quirk).
            let dd = if d < 0 { -1 } else { d - 1 };
            let mut p = 0.0;
            let mut tb = 0.0;
            let sp = split_at(&s.worlds, s.n, 0);
            for (_stype, ws) in sp.iter() {
                let ns = consume_front(s, ws, s.saw);
                let prob = ws_len(ws) as f64 / total;
                let v = if ns.n == 0 {
                    reload_value(&ns) // unreachable: beer needs n != 1
                } else {
                    dealer_iter(&ns, ctx, false, 0, target, used_med, dd)?
                };
                p += prob * v.0;
                tb += prob * v.1;
            }
            Ok((p, tb))
        }

        CUFFS => {
            let mut ns = *s;
            ns.p_cuff = CUFF_ACTIVE;
            dealer_iter(&ns, ctx, knows, known, target, used_med, d)
        }

        SAW => {
            let mut ns = *s;
            ns.saw = true;
            dealer_iter(&ns, ctx, knows, known, target, used_med, d)
        }

        PHONE => {
            // Dealer version: uniform over future slots.
            let n = s.n;
            let base = 1.0 / (n as f64 - 1.0);
            let mut merged: Vec<(u8, f64)> = Vec::new(); // insertion order
            for pos in 1..n {
                let mem = s.mem | (1u8 << pos);
                match merged.iter_mut().find(|(m, _)| *m == mem) {
                    Some((_, p)) => *p += base,
                    None => merged.push((mem, base)),
                }
            }
            let mut p = 0.0;
            let mut tb = 0.0;
            for &(mem, prob) in &merged {
                let mut ns = *s;
                ns.mem = mem;
                let v = dealer_iter(&ns, ctx, knows, known, target, used_med, d)?;
                p += prob * v.0;
                tb += prob * v.1;
            }
            Ok((p, tb))
        }

        INVERTER => {
            let ns = toggle_front(s);
            dealer_iter(&ns, ctx, true, LIVE, 2, used_med, d)
        }

        _ => dealer_iter(s, ctx, knows, known, target, used_med, d),
    }
}

/// _weighted_flip: CoinFlip() in endless mode; 1 -> player, 0 -> self.
fn weighted_flip(set: &WorldSet, n: u8) -> Vec<(f64, u8)> {
    let (lives, blanks) = counts(set, n);
    if lives > blanks {
        vec![(1.0, 1)]
    } else if lives < blanks {
        vec![(1.0, 0)]
    } else {
        vec![(0.5, 0), (0.5, 1)]
    }
}

/// _dealer_bonus_saw: leftover-handsaw coin flip.
fn dealer_bonus_saw(s: &St, ctx: &mut Ctx, knows: bool, known: u8, target: u8,
                    used_med: bool, d: i32) -> Result<Value, Ovf> {
    let has_saw = s.eitems[SAW] > 0 || (s.eitems[ADRENALINE] > 0 && s.pitems[SAW] > 0);
    if has_saw && !s.saw && known != BLANK {
        let mut p = 0.0;
        let mut tb = 0.0;
        for (prob, res) in weighted_flip(&s.worlds, s.n) {
            let v = if res == 0 {
                dealer_shoot(s, ctx, knows, known, 1, d)?
            } else {
                let mut ns = *s;
                if ns.eitems[SAW] > 0 {
                    ns.eitems[SAW] -= 1;
                } else {
                    ns.eitems[ADRENALINE] -= 1;
                    ns.pitems[SAW] -= 1;
                }
                ns.saw = true;
                dealer_iter(&ns, ctx, knows, known, 2, used_med, d)?
            };
            p += prob * v.0;
            tb += prob * v.1;
        }
        return Ok((p, tb));
    }
    dealer_shoot(s, ctx, knows, known, target, d)
}

/// _dealer_shoot: Shoot(dealerTarget) — or the count-weighted random choice.
fn dealer_shoot(s: &St, ctx: &mut Ctx, knows: bool, known: u8, target: u8,
                d: i32) -> Result<Value, Ovf> {
    if target == 0 {
        let mut p = 0.0;
        let mut tb = 0.0;
        for (prob, res) in weighted_flip(&s.worlds, s.n) {
            let v = dealer_shoot(s, ctx, knows, known, if res == 0 { 1 } else { 2 }, d)?;
            p += prob * v.0;
            tb += prob * v.1;
        }
        return Ok((p, tb));
    }

    let dmg = if s.saw { 2 } else { 1 };
    let total = ws_len(&s.worlds) as f64;
    let dd = if d < 0 { -1 } else { d - 1 };
    let mut p = 0.0;
    let mut tb = 0.0;
    let sp = split_at(&s.worlds, s.n, 0);
    for (stype, ws) in sp.iter() {
        let prob = ws_len(ws) as f64 / total;

        let v;
        if target == 1 {
            // dealer shoots himself
            if *stype == BLANK {
                // Free extra turn; the barrel does NOT regrow on this path.
                let ns = consume_front(s, ws, s.saw);
                v = if ns.n == 0 { reload_value(&ns) } else { dealer_turn(&ns, ctx, dd)? };
                p += prob * v.0;
                tb += prob * v.1;
                continue;
            }
            let mut ns = consume_front(s, ws, false);
            ns.ehp = ns.ehp.saturating_sub(dmg);
            v = if ns.ehp == 0 {
                win_value(&ns)
            } else if ns.n == 0 {
                reload_value(&ns)
            } else {
                pass_to_player(&ns, ctx, dd)?
            };
        } else {
            // dealer shoots the player
            let mut ns = consume_front(s, ws, false);
            if *stype == LIVE {
                ns.php = ns.php.saturating_sub(dmg);
            }
            v = if ns.php == 0 {
                loss_value(&ns)
            } else if ns.n == 0 {
                reload_value(&ns)
            } else {
                pass_to_player(&ns, ctx, dd)?
            };
        }
        p += prob * v.0;
        tb += prob * v.1;
    }
    Ok((p, tb))
}

// ── Root search (mirror of _plan_root_search) ────────────────────────────────
type RootHit = (Act, i8, Value, i32); // horizon: -1 == exact

fn plan_root_search(s: &St, max_hp: u8) -> (Option<RootHit>, Option<Ctx>) {
    let items_total: u32 =
        s.pitems.iter().map(|&c| c as u32).sum::<u32>() + s.eitems.iter().map(|&c| c as u32).sum::<u32>();
    let huge = ws_len(&s.worlds) >= FALLBACK_GATE_WORLDS && items_total >= FALLBACK_GATE_ITEMS;

    if !huge {
        let mut ctx = Ctx::new(EXACT_MEMO_CAP, max_hp);
        if let Ok((a, i, v)) = search_best_action(s, &mut ctx, -1) {
            return (Some((a, i, v, -1)), Some(ctx));
        }
    }

    // Iterative deepening over consumed shells; keep the memo of the last
    // completed horizon (like Python keeps the previous iteration's dict).
    let mut result: Option<RootHit> = None;
    let mut kept: Option<Ctx> = None;
    for h in 1..=(s.n as i32) {
        let mut ctx = Ctx::new(FALLBACK_MEMO_CAP, max_hp);
        match search_best_action(s, &mut ctx, h) {
            Ok((a, i, v)) => {
                result = Some((a, i, v, h));
                kept = Some(ctx);
            }
            Err(_) => break,
        }
    }
    (result, kept)
}

// ── Python interface ──────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn parse_state(php: u8, ehp: u8, pitems: &[u8], eitems: &[u8], worlds: &[Vec<u8>],
               mem: &[bool], saw: bool, e_cuff: u8, p_cuff: u8) -> PyResult<St> {
    if pitems.len() != 9 || eitems.len() != 9 {
        return Err(PyValueError::new_err("item lists must have 9 entries"));
    }
    if worlds.is_empty() {
        return Err(PyValueError::new_err("worlds must be non-empty"));
    }
    let n = worlds[0].len();
    if n == 0 || n > 8 {
        return Err(PyValueError::new_err("shell count must be 1..=8"));
    }
    if mem.len() != n {
        return Err(PyValueError::new_err("mem length must equal shell count"));
    }

    let mut set = [0u64; 4];
    for w in worlds {
        if w.len() != n {
            return Err(PyValueError::new_err("worlds must have equal length"));
        }
        let mut enc: u16 = 0;
        for (i, &sh) in w.iter().enumerate() {
            match sh {
                1 => {}
                2 => enc |= 1u16 << (n - 1 - i),
                _ => return Err(PyValueError::new_err("shells must be 1 (live) or 2 (blank)")),
            }
        }
        ws_set_bit(&mut set, enc as usize);
    }

    let mut memb: u8 = 0;
    for (i, &m) in mem.iter().enumerate() {
        if m {
            memb |= 1u8 << i;
        }
    }

    let mut pi = [0u8; 9];
    pi.copy_from_slice(pitems);
    let mut ei = [0u8; 9];
    ei.copy_from_slice(eitems);

    Ok(St {
        worlds: set,
        n: n as u8,
        mem: memb,
        php,
        ehp,
        pitems: pi,
        eitems: ei,
        saw,
        e_cuff,
        p_cuff,
    })
}

/// One search session: `root_search` mirrors _plan_root_search and fixes the
/// mode (exact / horizon h) plus the memo; `best_action` re-solves follow-up
/// plan states against that same memo, exactly like Python's build_plan loop.
#[pyclass]
struct Solver {
    ctx: Option<Ctx>,
    horizon: i32, // -1 == exact
}

#[pymethods]
impl Solver {
    #[new]
    fn new() -> Self {
        Solver { ctx: None, horizon: -1 }
    }

    /// Returns (act, idx, p, tb, horizon|None) or None when even the
    /// shallowest fallback horizon overflows.
    #[allow(clippy::too_many_arguments)]
    fn root_search(&mut self, py: Python<'_>, php: u8, ehp: u8, max_hp: u8,
                   pitems: Vec<u8>, eitems: Vec<u8>, worlds: Vec<Vec<u8>>,
                   mem: Vec<bool>, saw: bool, e_cuff: u8, p_cuff: u8)
                   -> PyResult<Option<(String, i64, f64, f64, Option<i64>)>> {
        let st = parse_state(php, ehp, &pitems, &eitems, &worlds, &mem, saw, e_cuff, p_cuff)?;
        let (result, ctx) = py.detach(move || plan_root_search(&st, max_hp));
        self.ctx = ctx;
        match result {
            None => Ok(None),
            Some((act, idx, v, h)) => {
                self.horizon = h;
                let horizon = if h < 0 { None } else { Some(h as i64) };
                Ok(Some((act.as_str().to_owned(), idx as i64, v.0, v.1, horizon)))
            }
        }
    }

    /// Returns (act, idx, p, tb) or None on memo overflow (plan loop stops).
    #[allow(clippy::too_many_arguments)]
    fn best_action(&mut self, py: Python<'_>, php: u8, ehp: u8, max_hp: u8,
                   pitems: Vec<u8>, eitems: Vec<u8>, worlds: Vec<Vec<u8>>,
                   mem: Vec<bool>, saw: bool, e_cuff: u8, p_cuff: u8)
                   -> PyResult<Option<(String, i64, f64, f64)>> {
        let _ = max_hp; // ctx keeps the root's max_hp (constant per game)
        let st = parse_state(php, ehp, &pitems, &eitems, &worlds, &mem, saw, e_cuff, p_cuff)?;
        let Some(ctx) = self.ctx.as_mut() else {
            return Ok(None);
        };
        let horizon = self.horizon;
        match py.detach(move || search_best_action(&st, ctx, horizon)) {
            Ok((act, idx, v)) => Ok(Some((act.as_str().to_owned(), idx as i64, v.0, v.1))),
            Err(_) => Ok(None),
        }
    }
}

// ── Debug / diagnostics (exact-mode mirrors of the Python helpers) ───────────
fn phase_from_str(phase: &str) -> PyResult<Phase> {
    match phase {
        "player" => Ok(Phase::Player),
        "dealer" => Ok(Phase::Dealer),
        "win" => Ok(Phase::Win),
        "loss" => Ok(Phase::Loss),
        "reload" => Ok(Phase::Reload),
        _ => Err(PyValueError::new_err("unknown phase")),
    }
}

/// _phase_value with a fresh memo, exact mode (diagnostics/tests only).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn debug_phase_value(phase: &str, php: u8, ehp: u8, max_hp: u8,
                     pitems: Vec<u8>, eitems: Vec<u8>, worlds: Vec<Vec<u8>>,
                     mem: Vec<bool>, saw: bool, e_cuff: u8, p_cuff: u8)
                     -> PyResult<(f64, f64)> {
    let st = parse_state(php, ehp, &pitems, &eitems, &worlds, &mem, saw, e_cuff, p_cuff)?;
    let ph = phase_from_str(phase)?;
    let mut ctx = Ctx::new(EXACT_MEMO_CAP, max_hp);
    phase_value(ph, &st, &mut ctx, -1)
        .map_err(|_| PyValueError::new_err("search overflow"))
}

/// _action_values: un-pruned per-action values (diagnostics/tests only).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn debug_action_values(php: u8, ehp: u8, max_hp: u8,
                       pitems: Vec<u8>, eitems: Vec<u8>, worlds: Vec<Vec<u8>>,
                       mem: Vec<bool>, saw: bool, e_cuff: u8, p_cuff: u8)
                       -> PyResult<Vec<(String, i64, f64, f64)>> {
    let st = parse_state(php, ehp, &pitems, &eitems, &worlds, &mem, saw, e_cuff, p_cuff)?;
    let mut ctx = Ctx::new(EXACT_MEMO_CAP, max_hp);
    let n0 = st.n;
    let mut out = Vec::new();
    for (act, idx) in player_actions(&st, ctx.max_hp) {
        let mut p = 0.0;
        let mut tb = 0.0;
        let _ = n0; // exact mode: d stays None on every branch
        for (prob, ns, phase) in apply_player(&st, act, idx, ctx.max_hp) {
            let v = phase_value(phase, &ns, &mut ctx, -1)
                .map_err(|_| PyValueError::new_err("search overflow"))?;
            p += prob * v.0;
            tb += prob * v.1;
        }
        out.push((act.as_str().to_owned(), idx as i64, p, tb));
    }
    Ok(out)
}

/// _dealer_iter with a fresh memo, exact mode (diagnostics/tests only).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn debug_dealer_iter(php: u8, ehp: u8, max_hp: u8,
                     pitems: Vec<u8>, eitems: Vec<u8>, worlds: Vec<Vec<u8>>,
                     mem: Vec<bool>, saw: bool, e_cuff: u8, p_cuff: u8,
                     knows: bool, known: u8, target: u8, used_med: bool)
                     -> PyResult<(f64, f64)> {
    let st = parse_state(php, ehp, &pitems, &eitems, &worlds, &mem, saw, e_cuff, p_cuff)?;
    let mut ctx = Ctx::new(EXACT_MEMO_CAP, max_hp);
    dealer_iter(&st, &mut ctx, knows, known, target, used_med, -1)
        .map_err(|_| PyValueError::new_err("search overflow"))
}

/// ITEM_W weighted sum (diagnostics only — float-order comparisons).
#[pyfunction]
fn debug_wsum(items: Vec<u8>) -> PyResult<f64> {
    if items.len() != 9 {
        return Err(PyValueError::new_err("need 9 items"));
    }
    let mut arr = [0u8; 9];
    arr.copy_from_slice(&items);
    Ok(wsum(&arr))
}

#[pymodule]
fn bsr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_function(wrap_pyfunction!(debug_phase_value, m)?)?;
    m.add_function(wrap_pyfunction!(debug_action_values, m)?)?;
    m.add_function(wrap_pyfunction!(debug_dealer_iter, m)?)?;
    m.add_function(wrap_pyfunction!(debug_wsum, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
