"""overlay.py — Buckshot Roulette HUD Overlay
Borderless, always-on-top transparent overlay for the primary monitor.
Press Escape to quit.

Windows-only: the click-through/transparency layer is built on win32gui.
Not supported on macOS or Linux.
"""

import os
import sys

if os.name != "nt":
    sys.exit(
        "overlay.py runs on Windows only (uses win32gui for the transparent "
        "click-through overlay). macOS/Linux are not supported."
    )


def _resource_path(relative_path: str) -> str:
    """Resolve a bundled resource so it works both run from source and from
    a PyInstaller-frozen exe (--onefile extracts data next to sys._MEIPASS,
    not the process's current working directory)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)

import tkinter as tk
import ctypes
import win32gui
import win32con
import win32api
import mss
import threading
import time
from PIL import Image, ImageTk

from scanner import Scanner

# ─────────────────────────────────────────────── Config ──────────────────────
MONITOR_INDEX = 1      # fallback only — see _primary_monitor()
ALPHA_IDLE    = 0.2
ALPHA_HOVER   = 1
ALPHA_STEP    = 0.2
CHROMA        = "#010101"

ITEMS_CONF = [
    ("glass",      "pics/glass.jpg"),
    ("pills",      "pics/pills.jpg"),
    ("phone",      "pics/phone.jpg"),
    ("cuffs",      "pics/cuffs.jpg"),
    ("adrenaline", "pics/adrenalin.jpg"),
    ("saw",        "pics/saw.jpg"),
    ("ciggs",      "pics/ciggs.jpg"),
    ("beer",       "pics/beer.jpg"),
    ("inverter",   "pics/inverter.jpg"),
]

TASKS_INIT = []

MAX_HP      = 6
MAX_BULLETS = 5
MAX_ITEMS   = 8

# ─────────────────────────────────────────────── Palette ─────────────────────
C = {
    "bg":         "#0B0B14",
    "border":     "#252538",
    "border_hi":  "#3A3A58",
    "red":        "#8A2020",
    "red_hi":     "#C83030",
    "blue":       "#1A4878",
    "blue_hi":    "#2870C0",
    "green":      "#18884A",
    "green_hi":   "#20C060",
    "text":       "#B8B8A8",
    "text_dim":   "#484840",
    "text_hi":    "#F0EAD8",
    "hp_fill":    "#18D058",
    "hp_empty":   "#0C1C14",
    "ehp_fill":   "#C83030",
    "ehp_empty":  "#1C0C0C",
    "item_bg":    "#0D0D1C",
    "item_brd":   "#202030",
    "btn_bg":     "#111120",
    "btn_brd":    "#2A2A40",
    "btn_text":   "#8A8A7A",
}

F      = ("Courier New", 20)
F_SM   = ("Courier New", 17)
F_LG   = ("Courier New", 38, "bold")
F_HDR  = ("Courier New", 17, "bold")
F_NUM  = ("Courier New", 17, "bold")

M        = 28    # screen margin
ISZ      = 84    # item icon size px (default: Full HD)
IG       = 7     # icon gap px (default: Full HD)
PP       = 18    # panel inner padding px
TP       = 16    # task panel inner padding px
TASK_LH  = 46    # task row height px

def _enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _primary_monitor(sct):
    """Windows guarantees the primary monitor's origin is (0, 0) in virtual
    screen space; mss's monitor list order is not guaranteed to put it
    first, so search for it explicitly instead of assuming index 1."""
    for mon in sct.monitors[1:]:
        if mon["left"] == 0 and mon["top"] == 0:
            return mon
    return sct.monitors[MONITOR_INDEX]


def _display_scale(height: int) -> tuple[int, int]:
    """Continuous icon size/gap scale, driven by screen height.

    Calibrated against two known-good manual profiles: 1080p -> 84px/7px
    and 1440p -> 112px/9px. Both fit a single line exactly, so every other
    resolution (1366x768 laptops up through 4K) is interpolated/extrapolated
    from that line instead of falling into a fixed bucket.
    Floor-rounded and clamped so odd resolutions only ever scale down from
    the formula, never up (no clutter).
    """
    icon_size = int(height * 84 / 1080)
    icon_gap  = int(height * 2 / 360 + 1)
    icon_size = max(56, min(180, icon_size))
    icon_gap  = max(5, min(14, icon_gap))
    return icon_size, icon_gap


# ─────────────────────────────────────────────── Overlay ─────────────────────
class BuckshotOverlay:
    def __init__(self):
        _enable_dpi_awareness()
        self._get_monitor()
        self._init_state()
        self.scanner = Scanner()          # kicks off background model load
        self._init_window()
        self._init_chat_window()
        self._load_images()
        self.redraw()
        self._poll_mouse()
        self.root.mainloop()

    # ── Monitor ───────────────────────────────────────────────────────────────
    def _get_monitor(self):
        global ISZ, IG
        with mss.mss() as sct:
            mon = _primary_monitor(sct)
            self.W, self.H = mon["width"], mon["height"]
            self.MX, self.MY = mon["left"], mon["top"]
        ISZ, IG = _display_scale(self.H)
        print(f"[DISPLAY] {self.W}x{self.H} | icon={ISZ}px gap={IG}px")

    # ── State ─────────────────────────────────────────────────────────────────
    def _init_state(self):
        self.player_items       = [0] * 9
        self.enemy_items        = [0] * 9
        self.live_shells        = 0
        self.blank_shells       = 0
        self.shell_sequence     = []       # 0=unknown, 1=live, 2=blank
        self.dealer_cuffed      = False
        self.saw_active         = False
        self.tasks              = [{"text": t, "done": False} for t in TASKS_INIT]
        self.max_hp_round       = 4
        self.player_hp          = self.max_hp_round
        self.enemy_hp           = self.max_hp_round
        self.tasks_scroll       = 0
        self._task_heights: list = []
        self._tasks_max_scroll  = 0
        self._alpha             = ALPHA_IDLE
        self._hot_zones: list   = []
        self._scan_state        = "idle"   # "idle" | "scanning"
        self._scan_items_state  = "idle"   # "idle" | "scanning"
        self._scanner_was_ready = False
        self._last_pin_time     = 0.0

    # ── Main overlay window ───────────────────────────────────────────────────
    def _init_window(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.W}x{self.H}+{self.MX}+{self.MY}")
        self.root.configure(bg=CHROMA)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", CHROMA)
        self.root.wm_attributes("-alpha", ALPHA_IDLE)
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

        self.cv = tk.Canvas(
            self.root, width=self.W, height=self.H,
            bg=CHROMA, highlightthickness=0,
        )
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<Button-1>",   lambda e: self._on_click(e.x, e.y, 1))
        self.cv.bind("<Button-3>",   lambda e: self._on_click(e.x, e.y, 3))
        self.cv.bind("<MouseWheel>", self._on_scroll)

        self.root.update()
        try:
            self.hwnd = int(self.root.frame(), 16)
            self._configure_hwnd(self.hwnd)
        except Exception:
            self.hwnd = None

    # ── Chat window (separate, always 100 % opaque, click-through) ───────────
    def _init_chat_window(self):
        self.chat_win = tk.Toplevel(self.root)
        self.chat_win.overrideredirect(True)
        self.chat_win.geometry(f"{self.W}x{self.H}+{self.MX}+{self.MY}")
        self.chat_win.configure(bg=CHROMA)
        self.chat_win.wm_attributes("-topmost", True)
        self.chat_win.wm_attributes("-transparentcolor", CHROMA)
        # No -alpha → defaults to 1.0 (always fully visible)

        self.chat_cv = tk.Canvas(
            self.chat_win, width=self.W, height=self.H,
            bg=CHROMA, highlightthickness=0,
        )
        self.chat_cv.pack(fill="both", expand=True)

        self.chat_win.update()
        try:
            self.chat_hwnd = int(self.chat_win.frame(), 16)
            self._configure_hwnd(self.chat_hwnd, click_through=True)
        except Exception:
            self.chat_hwnd = None

    def _configure_hwnd(self, hwnd, click_through=False):
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        style |= win32con.WS_EX_TOOLWINDOW
        if click_through:
            style |= win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
        win32gui.SetWindowPos(
            hwnd, win32con.HWND_TOPMOST,
            self.MX, self.MY, self.W, self.H,
            win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE,
        )

    def _pin_overlay_windows(self):
        try:
            self.root.wm_attributes("-topmost", True)
            self.chat_win.wm_attributes("-topmost", True)
            self.root.lift()
            self.chat_win.lift()
            if self.hwnd:
                win32gui.SetWindowPos(
                    self.hwnd, win32con.HWND_TOPMOST,
                    0, 0, 0, 0,
                    win32con.SWP_NOMOVE
                    | win32con.SWP_NOSIZE
                    | win32con.SWP_SHOWWINDOW
                    | win32con.SWP_NOACTIVATE,
                )
            if getattr(self, "chat_hwnd", None):
                win32gui.SetWindowPos(
                    self.chat_hwnd, win32con.HWND_TOPMOST,
                    0, 0, 0, 0,
                    win32con.SWP_NOMOVE
                    | win32con.SWP_NOSIZE
                    | win32con.SWP_SHOWWINDOW
                    | win32con.SWP_NOACTIVATE,
                )
        except Exception:
            pass

    # ── Images ────────────────────────────────────────────────────────────────
    def _load_images(self):
        base = _resource_path("")
        self.item_imgs:     list[ImageTk.PhotoImage] = []
        self.item_imgs_dim: list[ImageTk.PhotoImage] = []
        dark_bg = Image.new("RGB", (ISZ, ISZ), (11, 11, 20))
        for _, path in ITEMS_CONF:
            try:
                raw = Image.open(os.path.join(base, path)).resize(
                    (ISZ, ISZ), Image.LANCZOS
                ).convert("RGB")
            except Exception:
                raw = Image.new("RGB", (ISZ, ISZ), (13, 13, 28))
            self.item_imgs.append(
                ImageTk.PhotoImage(Image.blend(raw, dark_bg, 0.28))
            )
            self.item_imgs_dim.append(
                ImageTk.PhotoImage(Image.blend(raw, dark_bg, 0.76))
            )

    # ── Layout (recomputed each redraw for dynamic sizing) ────────────────────
    def _compute_layout(self):
        W, H = self.W, self.H
        sz, g = ISZ, IG

        grid_w = 5 * sz + 4 * g
        grid_h = 2 * sz + g
        hp_h   = 58
        pan_h  = PP + hp_h + 12 + grid_h + PP
        pan_w  = PP * 2 + grid_w

        py   = H - M - pan_h
        itop = py + PP + hp_h + 12

        # ── Player panel ──────────────────────────────────────────────────────
        px = M
        self.player_panel   = (px, py, px + pan_w, py + pan_h)
        self.player_hp_zone = (px + PP, py + PP,
                               px + PP + grid_w, py + PP + hp_h)
        self.player_item_zones = [
            (px + PP + (i % 5) * (sz + g),
             itop + (i // 5) * (sz + g),
             px + PP + (i % 5) * (sz + g) + sz,
             itop + (i // 5) * (sz + g) + sz)
            for i in range(9)
        ]

        # ── Enemy panel ───────────────────────────────────────────────────────
        ex = W - M - pan_w
        self.enemy_panel   = (ex, py, ex + pan_w, py + pan_h)
        self.enemy_hp_zone = (ex + PP, py + PP,
                              ex + PP + grid_w, py + PP + hp_h)
        self.enemy_item_zones = [
            (ex + PP + (i % 5) * (sz + g),
             itop + (i // 5) * (sz + g),
             ex + PP + (i % 5) * (sz + g) + sz,
             itop + (i // 5) * (sz + g) + sz)
            for i in range(9)
        ]

        # ── Bullets panel (bottom-center) ─────────────────────────────────────
        bw, bh    = 640, 100
        btn_h     = 66
        btn_gap   = 16
        lbl_h     = 40
        rst_w     = btn_h          # square reset button
        btn_gap_s = 6              # gap between action buttons
        mhp_w     = 148            # max hp button width
        sw        = (bw - mhp_w - rst_w - 3 * btn_gap_s) // 2  # equal scan button width
        total_h   = lbl_h + bh + btn_gap + btn_h + 10
        bx        = W // 2 - bw // 2
        by        = H - M - total_h

        self.live_zone  = (bx,           by + lbl_h, bx + bw // 2, by + lbl_h + bh)
        self.blank_zone = (bx + bw // 2, by + lbl_h, bx + bw,      by + lbl_h + bh)
        ba              = by + lbl_h + bh + btn_gap

        # Button order: [SCAN ROUNDS] [SCAN ITEMS] [MAX HP] [↺]
        self.btn_shells     = (bx,                                                ba, bx + sw,                                              ba + btn_h)
        self.btn_items      = (bx + sw + btn_gap_s,                              ba, bx + 2 * sw + btn_gap_s,                              ba + btn_h)
        self.maxhp_btn_zone = (bx + 2 * sw + 2 * btn_gap_s,                     ba, bx + 2 * sw + 2 * btn_gap_s + mhp_w,                  ba + btn_h)
        self.btn_reset      = (bx + 2 * sw + 2 * btn_gap_s + mhp_w + btn_gap_s, ba, bx + bw,                                             ba + btn_h)
        self.bullets_bg     = (bx - 14, by - 10, bx + bw + 14,                  ba + btn_h + 10)

        # ── Shell sequence panel (above bullets panel) ────────────────────────
        TOGGLE_ROW_H   = 30
        SEQ_H          = 90 + TOGGLE_ROW_H + 4
        SEQ_SHELL_W    = 44
        SEQ_SHELL_H    = 58
        SEQ_GAP        = 10
        seq_x1         = bx - 14
        seq_x2         = bx + bw + 14
        seq_y2         = by - 10 - 10    # 10px gap above bullets_bg
        seq_y1         = seq_y2 - SEQ_H
        self.shell_seq_panel  = (seq_x1, seq_y1, seq_x2, seq_y2)
        self._seq_shell_w     = SEQ_SHELL_W
        self._seq_shell_h     = SEQ_SHELL_H
        self._seq_gap         = SEQ_GAP

        # ── Status toggle zones (top row inside shell_seq_panel) ─────────
        tg_y1  = seq_y1 + 5
        tg_y2  = tg_y1 + TOGGLE_ROW_H - 2
        tg_pad = 8
        tg_w   = ((seq_x2 - seq_x1) - tg_pad * 3) // 2
        tg_x0  = seq_x1 + tg_pad
        self.toggle_zones = [
            (tg_x0,                  tg_y1, tg_x0 + tg_w,              tg_y2),
            (tg_x0 + tg_w + tg_pad, tg_y1, tg_x0 + 2 * tg_w + tg_pad, tg_y2),
        ]
        self._shell_area_top = tg_y2 + 4

        # Compute clickable zones for each shell in the sequence
        n = len(self.shell_sequence)
        if n > 0:
            total_sw = n * SEQ_SHELL_W + (n - 1) * SEQ_GAP
            sx0      = (seq_x1 + seq_x2) // 2 - total_sw // 2
            cy_shell = (self._shell_area_top + seq_y2) // 2
            sy1      = cy_shell - SEQ_SHELL_H // 2
            sy2      = cy_shell + SEQ_SHELL_H // 2
            self.shell_seq_zones = [
                (sx0 + i * (SEQ_SHELL_W + SEQ_GAP), sy1,
                 sx0 + i * (SEQ_SHELL_W + SEQ_GAP) + SEQ_SHELL_W, sy2)
                for i in range(n)
            ]
        else:
            self.shell_seq_zones = []

        # ── AI button zone (top-left) ─────────────────────────────────────────
        self.ai_btn_zone = (M, M, M + 90, M + 46)

        # ── Close button zone (top-right, mirrors AI button) ──────────────────
        self.close_btn_zone = (W - M - 46, M, W - M, M + 46)

        # ── Tasks panel (top-left, stacked under the AI button) ───────────────
        task_w      = 480
        tasks_gap   = 12   # vertical gap below the AI button
        wrap_w      = task_w - TP * 2 - 34   # available text width per row
        self._task_heights = []
        for task in self.tasks:
            tmp = self.cv.create_text(0, 0, text=task["text"],
                                      font=F_SM, anchor="nw", width=wrap_w)
            bb = self.cv.bbox(tmp)
            self.cv.delete(tmp)
            th = (bb[3] - bb[1]) if bb else 20
            self._task_heights.append(max(TASK_LH, th + 20))

        row_sum   = sum(self._task_heights) if self._task_heights else TASK_LH
        content_h = TP + 28 + row_sum + TP
        MAX_PH    = 560
        tasks_h   = min(content_h, MAX_PH)
        self._tasks_max_scroll = max(0, content_h - tasks_h)
        self.tasks_scroll      = max(0, min(self.tasks_scroll, self._tasks_max_scroll))
        tasks_y1        = self.ai_btn_zone[3] + tasks_gap
        self.tasks_area = (M, tasks_y1, M + task_w, tasks_y1 + tasks_h)

        # Chat is on its own always-visible window — not in hot zones
        # maxhp_btn_zone is inside bullets_bg so it's covered by that zone
        self._hot_zones = [
            self.player_panel,
            self.enemy_panel,
            self.bullets_bg,
            self.shell_seq_panel,
            self.tasks_area,
            self.ai_btn_zone,
            self.close_btn_zone,
        ]

    # ── Draw helpers ──────────────────────────────────────────────────────────
    def _r(self, x1, y1, x2, y2, fill="", outline="", w=1):
        return self.cv.create_rectangle(
            x1, y1, x2, y2, fill=fill, outline=outline, width=w
        )

    def _t(self, x, y, text, fill, font=F, anchor="nw", wrap=0):
        kw = {"width": wrap} if wrap else {}
        return self.cv.create_text(
            x, y, text=text, fill=fill, font=font, anchor=anchor, **kw
        )

    def _l(self, x1, y1, x2, y2, fill, w=1):
        return self.cv.create_line(x1, y1, x2, y2, fill=fill, width=w)

    def _bolt(self, cx, cy, size, color):
        h, ww = size / 2, size * 0.38
        pts = (
            cx + ww * 0.40,  cy - h,
            cx - ww * 0.20,  cy - h * 0.08,
            cx + ww * 0.28,  cy - h * 0.08,
            cx - ww * 0.40,  cy + h,
            cx + ww * 0.20,  cy + h * 0.08,
            cx - ww * 0.28,  cy + h * 0.08,
        )
        self.cv.create_polygon(*pts, fill=color, outline="")

    @staticmethod
    def _in(x, y, zone):
        return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

    # ── Redraw ────────────────────────────────────────────────────────────────
    def redraw(self):
        self.cv.delete("all")
        self.chat_cv.delete("all")
        self._compute_layout()
        self._draw_ai_btn()
        self._draw_close_btn()
        self._draw_tasks()
        self._draw_player_panel()
        self._draw_enemy_panel()
        self._draw_bullets_panel()
        self._draw_shell_sequence()
        self._pin_overlay_windows()

    # ── AI button (drawn on both layers so main cv captures clicks) ──────────
    def _draw_ai_btn(self, canvases=None):
        if canvases is None:
            canvases = (self.cv, self.chat_cv)
        x1, y1, x2, y2 = self.ai_btn_zone
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for canvas in canvases:
            canvas.create_rectangle(
                x1, y1, x2, y2, fill=C["btn_bg"], outline=C["green_hi"], width=2,
            )
            canvas.create_line(x1, y1, x2, y1, fill=C["green_hi"], width=2)
            canvas.create_text(
                cx, cy, text="AI", fill=C["green_hi"],
                font=("Courier New", 18, "bold"), anchor="center",
            )

    # ── Close button (top-right; drawn on both layers so main cv captures
    #    clicks — see the note in _draw_tasks about chat_cv being click-through) ─
    def _draw_close_btn(self):
        x1, y1, x2, y2 = self.close_btn_zone
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for cv in (self.cv, self.chat_cv):
            cv.create_rectangle(x1, y1, x2, y2, fill=C["btn_bg"], outline=C["red_hi"], width=2)
            cv.create_line(x1, y1, x2, y1, fill=C["red_hi"], width=2)
            cv.create_text(
                cx, cy, text="X", fill=C["red_hi"],
                font=("Courier New", 18, "bold"), anchor="center",
            )

    # ── Tasks panel (drawn on the always-opaque chat_cv layer, like AI button) ─
    def _draw_tasks(self):
        # Drawn on both layers: chat_cv is WS_EX_TRANSPARENT (click-through),
        # so every click actually lands on cv underneath — cv needs real
        # (non-chroma) pixels here too, or the click falls straight through
        # to whatever's behind the overlay. chat_cv's copy is what stays
        # always-opaque on top; cv's copy is invisible-in-effect but is what
        # actually stops/receives the click and scroll-wheel events.
        x1, y1, x2, y2 = self.tasks_area
        hx = (x1 + x2) // 2

        for cv in (self.cv, self.chat_cv):
            cv.create_rectangle(x1, y1, x2, y2, fill=C["bg"], outline=C["border"])
            cv.create_text(hx, y1 + TP, text="── TASKS ──", fill=C["text_dim"],
                            font=F_HDR, anchor="n")
            cv.create_line(x1 + TP, y1 + TP + 26, x2 - TP, y1 + TP + 26, fill=C["border"])

            if not self.tasks:
                row_cy = y1 + TP + 28 + TASK_LH // 2
                cv.create_text((x1 + x2) // 2, row_cy, text="no tasks",
                               fill=C["text_dim"], font=F_SM, anchor="center")
                continue

            wrap_w      = (x2 - x1) - TP * 2 - 34
            content_top = y1 + TP + 28 + 4
            vis_top     = content_top
            vis_bot     = y2 - TP
            curr_y      = content_top - self.tasks_scroll

            for i, task in enumerate(self.tasks):
                rh     = self._task_heights[i] if i < len(self._task_heights) else TASK_LH
                row_b  = curr_y + rh

                if row_b < vis_top:
                    curr_y += rh
                    continue
                if curr_y > vis_bot:
                    break

                pfx = "◇"
                col = C["text"]
                pc  = C["text_dim"]

                # Prefix glyph — centred vertically in the row
                pfx_y = max(vis_top, min(vis_bot, curr_y + rh // 2))
                cv.create_text(x1 + TP + 4, pfx_y, text=pfx, fill=pc,
                                font=F_SM, anchor="w")

                # Wrapped text — anchored to top of the row (clipped to vis_top)
                text_y = max(vis_top, curr_y + 8)
                cv.create_text(
                    x1 + TP + 30, text_y,
                    text=task["text"], fill=col,
                    font=F_SM, anchor="nw", width=wrap_w,
                )
                curr_y += rh

            # Scroll arrows
            if self._tasks_max_scroll > 0:
                if self.tasks_scroll > 0:
                    cv.create_text(hx, y1 + TP + 30, text="▲ scroll", fill=C["text_dim"],
                                    font=("Courier New", 11), anchor="n")
                if self.tasks_scroll < self._tasks_max_scroll:
                    cv.create_text(hx, y2 - 4, text="▼ scroll", fill=C["text_dim"],
                                    font=("Courier New", 11), anchor="s")

    # ── Player panel ──────────────────────────────────────────────────────────
    def _draw_player_panel(self):
        x1, y1, x2, y2 = self.player_panel
        self._r(x1, y1, x2, y2, fill=C["bg"], outline=C["border"])
        self._l(x1, y1, x1 + (x2 - x1) * 2 // 5, y1, C["green_hi"], 2)
        self._r(x1, y1, x1 + 5, y1 + 5, fill=C["green_hi"])

        self._draw_hp(self.player_hp_zone, self.player_hp,
                      C["hp_fill"], C["hp_empty"], "HEALTH")
        for i, z in enumerate(self.player_item_zones):
            self._draw_item(z, i, self.player_items[i])

    # ── Enemy panel ───────────────────────────────────────────────────────────
    def _draw_enemy_panel(self):
        x1, y1, x2, y2 = self.enemy_panel
        self._r(x1, y1, x2, y2, fill=C["bg"], outline=C["border"])
        self._l(x2 - (x2 - x1) * 2 // 5, y1, x2, y1, C["red_hi"], 2)
        self._r(x2 - 5, y1, x2, y1 + 5, fill=C["red_hi"])

        self._draw_hp(self.enemy_hp_zone, self.enemy_hp,
                      C["ehp_fill"], C["ehp_empty"], "ENEMY HP")
        for i, z in enumerate(self.enemy_item_zones):
            self._draw_item(z, i, self.enemy_items[i])

    # ── HP bar ────────────────────────────────────────────────────────────────
    def _draw_hp(self, zone, hp, fill_c, empty_c, label):
        x1, y1, x2, y2 = zone
        cy = (y1 + y2) // 2

        self._bolt(x1 + 14, cy, 26, fill_c)
        self._t(x1 + 30, cy, label, fill_c, F_HDR, anchor="w")

        pw, ph, pg = 24, 24, 7
        slots = self.max_hp_round
        total = slots * pw + (slots - 1) * pg
        sx = x2 - total - 6
        sy = cy - ph // 2
        for i in range(slots):
            ix = sx + i * (pw + pg)
            fc = fill_c if i < hp else empty_c
            oc = fill_c if i < hp else C["border"]
            self._r(ix, sy, ix + pw, sy + ph, fill=fc, outline=oc)

    # ── Item slot ─────────────────────────────────────────────────────────────
    def _draw_item(self, zone, idx, count):
        x1, y1, x2, y2 = zone
        self._r(x1, y1, x2, y2, fill=C["item_bg"], outline=C["item_brd"])
        imgs = self.item_imgs if count > 0 else self.item_imgs_dim
        if idx < len(imgs):
            self.cv.create_image(x1, y1, image=imgs[idx], anchor="nw")

        bw_, bh_ = 34, 30
        bx1, by1 = x2 - bw_, y2 - bh_
        self._r(bx1, by1, x2, y2, fill=C["item_bg"])
        col = C["text_hi"] if count > 0 else C["text_dim"]
        self._t((bx1 + x2) // 2, (by1 + y2) // 2,
                str(count), col, F_NUM, anchor="center")

    # ── Status toggles (cuffs / saw) ────────────────────────────────────────────
    def _draw_status_toggles(self):
        labels  = ("SAW", "D CUFF")
        actives = (self.saw_active, self.dealer_cuffed)
        colors  = (C["green"], C["red"])
        hi_cols = (C["green_hi"], C["red_hi"])
        for i, (zone, lbl, on, col, hi) in enumerate(
            zip(self.toggle_zones, labels, actives, colors, hi_cols)
        ):
            zx1, zy1, zx2, zy2 = zone
            if on:
                self._r(zx1, zy1, zx2, zy2, fill=col, outline=hi)
                self._t((zx1+zx2)//2, (zy1+zy2)//2, lbl, C["text_hi"],
                        F_HDR, anchor="center")
            else:
                self._r(zx1, zy1, zx2, zy2, fill=C["bg"], outline=C["border"])
                self._t((zx1+zx2)//2, (zy1+zy2)//2, lbl, C["text_dim"],
                        F_HDR, anchor="center")

    # ── Shell sequence panel ───────────────────────────────────────────────────
    def _draw_shell_sequence(self):
        x1, y1, x2, y2 = self.shell_seq_panel
        self._r(x1, y1, x2, y2, fill=C["bg"], outline=C["border"])
        span = (x2 - x1) // 5
        self._l(x1, y1, x1 + span, y1, C["text_dim"], 2)
        self._l(x2 - span, y1, x2, y1, C["text_dim"], 2)

        self._draw_status_toggles()

        sw, sh, sg = self._seq_shell_w, self._seq_shell_h, self._seq_gap
        n = len(self.shell_sequence)

        if n == 0:
            cy_no = (self._shell_area_top + y2) // 2
            self._t((x1 + x2) // 2, cy_no,
                    "NO SHELLS", C["text_dim"], F_HDR, anchor="center")
            return

        total_sw = n * sw + (n - 1) * sg
        sx0      = (x1 + x2) // 2 - total_sw // 2
        cy_shell = (self._shell_area_top + y2) // 2
        sy1      = cy_shell - sh // 2
        sy2      = cy_shell + sh // 2

        for i, state in enumerate(self.shell_sequence):
            sx1 = sx0 + i * (sw + sg)
            sx2 = sx1 + sw
            if state == 1:
                bg, brd = C["red"], C["red_hi"]
            elif state == 2:
                bg, brd = C["blue"], C["blue_hi"]
            else:
                bg, brd = "#1A1A2A", "#2A2A40"

            self._r(sx1, sy1, sx2, sy2, fill=bg, outline=brd)
            # Bullet tip (small arc at top)
            self.cv.create_arc(
                sx1 + 2, sy1 - 6, sx2 - 2, sy1 + 10,
                start=0, extent=180, fill=brd, outline="",
            )
            # Dim number centered in shell
            num_col = C["text_dim"] if state == 0 else "#604040" if state == 1 else "#404860"
            self._t((sx1 + sx2) // 2, (sy1 + sy2) // 2,
                    str(i + 1), num_col, ("Courier New", 13), anchor="center")

        # Arrow under first shell
        if n > 0:
            ax = sx0 + sw // 2
            self._t(ax, sy2 + 4, "▲", C["text_dim"], ("Courier New", 11), anchor="n")

    # ── Bullets panel ─────────────────────────────────────────────────────────
    def _draw_bullets_panel(self):
        bx1, by1, bx2, by2 = self.bullets_bg
        self._r(bx1, by1, bx2, by2, fill=C["bg"], outline=C["border"])
        span = (bx2 - bx1) // 4
        self._l(bx1, by1, bx1 + span, by1, C["red_hi"], 2)
        self._l(bx2 - span, by1, bx2, by1, C["blue_hi"], 2)

        lx1, ly1, lx2, ly2 = self.live_zone
        rx1, ry1, rx2, ry2 = self.blank_zone

        self._r(lx1, ly1, lx2, ly2, fill=C["red"], outline=C["border_hi"])
        self._t((lx1 + lx2) // 2, ly1 - 22,
                "LIVE", C["red_hi"], F_HDR, anchor="center")
        self._t((lx1 + lx2) // 2, (ly1 + ly2) // 2,
                str(self.live_shells), C["text_hi"], F_LG, anchor="center")

        self._r(rx1, ry1, rx2, ry2, fill=C["blue"], outline=C["border_hi"])
        self._t((rx1 + rx2) // 2, ry1 - 22,
                "BLANK", C["blue_hi"], F_HDR, anchor="center")
        self._t((rx1 + rx2) // 2, (ry1 + ry2) // 2,
                str(self.blank_shells), C["text_hi"], F_LG, anchor="center")

        mx = (lx2 + rx1) // 2
        self._l(mx, ly1, mx, ly2, C["border_hi"], 2)

        if self._scan_state == "scanning":
            shell_label, shell_state = "SCANNING...", "scanning"
        else:
            shell_label, shell_state = "SCAN ROUNDS", "idle"

        items_label = "SCANNING..." if self._scan_items_state == "scanning" else "SCAN ITEMS"
        items_state = self._scan_items_state

        self._draw_btn(self.btn_shells,     shell_label,                    shell_state)
        self._draw_btn(self.btn_items,      items_label,                    items_state)
        self._draw_btn(self.btn_reset,      "↺",                            state="reset")
        self._draw_btn(self.maxhp_btn_zone, f"MAX HP: {self.max_hp_round}", state="maxhp")

    def _draw_btn(self, zone, text, state="idle"):
        x1, y1, x2, y2 = zone
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if state == "scanning":
            bg, brd, tc = C["green"],    C["green_hi"], C["text_hi"]
        elif state == "loading":
            bg, brd, tc = C["btn_bg"],   C["border"],   C["text_dim"]
        elif state == "reset":
            bg, brd, tc = C["btn_bg"],   C["red_hi"],   C["red_hi"]
        elif state == "maxhp":
            bg, brd, tc = C["btn_bg"],   C["blue_hi"],  C["blue_hi"]
        else:
            bg, brd, tc = C["btn_bg"],   C["btn_brd"],  C["btn_text"]
        self._r(x1, y1, x2, y2, fill=bg, outline=brd)
        top_line = C["red_hi"] if state == "reset" else (C["blue_hi"] if state == "maxhp" else C["border_hi"])
        self._l(x1, y1, x2, y1, top_line)
        font = ("Courier New", 26, "bold") if state == "reset" else F_HDR
        self._t(cx, cy, text, tc, font, anchor="center")

    def _hide_overlay_for_scan(self):
        self.root.wm_attributes("-alpha", 0)
        self.chat_win.wm_attributes("-alpha", 0)
        self.root.update()

    def _restore_overlay_after_scan(self):
        self.root.wm_attributes("-alpha", self._alpha)
        self.chat_win.wm_attributes("-alpha", 1.0)
        self.redraw()
        self._pin_overlay_windows()

    # ── Scan items ────────────────────────────────────────────────────────────
    def _scan_items(self):
        if self._scan_items_state == "scanning" or not self.scanner.is_ready:
            return
        self._scan_items_state = "scanning"
        self._hide_overlay_for_scan()

        def _worker():
            import time
            time.sleep(0.15)  # wait for OS to actually hide the windows
            result = self.scanner.scan_items()
            def _apply():
                if result:
                    self.player_items = result["player"]
                    self.enemy_items  = result["enemy"]
                    p_log = ", ".join(
                        f"{name}={result['player'][i]}"
                        for i, (name, _) in enumerate(ITEMS_CONF)
                        if result['player'][i] > 0
                    ) or "none"
                    e_log = ", ".join(
                        f"{name}={result['enemy'][i]}"
                        for i, (name, _) in enumerate(ITEMS_CONF)
                        if result['enemy'][i] > 0
                    ) or "none"
                    print(f"[SCAN ITEMS]  player: {p_log}")
                    print(f"[SCAN ITEMS]  enemy:  {e_log}")
                else:
                    print("[SCAN ITEMS]  no result returned (model not ready?)")
                self._scan_items_state = "idle"
                self._restore_overlay_after_scan()
            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Reset state ───────────────────────────────────────────────────────────
    def _reset_state(self):
        self.player_items  = [0] * 9
        self.enemy_items   = [0] * 9
        self.max_hp_round  = 4
        self.player_hp     = self.max_hp_round
        self.enemy_hp      = self.max_hp_round
        self.live_shells   = 0
        self.blank_shells  = 0
        self.shell_sequence = []
        self.dealer_cuffed = False
        self.saw_active    = False
        print("[RESET] State cleared — items/shells 0, player & dealer HP = MAX HP (4).")
        self.redraw()

    # ── Scan rounds ───────────────────────────────────────────────────────────
    def _scan_rounds(self):
        if self._scan_state == "scanning" or not self.scanner.is_ready:
            return
        self._scan_state = "scanning"
        self._hide_overlay_for_scan()

        def _worker():
            import time
            time.sleep(0.15)  # wait for OS to actually hide the windows
            result = self.scanner.scan()
            def _apply():
                if result:
                    self.live_shells   = result["live"]
                    self.blank_shells  = result["blank"]
                    total = result["live"] + result["blank"]
                    self.shell_sequence = [0] * total
                    self.dealer_cuffed = False
                    self.saw_active    = False
                    print(f"[SCAN]  live={result['live']}  blank={result['blank']}")
                else:
                    print("[SCAN]  no result returned (model not ready?)")
                self._scan_state = "idle"
                self._restore_overlay_after_scan()
            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    # ── AI button ─────────────────────────────────────────────────────────────
    def _on_ai_click(self):
        self.tasks = [{"text": "Calculating...", "done": False}]
        self.redraw()

        php    = self.player_hp
        ehp    = self.enemy_hp
        pitems = self.player_items[:]
        eitems = self.enemy_items[:]
        shells = self.shell_sequence[:]
        live   = self.live_shells
        blank  = self.blank_shells
        mhp    = self.max_hp_round
        dc     = self.dealer_cuffed
        saw    = self.saw_active

        def _engine_worker():
            from ai_engine import run_ai
            tasks = run_ai(php, ehp, pitems, eitems, shells, live, blank, mhp,
                           e_cuffed=dc, saw_active=saw)

            def _show_tasks():
                self.tasks        = [{"text": t, "done": False} for t in tasks]
                self.tasks_scroll = 0
                self.redraw()

            self.root.after(0, _show_tasks)

        threading.Thread(target=_engine_worker, daemon=True).start()

    # ── Click routing ─────────────────────────────────────────────────────────
    def _on_click(self, x, y, btn):
        d = +1 if btn == 1 else -1

        if btn == 1 and self._in(x, y, self.ai_btn_zone):
            self._on_ai_click(); return

        if btn == 1 and self._in(x, y, self.close_btn_zone):
            self.root.destroy(); return

        if self._in(x, y, self.maxhp_btn_zone):
            vals = [2, 3, 4]
            cur_idx = vals.index(self.max_hp_round) if self.max_hp_round in vals else 2
            self.max_hp_round = vals[(cur_idx + d) % len(vals)]
            self.player_hp = self.max_hp_round
            self.enemy_hp = self.max_hp_round
            self.redraw(); return

        if self._in(x, y, self.player_hp_zone):
            self.player_hp = max(0, min(self.max_hp_round, self.player_hp + d))
            self.redraw(); return
        if self._in(x, y, self.enemy_hp_zone):
            self.enemy_hp = max(0, min(self.max_hp_round, self.enemy_hp + d))
            self.redraw(); return

        if self._in(x, y, self.live_zone):
            new_val = max(0, min(MAX_BULLETS, self.live_shells + d))
            if new_val > self.live_shells:
                self.shell_sequence.append(0)
            elif new_val < self.live_shells and self.shell_sequence:
                self.shell_sequence.pop(0)
            self.live_shells = new_val
            self.redraw(); return
        if self._in(x, y, self.blank_zone):
            new_val = max(0, min(MAX_BULLETS, self.blank_shells + d))
            if new_val > self.blank_shells:
                self.shell_sequence.append(0)
            elif new_val < self.blank_shells and self.shell_sequence:
                self.shell_sequence.pop(0)
            self.blank_shells = new_val
            self.redraw(); return

        # Status toggles (left-click only)
        if btn == 1:
            for i, z in enumerate(self.toggle_zones):
                if self._in(x, y, z):
                    if i == 0:
                        self.saw_active = not self.saw_active
                    else:
                        self.dealer_cuffed = not self.dealer_cuffed
                    self.redraw(); return

        # Shell sequence slot cycling (left-click only)
        if btn == 1:
            for i, z in enumerate(self.shell_seq_zones):
                if self._in(x, y, z):
                    self.shell_sequence[i] = (self.shell_sequence[i] + 1) % 3
                    self.redraw(); return

        for i, z in enumerate(self.player_item_zones):
            if self._in(x, y, z):
                self.player_items[i] = max(0, min(MAX_ITEMS, self.player_items[i] + d))
                self.redraw(); return
        for i, z in enumerate(self.enemy_item_zones):
            if self._in(x, y, z):
                self.enemy_items[i] = max(0, min(MAX_ITEMS, self.enemy_items[i] + d))
                self.redraw(); return
        if btn == 1 and self._in(x, y, self.btn_shells):
            self._scan_rounds(); return
        if btn == 1 and self._in(x, y, self.btn_items):
            self._scan_items(); return
        if btn == 1 and self._in(x, y, self.btn_reset):
            self._reset_state(); return

    # ── Scroll handler ────────────────────────────────────────────────────────
    def _on_scroll(self, event):
        x, y = event.x, event.y
        if self._in(x, y, self.tasks_area) and self._tasks_max_scroll > 0:
            self.tasks_scroll = max(
                0, min(self._tasks_max_scroll, self.tasks_scroll - event.delta // 4)
            )
            self.redraw()

    # ── Hover / smooth alpha fade ─────────────────────────────────────────────
    def _poll_mouse(self):
        try:
            mx, my = win32api.GetCursorPos()
            rx, ry = mx - self.MX, my - self.MY
            over   = any(
                z[0] <= rx <= z[2] and z[1] <= ry <= z[3]
                for z in self._hot_zones
            )
            target = ALPHA_HOVER if over else ALPHA_IDLE
            diff   = target - self._alpha
            if abs(diff) > 0.001:
                step        = min(abs(diff), ALPHA_STEP) * (1 if diff > 0 else -1)
                self._alpha = round(self._alpha + step, 3)
                self.root.wm_attributes("-alpha", self._alpha)

            now = time.time()
            if now - self._last_pin_time >= 0.5:
                self._last_pin_time = now
                self._pin_overlay_windows()

            # Redraw once when model finishes loading (button label update)
            now_ready = self.scanner.is_ready
            if now_ready != self._scanner_was_ready:
                self._scanner_was_ready = now_ready
                self.redraw()
        except Exception:
            pass
        self.root.after(40, self._poll_mouse)


if __name__ == "__main__":
    BuckshotOverlay()
