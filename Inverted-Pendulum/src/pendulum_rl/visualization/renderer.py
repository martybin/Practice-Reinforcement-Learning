"""Pygame renderer for the translating-support pendulum.

The scene is a front elevation of the mechanism: two parallel horizontal guide rods held
by end posts, a carriage whose linear bearings ride on both rods, and the pendulum hinged
on the carriage face. The renderer only draws; it owns no physics and never steps the
simulation, so it can be used live (``SimulatorApp``), from ``env.render()`` or headless
to produce screenshots.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..physics import TH, THD, X, XD, MechanismModel
from ..physics.dynamics import wrap_angle

# ---------------------------------------------------------------- palette (restrained, one accent)
BG_TOP = (11, 18, 36)
BG_BOTTOM = (7, 11, 22)
PANEL = (16, 26, 50)
PANEL_BORDER = (31, 45, 77)
GRID_MINOR = (5, 8, 16)           # added on top of the background, so values are small
GRID_MAJOR = (10, 15, 30)
TEXT = (230, 237, 247)
TEXT_DIM = (139, 155, 184)
TEXT_FAINT = (91, 107, 136)
ACCENT = (56, 189, 248)         # cyan: reference line, command, joint
STEEL = (154, 168, 189)
STEEL_LIGHT = (219, 228, 240)
STEEL_DARK = (74, 86, 107)
CARRIAGE = (42, 59, 96)
CARRIAGE_EDGE = (78, 100, 144)
BEARING = (59, 77, 120)
ROD_COLOR = (223, 231, 245)
BOB_COLOR = (245, 158, 11)       # amber
LIMIT_COLOR = (248, 113, 113)
PHASE_COLORS = {
    "swinging_up": (96, 165, 250),
    "stabilizing": (251, 191, 36),
    "balanced": (52, 211, 153),
    "failed": (248, 113, 113),
}
PHASE_LABELS = {"swinging_up": "SWINGING UP", "stabilizing": "STABILIZING", "balanced": "BALANCED", "failed": "FAILED"}

HUD_WIDTH = 400
MONO_FONTS = "consolas,dejavusansmono,menlo,couriernew,monospace"
SANS_FONTS = "segoeui,helveticaneue,dejavusans,arial,helvetica"


@dataclass
class HudInfo:
    """Everything the HUD shows besides the physical state."""

    command: float = 0.0
    time: float = 0.0
    reward: float = 0.0
    phase: str = "swinging_up"
    mode_label: str = ""
    paused: bool = False
    toast: tuple[str, float] | None = None       # (message, opacity 0..1)
    show_help: bool = False
    controls: list[tuple[str, str]] = field(default_factory=list)
    notice: str = ""                             # persistent line under the status pill (e.g. "press R to reset")


def lerp_color(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


class PendulumView:
    """Draws one frame of the mechanism plus HUD on a pygame surface."""

    def __init__(self, model: MechanismModel, size: tuple[int, int] = (1280, 760), headless: bool = False,
                 trail_seconds: float = 1.5, show_trail: bool = True, caption: str = "Moving-support pendulum"):
        if model.geometry != "translating_support":
            raise NotImplementedError(
                f"No renderer for mechanism geometry '{model.geometry}'. Add a drawing routine to "
                f"visualization/renderer.py (a tilting or rotating support must not be drawn as a sliding cart).")
        import pygame
        from pygame import gfxdraw
        if headless and not pygame.display.get_init():
            os.environ["SDL_VIDEODRIVER"] = "dummy"   # offscreen: no window, but the event queue still works
        self.pg, self.gfx = pygame, gfxdraw
        self.model = model
        self.size = size
        self.headless = headless
        self.show_trail = show_trail
        self.trail_seconds = trail_seconds
        self._trail: deque = deque()          # (screen point, simulation time); pruned to the last `trail_seconds`

        pygame.font.init()
        pygame.display.init()
        if headless:
            self.screen = pygame.Surface(size)
        else:
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self._fonts = {
            "title": self._font(SANS_FONTS, 17, bold=True),
            "label": self._font(SANS_FONTS, 13),
            "small": self._font(SANS_FONTS, 12),
            "value": self._font(MONO_FONTS, 15, bold=True),
            "big": self._font(SANS_FONTS, 20, bold=True),
            "huge": self._font(SANS_FONTS, 34, bold=True),
            "tick": self._font(MONO_FONTS, 11),
        }
        self._text_cache: dict = {}
        self._layout()
        self._static = self._build_static()

    # ------------------------------------------------------------------ setup
    def _font(self, names: str, size: int, bold: bool = False):
        font = self.pg.font.SysFont(names, size, bold=bold)
        return font

    def _layout(self) -> None:
        w, h = self.size
        p = self.model.params
        self.scene_w = w - HUD_WIDTH
        half_world = p.rail_half_length + 0.17
        self.scale = min((self.scene_w - 60) / (2 * half_world), (h - 150) / (2 * p.pendulum_length + 0.3))
        self.cx = self.scene_w // 2
        self.cy = int(h * 0.49)
        self.rod_dy = int(0.05 * self.scale)                 # rods sit +-5 cm from the pivot line
        self.post_x = p.rail_half_length + 0.12              # end-post centre, metres from the rail centre
        self.hud_x = self.scene_w + 8

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.cx + x * self.scale, self.cy - y * self.scale

    # ------------------------------------------------------------------ static layer
    def _build_static(self):
        pg = self.pg
        w, h = self.size
        surf = pg.Surface((w, h))
        for yy in range(h):                                   # vertical gradient background
            surf.fill(lerp_color(BG_TOP, BG_BOTTOM, yy / h), (0, yy, w, 1))

        # engineering grid: 5 cm minor, 25 cm major, centred on the rail centre / pivot line
        minor, major = 0.05 * self.scale, 0.25 * self.scale
        grid = pg.Surface((self.scene_w, h), pg.SRCALPHA)
        for k in range(-40, 41):
            gx = self.cx + k * minor
            if 0 <= gx < self.scene_w:
                col = GRID_MAJOR if k % 5 == 0 else GRID_MINOR
                pg.draw.line(grid, (*col, 255), (gx, 0), (gx, h))
            gy = self.cy + k * minor
            if 0 <= gy < h:
                col = GRID_MAJOR if k % 5 == 0 else GRID_MINOR
                pg.draw.line(grid, (*col, 255), (0, gy), (self.scene_w, gy))
        surf.blit(grid, (0, 0), special_flags=pg.BLEND_ADD)

        self._draw_ruler(surf)
        self._draw_limits(surf)
        self._draw_rails(surf)
        # scene captions
        self._blit_text(surf, "MOVING-SUPPORT PENDULUM", (24, 20), "title", TEXT)
        self._blit_text(surf, "Translating support on two parallel guide rods  |  swing-up and stabilization",
                        (24, 44), "small", TEXT_DIM)
        return surf

    def _draw_ruler(self, surf) -> None:
        pg = self.pg
        p = self.model.params
        y = self.size[1] - 44
        x0, x1 = self.to_screen(-p.rail_half_length, 0)[0], self.to_screen(p.rail_half_length, 0)[0]
        pg.draw.line(surf, STEEL_DARK, (x0, y), (x1, y), 1)
        n = int(round(2 * p.rail_half_length / 0.1))
        for i in range(n + 1):
            xm = -p.rail_half_length + i * 0.1
            sx = self.to_screen(xm, 0)[0]
            major = abs(round(xm / 0.2) * 0.2 - xm) < 1e-6
            pg.draw.line(surf, STEEL if major else STEEL_DARK, (sx, y), (sx, y + (9 if major else 5)), 1)
            if major:
                label = f"{xm:+.1f}" if abs(xm) > 1e-6 else "0"
                self._blit_text(surf, label, (sx, y + 12), "tick", TEXT_FAINT, anchor="midtop")
        self._blit_text(surf, "support position x [m]", (x1 + 12, y - 7), "tick", TEXT_FAINT)

    def _draw_limits(self, surf) -> None:
        pg = self.pg
        p = self.model.params
        top, bottom = self.cy - int(0.34 * self.scale), self.cy + int(0.34 * self.scale)
        for side in (-1, 1):
            lx = self.to_screen(side * p.rail_half_length, 0)[0]
            for yy in range(top, bottom, 12):                  # dashed limit line
                pg.draw.line(surf, LIMIT_COLOR, (lx, yy), (lx, min(yy + 6, bottom)), 1)
            px = self.to_screen(side * self.post_x, 0)[0]
            band = pg.Surface((abs(px - lx), bottom - top), pg.SRCALPHA)
            band.fill((*LIMIT_COLOR, 22))
            surf.blit(band, (min(px, lx), top))
        self._blit_text(surf, "TRAVEL LIMIT", (self.to_screen(-p.rail_half_length, 0)[0], top - 14), "tick",
                        LIMIT_COLOR, anchor="midbottom")
        self._blit_text(surf, "TRAVEL LIMIT", (self.to_screen(p.rail_half_length, 0)[0], top - 14), "tick",
                        LIMIT_COLOR, anchor="midbottom")

    def _draw_rails(self, surf) -> None:
        pg = self.pg
        p = self.model.params
        x_end = self.post_x + 0.02
        xs0, xs1 = int(self.to_screen(-x_end, 0)[0]), int(self.to_screen(x_end, 0)[0])
        thickness = 12
        profile = (0.42, 0.62, 0.9, 1.0, 0.96, 0.82, 0.66, 0.52, 0.42, 0.34, 0.28, 0.24)
        for row_y in (self.cy - self.rod_dy, self.cy + self.rod_dy):
            for i in range(thickness):                        # cylinder shading
                shade = profile[i * len(profile) // thickness]
                col = lerp_color(STEEL_DARK, STEEL_LIGHT, shade)
                pg.draw.line(surf, col, (xs0, row_y - thickness // 2 + i), (xs1, row_y - thickness // 2 + i))
            pg.draw.line(surf, (255, 255, 255), (xs0 + 8, row_y - thickness // 2 + 3), (xs1 - 8, row_y - thickness // 2 + 3))
        # end posts
        post_h = 2 * self.rod_dy + 56
        for side in (-1, 1):
            px = int(self.to_screen(side * self.post_x, 0)[0])
            rect = pg.Rect(0, 0, 26, post_h)
            rect.center = (px, self.cy)
            pg.draw.rect(surf, (36, 49, 80), rect, border_radius=5)
            pg.draw.rect(surf, CARRIAGE_EDGE, rect, width=1, border_radius=5)
            pg.draw.line(surf, (66, 84, 126), (rect.left + 4, rect.top + 4), (rect.left + 4, rect.bottom - 4))
            for by in (rect.top + 10, rect.bottom - 10):       # bolts
                self.gfx.filled_circle(surf, px, by, 3, STEEL_DARK)
                self.gfx.aacircle(surf, px, by, 3, STEEL)
        # rod labels
        label_x = self.to_screen(-self.post_x, 0)[0] + 22
        self._blit_text(surf, "GUIDE ROD", (label_x, self.cy - self.rod_dy - 10), "tick", TEXT_FAINT, anchor="bottomleft")
        self._blit_text(surf, "GUIDE ROD", (label_x, self.cy + self.rod_dy + 10), "tick", TEXT_FAINT, anchor="topleft")

    # ------------------------------------------------------------------ text helpers
    def _text(self, text: str, font: str, color):
        key = (text, font, color)
        surf = self._text_cache.get(key)
        if surf is None:
            surf = self._fonts[font].render(text, True, color)
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            self._text_cache[key] = surf
        return surf

    def _blit_text(self, target, text: str, pos, font: str, color, anchor: str = "topleft"):
        surf = self._text(text, font, color)
        rect = surf.get_rect()
        setattr(rect, anchor, (int(pos[0]), int(pos[1])))
        target.blit(surf, rect)
        return rect

    # ------------------------------------------------------------------ dynamic drawing
    def reset_trail(self) -> None:
        self._trail.clear()

    def draw(self, state: np.ndarray, hud: HudInfo | None = None, *, show: bool = False, fps: int = 0,
             as_array: bool = False, advance_trail: bool = True):
        """Render one frame. ``show`` flips the display (window mode); ``as_array`` returns an RGB array."""
        hud = hud or HudInfo()
        pg = self.pg
        scr = self.screen
        scr.blit(self._static, (0, 0))

        x, xd, theta, theta_dot = float(state[X]), float(state[XD]), float(state[TH]), float(state[THD])
        px, py = self.to_screen(x, 0.0)
        L = self.model.params.pendulum_length
        tip = (px + L * self.scale * math.sin(theta), py - L * self.scale * math.cos(theta))
        if advance_trail:
            if self._trail and hud.time < self._trail[-1][1]:      # simulation was reset
                self._trail.clear()
            self._trail.append((tip, hud.time))
            while hud.time - self._trail[0][1] > self.trail_seconds:
                self._trail.popleft()
        phase_color = PHASE_COLORS.get(hud.phase, TEXT_DIM)

        scr.set_clip(pg.Rect(0, 0, self.scene_w, self.size[1]))     # scene drawing must not leak onto the HUD
        self._draw_reference(px, py, theta)
        if self.show_trail:
            self._draw_trail()
        self._draw_pendulum(px, py, tip, theta, phase_color)
        self._draw_carriage(px, py, hud.command)
        self._draw_pivot(px, py)
        scr.set_clip(None)
        self._draw_hud(state, hud, phase_color)
        if hud.toast:
            self._draw_toast(*hud.toast)
        if hud.paused:
            self._draw_paused()

        if show:
            pg.display.flip()
            if fps:
                self.clock.tick(fps)
        if as_array:
            return np.transpose(np.array(pg.surfarray.pixels3d(scr)), (1, 0, 2)).copy()
        return None

    def _draw_reference(self, px, py, theta) -> None:
        pg, gfx = self.pg, self.gfx
        L_px = self.model.params.pendulum_length * self.scale
        # swing circle (subtle) and dashed upright reference line
        gfx.aacircle(self.screen, int(px), int(py), int(L_px), lerp_color(BG_TOP, STEEL_DARK, 0.55))
        top = py - L_px - 34
        yy = py - 14
        while yy > top:
            pg.draw.line(self.screen, ACCENT, (px, yy), (px, max(yy - 7, top)), 1)
            yy -= 13
        self._blit_text(self.screen, "UPRIGHT  theta = 0", (px + 8, top - 2), "tick", ACCENT)
        # angle arc between the upright line and the pendulum
        r = 62
        theta_w = float(wrap_angle(theta))
        steps = max(2, int(abs(theta_w) / 0.05))
        pts = [(px + r * math.sin(theta_w * i / steps), py - r * math.cos(theta_w * i / steps)) for i in range(steps + 1)]
        if len(pts) > 1:
            pg.draw.lines(self.screen, ACCENT, False, pts, 2)
        mid = theta_w / 2
        lx, ly = px + (r + 12) * math.sin(mid), py - (r + 12) * math.cos(mid)
        right = math.sin(mid) >= 0
        self._blit_text(self.screen, f"{math.degrees(theta_w):+.1f} deg", (lx + (8 if right else -8), ly), "tick", TEXT,
                        anchor="midleft" if right else "midright")

    def _draw_trail(self) -> None:
        n = len(self._trail)
        if n < 2:
            return
        pg = self.pg
        pts = [p for p, _ in self._trail]
        for i in range(1, n):
            t = i / n
            col = lerp_color(BG_TOP, BOB_COLOR, 0.15 + 0.7 * t * t)
            pg.draw.line(self.screen, col, pts[i - 1], pts[i], 2)

    def _draw_pendulum(self, px, py, tip, theta, phase_color) -> None:
        pg, gfx = self.pg, self.gfx
        dx, dy = tip[0] - px, tip[1] - py
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length                      # unit normal
        # counter-weight stub behind the pivot, rod from pivot to tip
        back = (px - dx / length * 14, py - dy / length * 14)

        def quad(width: float, a, b):
            h = width / 2
            return [(int(a[0] + nx * h), int(a[1] + ny * h)), (int(b[0] + nx * h), int(b[1] + ny * h)),
                    (int(b[0] - nx * h), int(b[1] - ny * h)), (int(a[0] - nx * h), int(a[1] - ny * h))]

        shadow = [(x + 4, y + 5) for x, y in quad(8, back, tip)]
        sh = pg.Surface(self.size, pg.SRCALPHA)
        gfx.filled_polygon(sh, shadow, (0, 0, 0, 70))
        self.screen.blit(sh, (0, 0))
        edge = quad(9, back, tip)
        gfx.filled_polygon(self.screen, edge, lerp_color(ROD_COLOR, BG_TOP, 0.45))
        gfx.aapolygon(self.screen, edge, lerp_color(ROD_COLOR, BG_TOP, 0.45))
        body = quad(6, back, tip)
        gfx.filled_polygon(self.screen, body, ROD_COLOR)
        gfx.aapolygon(self.screen, body, ROD_COLOR)
        # bob
        bx, by = int(tip[0]), int(tip[1])
        gfx.filled_circle(self.screen, bx, by, 16, lerp_color(BOB_COLOR, (0, 0, 0), 0.45))
        gfx.filled_circle(self.screen, bx, by, 14, BOB_COLOR)
        gfx.filled_circle(self.screen, bx - 4, by - 5, 5, lerp_color(BOB_COLOR, (255, 255, 255), 0.55))
        gfx.aacircle(self.screen, bx, by, 16, lerp_color(BOB_COLOR, (255, 255, 255), 0.25))
        # status halo around the bob
        gfx.aacircle(self.screen, bx, by, 21, phase_color)

    def _draw_carriage(self, px, py, command) -> None:
        pg, gfx = self.pg, self.gfx
        half_w = int(0.095 * self.scale)
        plate = pg.Rect(0, 0, 2 * half_w, 2 * self.rod_dy + 10)
        plate.center = (int(px), int(py))
        sh = pg.Surface((plate.w + 20, plate.h + 20), pg.SRCALPHA)
        pg.draw.rect(sh, (0, 0, 0, 70), pg.Rect(10, 14, plate.w, plate.h), border_radius=7)
        self.screen.blit(sh, (plate.x - 10 + 2, plate.y - 10 + 3))
        pg.draw.rect(self.screen, CARRIAGE, plate, border_radius=7)
        pg.draw.rect(self.screen, CARRIAGE_EDGE, plate, width=2, border_radius=7)
        pg.draw.line(self.screen, lerp_color(CARRIAGE_EDGE, (255, 255, 255), 0.2), (plate.left + 8, plate.top + 3),
                     (plate.right - 8, plate.top + 3))
        # linear bearing blocks wrapped around each rod
        for sign in (-1, 1):
            blk = pg.Rect(0, 0, 2 * half_w + 8, 20)
            blk.center = (int(px), int(py + sign * self.rod_dy))
            pg.draw.rect(self.screen, BEARING, blk, border_radius=5)
            pg.draw.rect(self.screen, CARRIAGE_EDGE, blk, width=1, border_radius=5)
            for bx in (blk.left + 8, blk.right - 8):
                gfx.filled_circle(self.screen, bx, blk.centery, 2, STEEL)
        # drive command: arrow above the carriage, length proportional to command
        ay = int(py - self.rod_dy - 34)
        length = command * 80
        pg.draw.line(self.screen, TEXT_FAINT, (px - 84, ay), (px + 84, ay), 1)
        pg.draw.line(self.screen, TEXT_FAINT, (px, ay - 4), (px, ay + 4), 1)
        if abs(length) > 2:
            end = px + length
            pg.draw.line(self.screen, ACCENT, (px, ay), (end, ay), 3)
            s = 1 if length > 0 else -1
            gfx.filled_polygon(self.screen, [(int(end + s * 9), ay), (int(end), ay - 6), (int(end), ay + 6)], ACCENT)
        self._blit_text(self.screen, "DRIVE", (px - 90, ay), "tick", TEXT_FAINT, anchor="midright")

    def _draw_pivot(self, px, py) -> None:
        gfx = self.gfx
        gfx.filled_circle(self.screen, int(px), int(py), 13, (9, 14, 28))
        gfx.filled_circle(self.screen, int(px), int(py), 10, lerp_color(STEEL_DARK, STEEL_LIGHT, 0.35))
        gfx.aacircle(self.screen, int(px), int(py), 13, ACCENT)
        gfx.aacircle(self.screen, int(px), int(py), 12, ACCENT)
        gfx.filled_circle(self.screen, int(px), int(py), 4, (9, 14, 28))

    # ------------------------------------------------------------------ HUD
    def _card(self, rect, title: str | None = None):
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pg.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=10)
        if title:
            self._blit_text(self.screen, title, (rect.x + 16, rect.y + 12), "small", TEXT_FAINT)

    def _bar(self, rect, value: float, lo: float, hi: float, color, centered: bool = False, marker: float | None = None):
        pg = self.pg
        pg.draw.rect(self.screen, (10, 16, 32), rect, border_radius=4)
        frac = (value - lo) / (hi - lo)
        if centered:
            mid = rect.centerx
            end = rect.x + int(max(0.0, min(1.0, frac)) * rect.w)
            fill = pg.Rect(min(mid, end), rect.y, abs(end - mid), rect.h)
        else:
            fill = pg.Rect(rect.x, rect.y, int(max(0.0, min(1.0, frac)) * rect.w), rect.h)
        if fill.w > 0:
            pg.draw.rect(self.screen, color, fill, border_radius=4)
        if centered:
            pg.draw.line(self.screen, TEXT_DIM, (rect.centerx, rect.y - 2), (rect.centerx, rect.bottom + 1), 1)
        if marker is not None:
            mx = rect.x + int((marker - lo) / (hi - lo) * rect.w)
            pg.draw.line(self.screen, TEXT, (mx, rect.y - 3), (mx, rect.bottom + 2), 2)
        pg.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=4)

    def _draw_hud(self, state, hud: HudInfo, phase_color) -> None:
        pg = self.pg
        hx, w_total = self.hud_x, HUD_WIDTH - 24
        x0 = hx + 8
        theta = float(wrap_angle(float(state[TH])))
        theta_dot, x, xd = float(state[THD]), float(state[X]), float(state[XD])
        energy = self.model.energy(np.asarray(state, dtype=np.float64)).swing_normalized

        # --- card 1: mode + status
        r1 = pg.Rect(x0, 16, w_total, 126 if hud.notice else 108)
        self._card(r1, "CONTROL MODE")
        self._blit_text(self.screen, hud.mode_label or "-", (r1.x + 16, r1.y + 30), "big", TEXT)
        pill = pg.Rect(r1.x + 16, r1.y + 66, 176, 28)
        pg.draw.rect(self.screen, lerp_color(PANEL, phase_color, 0.18), pill, border_radius=14)
        pg.draw.rect(self.screen, lerp_color(PANEL, phase_color, 0.7), pill, width=1, border_radius=14)
        self.gfx.filled_circle(self.screen, pill.x + 16, pill.centery, 5, phase_color)
        self.gfx.aacircle(self.screen, pill.x + 16, pill.centery, 5, phase_color)
        self._blit_text(self.screen, PHASE_LABELS.get(hud.phase, hud.phase.upper()), (pill.x + 30, pill.centery),
                        "label", phase_color, anchor="midleft")
        self._blit_text(self.screen, f"t = {hud.time:6.2f} s", (r1.right - 16, r1.y + 80), "value", TEXT, anchor="midright")
        if hud.notice:
            self._blit_text(self.screen, hud.notice, (r1.x + 16, r1.bottom - 16), "small", TEXT_DIM, anchor="midleft")

        # --- card 2: target and error
        r2 = pg.Rect(x0, r1.bottom + 12, w_total, 74)
        self._card(r2, "TARGET")
        self._blit_text(self.screen, "upright  theta* = 0.0 deg", (r2.x + 16, r2.y + 34), "label", TEXT)
        err = abs(math.degrees(theta))
        self._blit_text(self.screen, f"error {err:6.1f} deg", (r2.right - 16, r2.y + 34), "value",
                        PHASE_COLORS["balanced"] if err < 6 else (PHASE_COLORS["stabilizing"] if err < 20 else TEXT), anchor="topright")
        self._bar(pg.Rect(r2.x + 16, r2.y + 56, w_total - 32, 6), 180 - err, 0, 180, phase_color)

        # --- card 3: telemetry
        r3 = pg.Rect(x0, r2.bottom + 12, w_total, 230)
        self._card(r3, "TELEMETRY")
        rows = [
            ("Pendulum angle", f"{math.degrees(theta):+8.2f} deg", f"{theta:+.3f} rad"),
            ("Angular velocity", f"{theta_dot:+8.2f} rad/s", f"{math.degrees(theta_dot):+.0f} deg/s"),
            ("Support position", f"{x:+8.3f} m", f"{x / self.model.support_limit * 100:+.0f}% of travel"),
            ("Support velocity", f"{xd:+8.3f} m/s", ""),
            ("Episode reward", f"{hud.reward:+9.1f}", ""),
        ]
        y = r3.y + 38
        for label, value, sub in rows:
            self._blit_text(self.screen, label, (r3.x + 16, y), "label", TEXT_DIM)
            self._blit_text(self.screen, value, (r3.right - 16, y - 1), "value", TEXT, anchor="topright")
            if sub:
                self._blit_text(self.screen, sub, (r3.right - 16, y + 17), "tick", TEXT_FAINT, anchor="topright")
            y += 38

        # --- card 4: command + energy
        r4 = pg.Rect(x0, r3.bottom + 12, w_total, 92)
        self._card(r4, "ACTUATOR / ENERGY")
        self._blit_text(self.screen, "Command u", (r4.x + 16, r4.y + 34), "label", TEXT_DIM)
        self._blit_text(self.screen, f"{hud.command:+.2f}", (r4.right - 16, r4.y + 33), "value", ACCENT, anchor="topright")
        self._bar(pg.Rect(r4.x + 110, r4.y + 38, w_total - 110 - 90, 8), hud.command, -1, 1, ACCENT, centered=True)
        self._blit_text(self.screen, "Swing energy", (r4.x + 16, r4.y + 62), "label", TEXT_DIM)
        self._blit_text(self.screen, f"{energy:5.2f}", (r4.right - 16, r4.y + 61), "value", TEXT, anchor="topright")
        self._bar(pg.Rect(r4.x + 110, r4.y + 66, w_total - 110 - 90, 8), energy, 0, 1.25, BOB_COLOR, marker=1.0)

        # --- card 5: controls
        if hud.show_help and hud.controls:
            r5 = pg.Rect(x0, r4.bottom + 12, w_total, self.size[1] - r4.bottom - 12 - 16)
            self._card(r5, "KEYBOARD")
            cols = 2
            per_col = (len(hud.controls) + cols - 1) // cols
            for i, (key, desc) in enumerate(hud.controls):
                cx = r5.x + 16 + (i // per_col) * (w_total // 2)
                cy = r5.y + 32 + (i % per_col) * 17
                self._blit_text(self.screen, key, (cx, cy), "tick", ACCENT)
                self._blit_text(self.screen, desc, (cx + 56, cy), "tick", TEXT_DIM)

    def _draw_toast(self, message: str, opacity: float) -> None:
        pg = self.pg
        surf = self._text(message, "label", TEXT)
        pad = 14
        rect = pg.Rect(0, 0, surf.get_width() + 2 * pad, surf.get_height() + 14)
        rect.midbottom = (self.scene_w // 2, self.size[1] - 66)
        overlay = pg.Surface(rect.size, pg.SRCALPHA)
        a = int(235 * max(0.0, min(1.0, opacity)))
        pg.draw.rect(overlay, (*PANEL, a), overlay.get_rect(), border_radius=9)
        pg.draw.rect(overlay, (*ACCENT, int(a * 0.8)), overlay.get_rect(), width=1, border_radius=9)
        self.screen.blit(overlay, rect)
        text = surf.copy()
        text.set_alpha(int(255 * max(0.0, min(1.0, opacity))))
        self.screen.blit(text, (rect.x + pad, rect.y + 7))

    def _draw_paused(self) -> None:
        pg = self.pg
        overlay = pg.Surface((self.scene_w, self.size[1]), pg.SRCALPHA)
        overlay.fill((5, 8, 18, 120))
        self.screen.blit(overlay, (0, 0))
        self._blit_text(self.screen, "PAUSED", (self.scene_w // 2, self.cy - 150), "huge", TEXT, anchor="center")
        self._blit_text(self.screen, "press P to resume", (self.scene_w // 2, self.cy - 118), "label", TEXT_DIM, anchor="center")

    # ------------------------------------------------------------------ misc
    def save(self, path: str) -> None:
        self.pg.image.save(self.screen, path)

    def close(self) -> None:
        if not self.headless:
            self.pg.display.quit()
        self._text_cache.clear()
