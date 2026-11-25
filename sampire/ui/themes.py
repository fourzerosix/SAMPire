"""Colour themes. Each entry: (fg, accent, hot, ok, dim) as 256-colour indexes (-1 = terminal default)."""

THEMES = {
    "default": (-1, 45, 205, 83, 244),
    "gruvbox": (223, 214, 167, 142, 245),
    "nord": (253, 110, 174, 108, 243),
    "tokyonight": (189, 111, 204, 114, 242),
    "mono": (-1, -1, -1, -1, 244),
}
NAMES = list(THEMES)
