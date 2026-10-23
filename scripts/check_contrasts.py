from voice_changer.gui.theme import Colors

def lum(hex_str: str) -> float:
    hex_str = hex_str.lstrip('#')
    rgb = [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    rgb = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def cr(c1: str, c2: str) -> float:
    l1, l2 = lum(c1), lum(c2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

print("=" * 60)
print("WCAG AA CONTRAST RATIO AUDIT (BEFORE vs AFTER)")
print("=" * 60)

pairs = [
    ("White text on SUCCESS (#047857)", Colors.TEXT_ON_PRIMARY, Colors.SUCCESS, "Was 2.54:1 (on #10b981)"),
    ("White text on SUCCESS_HOVER (#065f46)", Colors.TEXT_ON_PRIMARY, Colors.SUCCESS_HOVER, "Hover state"),
    ("SUCCESS_TEXT on SUCCESS_BG", Colors.SUCCESS_TEXT, Colors.SUCCESS_BG, "Success status badge"),
    ("White text on DANGER (#be123c)", Colors.TEXT_ON_PRIMARY, Colors.DANGER, "Was 3.67:1 (on #f43f5e)"),
    ("White text on DANGER_HOVER (#9f1239)", Colors.TEXT_ON_PRIMARY, Colors.DANGER_HOVER, "Hover state"),
    ("DANGER_TEXT on DANGER_BG", Colors.DANGER_TEXT, Colors.DANGER_BG, "Danger error banner"),
    ("DANGER_TEXT on BG_ELEVATED", Colors.DANGER_TEXT, Colors.BG_ELEVATED, "Was 4.28:1 (raw DANGER on BG_ELEVATED)"),
    ("DANGER_TEXT on BG_SURFACE", Colors.DANGER_TEXT, Colors.BG_SURFACE, "Error text on surface"),
    ("TEXT_MUTED on BG_DARK", Colors.TEXT_MUTED, Colors.BG_DARK, "Was 3.94:1 with TEXT_DISABLED"),
    ("TEXT_MUTED on BG_SIDEBAR", Colors.TEXT_MUTED, Colors.BG_SIDEBAR, "Was 3.80:1 with TEXT_DISABLED"),
    ("TEXT_MUTED on BG_SURFACE", Colors.TEXT_MUTED, Colors.BG_SURFACE, "Was 3.60:1 with TEXT_DISABLED"),
    ("TEXT_MUTED on BG_ELEVATED", Colors.TEXT_MUTED, Colors.BG_ELEVATED, "Was 3.31:1 with TEXT_DISABLED"),
    ("TEXT_MUTED on BG_CARD", Colors.TEXT_MUTED, Colors.BG_CARD, "Was 3.46:1 with TEXT_DISABLED"),
    ("TEXT_SECONDARY on BG_CARD", Colors.TEXT_SECONDARY, Colors.BG_CARD, "Description text"),
    ("TEXT_PRIMARY on BG_CARD", Colors.TEXT_PRIMARY, Colors.BG_CARD, "Title text"),
    ("White on PRIMARY (#7c3aed)", Colors.TEXT_ON_PRIMARY, Colors.PRIMARY, "Primary button"),
]

for label, fg, bg, note in pairs:
    ratio = cr(fg, bg)
    status = "PASS (>= 4.5:1)" if ratio >= 4.5 else ("PASS (>= 3:1)" if ratio >= 3.0 else "FAIL (< 3:1)")
    print(f"{label:42} | {ratio:5.2f}:1 | {status:15} | {note}")

print("-" * 60)
print("AVATAR BADGES CONTRAST (BEFORE: 2.15-4.23:1)")
print("-" * 60)

from voice_changer.gui.views.library_view import VoiceLibraryView
for pid, (glyph, bg_c, fg_c, cat) in VoiceLibraryView.AVATAR_PRESETS.items():
    ratio = cr(fg_c, bg_c)
    status = "PASS (>= 4.5:1)" if ratio >= 4.5 else "FAIL"
    print(f"Avatar {pid:18} ({glyph}) | {fg_c} on {bg_c:7} | {ratio:5.2f}:1 | {status}")

print("-" * 60)
print("TTK FOCUS RING CONTRAST")
print("-" * 60)
focus_pairs = [
    ("Focus on Primary.TButton", Colors.TEXT_ON_PRIMARY, Colors.PRIMARY),
    ("Focus on Success.TButton", Colors.TEXT_ON_PRIMARY, Colors.SUCCESS),
    ("Focus on Danger.TButton", Colors.TEXT_ON_PRIMARY, Colors.DANGER),
    ("Focus on Secondary.TButton", Colors.ACCENT_CYAN, Colors.BG_ELEVATED),
    ("Focus on TCheckbutton", Colors.ACCENT_CYAN, Colors.BG_SURFACE),
]
for label, fg, bg in focus_pairs:
    ratio = cr(fg, bg)
    status = "PASS (>= 3:1)" if ratio >= 3.0 else "FAIL"
    print(f"{label:30} | {ratio:5.2f}:1 | {status}")
