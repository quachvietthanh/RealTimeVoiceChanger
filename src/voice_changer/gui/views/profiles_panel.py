"""Voice Studio hero: the selected voice (avatar, name, availability) and the voice picker.

Choosing a voice here only selects it. Loading happens through Studio's primary action ("Nạp giọng"), and
whether a voice can be loaded comes only from ProfileSummary.selectable; what is loaded comes only from the
live engine status (set_active_profile).
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Mapping, Optional

from voice_changer.gui.presentation import (
    VoicePresentation,
    availability,
    language_label,
    load_voice_presentation,
    readiness_text,
    rights_label,
    voice_presentation_for,
)
from voice_changer.gui.theme import Colors, Fonts
from voice_changer.gui.views.widgets import create_avatar, make_link
from voice_changer.service import ProfileSummary

logger = logging.getLogger(__name__)

HERO_AVATAR_SIZE = 72


class ProfilesPanel(ttk.Frame):
    """Selected-voice hero with a picker limited to voices the backend says can be loaded."""

    def __init__(
        self,
        parent: tk.Widget,
        service: Any = None,
        on_profile_selected: Optional[Callable[[str], None]] = None,
        on_open_library: Optional[Callable[[], None]] = None,
        presentation: Optional[Mapping[str, VoicePresentation]] = None,
    ) -> None:
        super().__init__(parent, style="TFrame")
        self.service = service  # kept for callers that pass it; the panel never calls the backend itself
        self.on_profile_selected = on_profile_selected
        self.on_open_library = on_open_library
        self._presentation = presentation if presentation is not None else load_voice_presentation()

        self._profiles: list[ProfileSummary] = []
        self._combo_ids: list[str] = []
        self._selected_id: Optional[str] = None
        self._active_profile_id: Optional[str] = None
        self._is_loading: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        bg = Colors.BG_DARK
        hero = tk.Frame(self, bg=bg)
        hero.pack(fill="x")

        self.avatar_box = tk.Frame(hero, bg=bg)
        self.avatar_box.pack(side="left", anchor="n", padx=(0, 18))
        self.avatar_canvas: tk.Canvas = create_avatar(
            self.avatar_box, voice_presentation_for(_placeholder_profile(), {}).avatar, HERO_AVATAR_SIZE, bg)
        self.avatar_canvas.pack()

        text_col = tk.Frame(hero, bg=bg)
        text_col.pack(side="left", fill="x", expand=True)

        self.lbl_hero_name = tk.Label(text_col, text="Chưa chọn giọng", font=Fonts.HERO, bg=bg,
                                      fg=Colors.TEXT_PRIMARY, anchor="w", justify="left")
        self.lbl_hero_name.pack(fill="x")

        self.lbl_hero_category = tk.Label(text_col, text="", font=Fonts.BODY, bg=bg, fg=Colors.TEXT_SECONDARY,
                                          anchor="w", justify="left")
        self.lbl_hero_category.pack(fill="x", pady=(2, 6))

        tags = tk.Frame(text_col, bg=bg)
        tags.pack(fill="x")
        self.pill_status = tk.Label(tags, text="", font=Fonts.BADGE, padx=8, pady=3, bg=Colors.BG_ELEVATED,
                                    fg=Colors.TEXT_MUTED)
        self.pill_status.pack(side="left")
        self.lbl_language = tk.Label(tags, text="", font=Fonts.BADGE, padx=8, pady=3, bg=Colors.BG_ELEVATED,
                                     fg=Colors.TEXT_SECONDARY)
        self.lbl_language.pack(side="left", padx=(6, 0))
        self.pill_rights = tk.Label(tags, text="", font=Fonts.BADGE, padx=8, pady=3, bg=Colors.WARNING_BG,
                                    fg=Colors.WARNING_TEXT)

        self.lbl_readiness_msg = tk.Label(text_col, text="", font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED,
                                          anchor="w", justify="left")
        self.lbl_readiness_msg.pack(fill="x", pady=(6, 0))

        picker = tk.Frame(self, bg=bg)
        picker.pack(fill="x", pady=(14, 0))
        self.btn_library = tk.Label(picker, text="Thư viện giọng →", font=Fonts.CAPTION_BOLD, bg=bg,
                                    fg=Colors.ACCENT_CYAN, padx=4)
        make_link(self.btn_library, self._open_library, bg)
        self.btn_library.pack(side="right")
        tk.Label(picker, text="Giọng:", font=Fonts.BODY_BOLD, bg=bg, fg=Colors.TEXT_SECONDARY).pack(side="left")
        self.profile_combo = ttk.Combobox(picker, state="readonly", style="TCombobox")
        self.profile_combo.pack(side="left", fill="x", expand=True, padx=(8, 12), ipady=2)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_combo_changed)

        def _wrap(e: tk.Event) -> None:
            wrap = max(160, e.width - 4)
            for lab in (self.lbl_hero_name, self.lbl_hero_category, self.lbl_readiness_msg):
                lab.config(wraplength=wrap)

        text_col.bind("<Configure>", _wrap)

    # ----------------------------------------------------------------- public API
    def set_profiles(self, profiles: list[ProfileSummary], current_id: Optional[str] = None) -> None:
        """Show `profiles`; select `current_id`, else keep the current choice, else the first loadable voice."""
        self._profiles = list(profiles)
        ids = {p.profile_id for p in self._profiles}
        target = current_id if current_id in ids else self._selected_id if self._selected_id in ids else None
        if target is None:
            target = next((p.profile_id for p in self._profiles if p.selectable), None)
        self._selected_id = target
        self._refresh_combo()
        self._update_display()

    def set_active_profile(self, active_id: Optional[str]) -> None:
        """Record what the backend reports as loaded. A change of loaded voice also selects it."""
        if self._active_profile_id == active_id:
            return
        self._active_profile_id = active_id
        if active_id and any(p.profile_id == active_id for p in self._profiles):
            self._selected_id = active_id
            self._refresh_combo()
        self._update_display()

    def select_profile(self, profile_id: str) -> None:
        """Show `profile_id` as the selected voice (not loaded). Does not call on_profile_selected."""
        if any(p.profile_id == profile_id for p in self._profiles):
            self._selected_id = profile_id
            self._refresh_combo()
            self._update_display()

    def get_selected_profile(self) -> Optional[ProfileSummary]:
        return next((p for p in self._profiles if p.profile_id == self._selected_id), None)

    def set_loading(self, is_loading: bool) -> None:
        self._is_loading = is_loading
        self.profile_combo.config(state="disabled" if is_loading else "readonly")

    # ----------------------------------------------------------------- internals
    def _refresh_combo(self) -> None:
        # The picker offers only voices the backend says can be loaded, plus the current choice if it cannot
        # (so the hero and the picker never disagree). Everything else is browsed in the Library.
        shown = [p for p in self._profiles if p.selectable or p.profile_id == self._selected_id]
        self._combo_ids = [p.profile_id for p in shown]
        self.profile_combo["values"] = [f"{p.display_name} — {language_label(p.language)}" for p in shown]
        if self._selected_id in self._combo_ids:
            self.profile_combo.current(self._combo_ids.index(self._selected_id))
        else:
            self.profile_combo.set("")

    def _on_combo_changed(self, event: Optional[tk.Event] = None) -> None:
        idx = self.profile_combo.current()
        if not 0 <= idx < len(self._combo_ids):
            return
        pid = self._combo_ids[idx]
        if self.on_profile_selected is not None:
            self.on_profile_selected(pid)  # the app validates, then calls select_profile()
        else:
            self.select_profile(pid)

    def _open_library(self) -> None:
        if self.on_open_library is not None:
            self.on_open_library()

    def _update_display(self) -> None:
        profile = self.get_selected_profile()
        bg = Colors.BG_DARK
        pres = voice_presentation_for(profile or _placeholder_profile(), self._presentation)
        self.avatar_canvas.destroy()
        self.avatar_canvas = create_avatar(self.avatar_box, pres.avatar, HERO_AVATAR_SIZE, bg)
        self.avatar_canvas.pack()

        if profile is None:
            self.lbl_hero_name.config(text="Chưa chọn giọng")
            self.lbl_hero_category.config(text="Chọn một giọng trong danh sách bên dưới hoặc trong Thư viện.")
            self.pill_status.pack_forget()
            self.lbl_language.pack_forget()
            self.pill_rights.pack_forget()
            self.lbl_readiness_msg.config(text="", fg=Colors.TEXT_MUTED)
            return

        av = availability(profile, self._active_profile_id)
        self.lbl_hero_name.config(text=profile.display_name)
        self.lbl_hero_category.config(text=pres.tagline)
        self.pill_status.config(text=av.label, bg=av.bg, fg=av.fg)
        self.pill_status.pack(side="left")
        self.lbl_language.config(text=language_label(profile.language))
        self.lbl_language.pack(side="left", padx=(6, 0))
        rights = rights_label(profile)
        if rights:
            self.pill_rights.config(text=rights)
            self.pill_rights.pack(side="left", padx=(6, 0))
        else:
            self.pill_rights.pack_forget()

        text = readiness_text(profile, self._active_profile_id)
        if av.key == "withdrawn" and profile.withdrawn_reason:
            text += f"\nLý do (ghi trong registry): {profile.withdrawn_reason}"
        elif not av.loadable and profile.setup_hint:
            text += f"\nHướng dẫn: {profile.setup_hint}"
        fg = Colors.TEXT_MUTED if av.loadable else (Colors.DANGER_TEXT if av.key != "planned" else Colors.WARNING_TEXT)
        self.lbl_readiness_msg.config(text=text, fg=fg)


def _placeholder_profile() -> Any:
    """A stand-in for "no voice selected", so the avatar code has one path."""
    from types import SimpleNamespace

    return SimpleNamespace(profile_id="", display_name="?", language="")
