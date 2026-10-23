"""Voice Library: registered voice profiles, plus a collapsible, presentation-only character catalog.

Only registered profiles that the backend reports as selectable get "Dùng trong Studio", and that action only
*selects* the voice in Studio (through the app's _handle_profile_selected); loading is a separate, explicit
step there. Catalog entries without a valid link to a registered profile show "Chưa có preset giọng" and have
no action at all.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Mapping, Optional

from voice_changer.gui.dpi import px
from voice_changer.gui.presentation import (
    NO_PRESET,
    AvatarSpec,
    CharacterCatalog,
    ResolvedCatalogEntry,
    VoicePresentation,
    availability,
    character_avatar_image,
    language_label,
    load_character_catalog,
    load_voice_presentation,
    monogram,
    readiness_text,
    resolve_catalog,
    rights_label,
    voice_presentation_for,
)
from voice_changer.gui.theme import Colors, Fonts
from voice_changer.gui.views.widgets import CollapsibleSection, create_avatar, make_link
from voice_changer.service import ProfileSummary

logger = logging.getLogger(__name__)

FILTERS: list[tuple[str, str]] = [
    ("all", "Tất cả"),
    ("available", "Dùng được"),
    ("preview", "Thử nghiệm"),
    ("planned", "Dự kiến"),
    ("unavailable", "Không khả dụng"),
    ("vi", "Tiếng Việt"),
    ("en", "Tiếng Anh"),
]


class VoiceLibraryView(ttk.Frame):
    """Searchable, filterable voice cards with truthful availability and data-rights badges."""

    def __init__(
        self,
        parent: tk.Widget,
        on_profile_selected: Callable[[str], None],
        presentation: Optional[Mapping[str, VoicePresentation]] = None,
        catalog: Optional[CharacterCatalog] = None,
    ) -> None:
        super().__init__(parent, style="TFrame")
        self.on_profile_selected = on_profile_selected
        self._presentation = presentation if presentation is not None else load_voice_presentation()
        self._catalog = catalog if catalog is not None else load_character_catalog()

        self._all_profiles: list[ProfileSummary] = []
        self._filtered_profiles: list[ProfileSummary] = []
        self._catalog_entries: list[ResolvedCatalogEntry] = []
        self._filtered_catalog: list[ResolvedCatalogEntry] = []
        self.rejected_catalog_links: list[str] = []
        self._rendered_cards: dict[str, dict[str, Any]] = {}
        self._catalog_cards: dict[str, dict[str, Any]] = {}
        self._active_profile_id: Optional[str] = None
        self._is_loading: bool = False
        self._filter_category: str = "all"
        self._search_query: str = ""
        # (label, inset) of every card's wrapping labels, and the last canvas width they were sized for.
        self._wrap_targets: list[tuple[tk.Label, int]] = []
        self._canvas_width: int = 0
        self._columns: int = 2

        self._build_ui()

    # ----------------------------------------------------------------- layout
    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg=Colors.BG_DARK)
        toolbar.pack(fill="x", padx=20, pady=(14, 8))

        search_box = tk.Frame(toolbar, bg=Colors.BG_ELEVATED, highlightbackground=Colors.BORDER,
                              highlightthickness=1, padx=8, pady=5)
        search_box.pack(fill="x")
        tk.Label(search_box, text="🔍", font=Fonts.BODY, bg=Colors.BG_ELEVATED, fg=Colors.TEXT_MUTED).pack(
            side="left", padx=(0, 6))
        self.var_search = tk.StringVar()
        self.var_search.trace_add("write", self._on_search_changed)
        self.entry_search = tk.Entry(search_box, textvariable=self.var_search, font=Fonts.BODY,
                                     bg=Colors.BG_ELEVATED, fg=Colors.TEXT_PRIMARY,
                                     insertbackground=Colors.TEXT_PRIMARY, borderwidth=0)
        self.entry_search.pack(side="left", fill="x", expand=True)
        self.btn_clear = tk.Label(search_box, text="✕", font=Fonts.CAPTION_BOLD, bg=Colors.BG_ELEVATED,
                                  fg=Colors.TEXT_MUTED, padx=4)

        def _clear_search() -> None:
            self.var_search.set("")
            self.entry_search.focus_set()

        make_link(self.btn_clear, _clear_search, Colors.BG_ELEVATED)
        self._on_clear_search = lambda e=None: _clear_search()
        self.btn_clear.pack(side="right")

        filter_row = tk.Frame(toolbar, bg=Colors.BG_DARK)
        filter_row.pack(fill="x", pady=(10, 0))
        self._chip_buttons: dict[str, tk.Label] = {}
        for cat_id, cat_label in FILTERS:
            chip = tk.Label(filter_row, text=cat_label, font=Fonts.CAPTION_BOLD, padx=10, pady=4)
            make_link(chip, lambda c=cat_id: self._on_filter_chip_clicked(c), Colors.BG_DARK)
            chip.pack(side="left", padx=(0, 6))
            self._chip_buttons[cat_id] = chip
        self.lbl_count = tk.Label(filter_row, text="", font=Fonts.CAPTION, bg=Colors.BG_DARK, fg=Colors.TEXT_MUTED)
        self.lbl_count.pack(side="right")
        self._style_chips()

        cards_outer = tk.Frame(self, bg=Colors.BG_DARK)
        cards_outer.pack(fill="both", expand=True, padx=(12, 8), pady=(0, 10))
        self.canvas = tk.Canvas(cards_outer, bg=Colors.BG_DARK, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(cards_outer, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg=Colors.BG_DARK)
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        def _on_canvas_configure(event: tk.Event) -> None:
            self.canvas.itemconfig(self.canvas_window, width=event.width)
            self._canvas_width = event.width
            cols = 3 if event.width >= px(900) else 2
            if cols != self._columns:
                self._columns = cols
                self._grid_cards()
            self._apply_card_wraps()

        self.canvas.bind("<Configure>", _on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        def _on_mousewheel(event: tk.Event) -> None:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Registered voices
        self.lbl_profiles_title = tk.Label(self.content, text="Giọng đã đăng ký", font=Fonts.HEADING,
                                           bg=Colors.BG_DARK, fg=Colors.TEXT_PRIMARY, anchor="w")
        self.lbl_profiles_title.pack(fill="x", padx=8, pady=(4, 2))
        self.cards_frame = tk.Frame(self.content, bg=Colors.BG_DARK)
        self.cards_frame.pack(fill="x")
        self.empty_state = tk.Frame(self.content, bg=Colors.BG_DARK, pady=30)
        tk.Label(self.empty_state, text="Không có giọng nào khớp bộ lọc.", font=Fonts.SUBTITLE, bg=Colors.BG_DARK,
                 fg=Colors.TEXT_PRIMARY).pack(pady=(0, 8))
        btn_reset = tk.Label(self.empty_state, text="Xóa bộ lọc", font=Fonts.CAPTION_BOLD, bg=Colors.BG_DARK,
                             fg=Colors.ACCENT_CYAN, padx=6)
        make_link(btn_reset, self._reset_filters, Colors.BG_DARK)
        btn_reset.pack()

        # Planned character collection (presentation only)
        self.section_catalog = CollapsibleSection(self.content, self._catalog_title(), expanded=False,
                                                  bg=Colors.BG_SURFACE)
        self.lbl_catalog_note = tk.Label(self.section_catalog.body, text=self._catalog_note(), font=Fonts.CAPTION,
                                         bg=Colors.BG_SURFACE, fg=Colors.TEXT_MUTED, anchor="w", justify="left")
        self.lbl_catalog_note.pack(fill="x", padx=16, pady=(0, 8))
        self.catalog_frame = tk.Frame(self.section_catalog.body, bg=Colors.BG_SURFACE)
        self.catalog_frame.pack(fill="x", padx=8, pady=(0, 10))
        self._wrap_targets_static: list[tuple[tk.Label, int]] = [(self.lbl_catalog_note, -20)]

    def _catalog_title(self) -> str:
        franchise = self._catalog.franchise or "Nhân vật"
        return f"Bộ sưu tập {franchise} — dự kiến"

    def _catalog_note(self) -> str:
        parts = ["Danh sách nhân vật chỉ để tham khảo. Nhân vật chưa có preset giọng thì không thể sử dụng."]
        if self._catalog.source_url:
            parts.append(f"Danh sách lấy từ {self._catalog.source_url} (kiểm tra ngày {self._catalog.checked_on}).")
        if self._catalog.disclaimer:
            parts.append(self._catalog.disclaimer)
        return " ".join(parts)

    # ----------------------------------------------------------------- public API
    def set_profiles(self, profiles: list[ProfileSummary], active_id: Optional[str] = None) -> None:
        """Update the profiles (and catalog links, which are checked against them) and re-render."""
        self._all_profiles = list(profiles)
        self._active_profile_id = active_id
        self._catalog_entries, self.rejected_catalog_links = resolve_catalog(self._catalog, self._all_profiles)
        self._apply_filter_and_render()

    def set_loading(self, is_loading: bool) -> None:
        if self._is_loading == is_loading:
            return
        self._is_loading = is_loading
        for p_id in list(self._rendered_cards):
            self._update_card_state(p_id)
        for cid in list(self._catalog_cards):
            self._update_catalog_card(cid)

    def set_active_profile(self, active_id: Optional[str]) -> None:
        """Highlight the loaded profile in place (called on every status poll; must not rebuild cards)."""
        if self._active_profile_id == active_id:
            return
        old_id = self._active_profile_id
        self._active_profile_id = active_id
        for pid in (old_id, active_id):
            if pid and pid in self._rendered_cards:
                self._update_card_state(pid)
        for cid, data in self._catalog_cards.items():
            if data["resolved"].profile is not None and data["resolved"].profile.profile_id in (old_id, active_id):
                self._update_catalog_card(cid)

    # ----------------------------------------------------------------- filtering
    def _on_search_changed(self, *args: object) -> None:
        self._search_query = self.var_search.get().strip().lower()
        self._apply_filter_and_render()

    def _on_filter_chip_clicked(self, category: str) -> None:
        self._filter_category = category
        self._style_chips()
        self._apply_filter_and_render()

    def _style_chips(self) -> None:
        for cat_id, chip in self._chip_buttons.items():
            selected = cat_id == self._filter_category
            chip.config(bg=Colors.PRIMARY if selected else Colors.BG_ELEVATED,
                        fg=Colors.TEXT_ON_PRIMARY if selected else Colors.TEXT_MUTED)

    def _reset_filters(self) -> None:
        self.var_search.set("")
        self._on_filter_chip_clicked("all")

    def _profile_matches(self, p: ProfileSummary) -> bool:
        key = availability(p).key
        cat = self._filter_category
        if cat == "available" and not p.selectable:
            return False
        if cat == "preview" and not (p.selectable and p.status in ("preview", "prototype")):
            return False
        if cat == "planned" and key != "planned":
            return False
        if cat == "unavailable" and key not in ("withdrawn", "files_missing"):
            return False
        if cat in ("vi", "en") and p.language != cat:
            return False
        if self._search_query:
            pres = voice_presentation_for(p, self._presentation)
            haystack = " ".join((p.display_name, pres.tagline, p.profile_id, p.description,
                                 language_label(p.language))).lower()
            if self._search_query not in haystack:
                return False
        return True

    def _catalog_matches(self, r: ResolvedCatalogEntry) -> bool:
        if self._filter_category not in ("all", "planned"):
            return False
        if self._search_query:
            role = self._catalog.role(r.entry.role).label
            haystack = f"{r.entry.name} {role} {self._catalog.franchise}".lower()
            return self._search_query in haystack
        return True

    def _apply_filter_and_render(self) -> None:
        self._filtered_profiles = [p for p in self._all_profiles if self._profile_matches(p)]
        self._filtered_catalog = [r for r in self._catalog_entries if self._catalog_matches(r)]
        self.lbl_count.config(text=f"Hiển thị {len(self._filtered_profiles)} / {len(self._all_profiles)} giọng")

        self._rendered_cards.clear()
        self._catalog_cards.clear()
        self._wrap_targets = list(self._wrap_targets_static)
        for frame in (self.cards_frame, self.catalog_frame):
            for widget in frame.winfo_children():
                widget.destroy()

        for profile in self._filtered_profiles:
            self._render_profile_card(profile)
        for resolved in self._filtered_catalog:
            self._render_catalog_card(resolved)

        # Tk keeps a grid master at its last requested size once its final slave is gone, so an emptied
        # cards_frame would still reserve the height of the cards it used to hold: unpack it instead.
        if self._filtered_profiles:
            self.empty_state.pack_forget()
            self.cards_frame.pack(fill="x", after=self.lbl_profiles_title)
        else:
            self.cards_frame.pack_forget()
            self.empty_state.pack(fill="x", after=self.lbl_profiles_title)
        if self._filtered_catalog:
            linked = sum(1 for r in self._catalog_entries if r.profile is not None)
            self.section_catalog.set_summary(f"{len(self._catalog_entries)} nhân vật · {linked} có preset")
            self.section_catalog.pack(fill="x", padx=8, pady=(16, 8))
            if self._search_query:
                self.section_catalog.set_expanded(True)
        else:
            self.section_catalog.pack_forget()
        self._grid_cards()
        self._apply_card_wraps()

    def _grid_cards(self) -> None:
        cols = self._columns
        for frame, cards in ((self.cards_frame, self._rendered_cards), (self.catalog_frame, self._catalog_cards)):
            for c in range(3):
                frame.columnconfigure(c, weight=1 if c < cols else 0, uniform="lib_col" if c < cols else "")
            for idx, data in enumerate(cards.values()):
                data["card"].grid(row=idx // cols, column=idx % cols, padx=6, pady=6, sticky="nsew")

    def _apply_card_wraps(self) -> None:
        """Wrap card text to the column width derived from the canvas, which the window size (not the cards) sets."""
        if self._canvas_width <= 1:
            return
        # Each card has grid padx 6 per side, card padx 14 per side and a 1-2 px border.
        text_width = max(140, self._canvas_width // self._columns - 52)
        for lab, inset in self._wrap_targets:
            try:
                lab.config(wraplength=max(120, (self._canvas_width - 60) if inset < 0 else text_width - inset))
            except tk.TclError:
                pass  # card destroyed by a re-render

    # ----------------------------------------------------------------- profile cards
    DATA_RIGHTS_LABELS: dict[str, str] = {
        "unverified": "Bản quyền: chưa xác minh (chỉ dùng cá nhân)",
        "owner_recorded": "Bản quyền: giọng do chủ dự án ghi âm",
        "consented": "Bản quyền: có sự đồng ý của người nói",
        "licensed": "Bản quyền: có giấy phép",
    }

    @classmethod
    def _verified_spec_text(cls, profile: Any) -> str:
        """Card details for a loadable profile, taken only from fields ProfileSummary actually carries.

        The checkpoint's RVC version and sample rate are known only after validate_profile() reads (and hashes)
        the files, which is too expensive per card, so they are not shown rather than assumed.
        """
        parts = []
        defaults = getattr(profile, "defaults", None)
        if defaults is not None:
            parts.append(f"F0: {defaults.f0_method.upper()}")
            parts.append(f"Pitch mặc định: {defaults.pitch_shift:+.0f}")
        rights = cls.DATA_RIGHTS_LABELS.get(getattr(profile, "data_rights", ""))
        if rights:
            parts.append(rights)
        return " • ".join(parts) if parts else "Thông số mô hình: không có dữ liệu"

    @classmethod
    def _spec_text(cls, profile: ProfileSummary) -> str:
        key = availability(profile).key
        if key == "planned":
            return "Mô hình: Chưa cài đặt • Chưa có checkpoint hoặc chỉ mục"
        if key == "files_missing":
            return "Mô hình: Chưa tải về máy • Thông số kỹ thuật chưa xác định"
        if key == "withdrawn":
            return f"Trạng thái: Đã thu hồi • {profile.withdrawn_reason}"
        return cls._verified_spec_text(profile)

    def _render_profile_card(self, profile: ProfileSummary) -> None:
        bg = Colors.BG_CARD
        pres = voice_presentation_for(profile, self._presentation)
        if pres.avatar.image is None:
            # A voice linked from a catalog character wears that character's personal picture, if there is one.
            linked = next((r.entry.catalog_id for r in self._catalog_entries
                           if r.profile is not None and r.profile.profile_id == profile.profile_id), None)
            image = character_avatar_image(linked) if linked else None
            if image is not None:
                pres = VoicePresentation(pres.tagline, AvatarSpec(pres.avatar.glyph, pres.avatar.color, image))
        card = tk.Frame(self.cards_frame, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1,
                        padx=14, pady=14)

        top_row = tk.Frame(card, bg=bg)
        top_row.pack(fill="x", pady=(0, 8))
        avatar_widget = create_avatar(top_row, pres.avatar, 48, bg)
        avatar_widget.pack(side="left", padx=(0, 10))
        name_col = tk.Frame(top_row, bg=bg)
        name_col.pack(side="left", fill="x", expand=True)
        lbl_name = tk.Label(name_col, text=profile.display_name, font=Fonts.SUBTITLE, bg=bg, fg=Colors.TEXT_PRIMARY,
                            anchor="w", justify="left")
        lbl_name.pack(fill="x")
        lbl_lang = tk.Label(name_col, text=language_label(profile.language), font=Fonts.CAPTION, bg=bg,
                            fg=Colors.TEXT_MUTED, anchor="w")
        lbl_lang.pack(fill="x")

        badges = tk.Frame(card, bg=bg)
        badges.pack(fill="x", pady=(0, 8))
        pill_status = tk.Label(badges, text="", font=Fonts.BADGE, padx=8, pady=3)
        pill_status.pack(side="left")
        rights = rights_label(profile)
        if rights:
            tk.Label(badges, text=rights, font=Fonts.BADGE, padx=8, pady=3, bg=Colors.WARNING_BG,
                     fg=Colors.WARNING_TEXT).pack(side="left", padx=(6, 0))

        lbl_cat = tk.Label(card, text=pres.tagline, font=Fonts.BODY, bg=bg, fg=Colors.TEXT_SECONDARY, anchor="w",
                           justify="left")
        lbl_cat.pack(fill="x", pady=(0, 6))
        lbl_desc = tk.Label(card, text=readiness_text(profile), font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED,
                            anchor="w", justify="left")
        lbl_desc.pack(fill="x", pady=(0, 8))

        # Technical details (registry text is English and written for engineers): hidden until asked for.
        details = tk.Frame(card, bg=Colors.BG_ELEVATED, padx=10, pady=8)
        lbl_spec = tk.Label(details, text=self._spec_text(profile), font=Fonts.CAPTION, bg=Colors.BG_ELEVATED,
                            fg=Colors.TEXT_MUTED, anchor="w", justify="left")
        lbl_spec.pack(fill="x")
        tech_lines = [f"ID: {profile.profile_id} · trạng thái registry: {profile.status}"]
        if profile.description:
            tech_lines.append(profile.description)
        if profile.setup_hint:
            tech_lines.append(f"Hướng dẫn: {profile.setup_hint}")
        lbl_tech = tk.Label(details, text="\n".join(tech_lines), font=Fonts.CAPTION, bg=Colors.BG_ELEVATED,
                            fg=Colors.TEXT_MUTED, anchor="w", justify="left")
        lbl_tech.pack(fill="x", pady=(4, 0))
        btn_details = tk.Label(card, text="Chi tiết kỹ thuật ▸", font=Fonts.CAPTION_BOLD, bg=bg,
                               fg=Colors.ACCENT_CYAN, anchor="w", padx=2)

        def _toggle_details() -> None:
            if details.winfo_ismapped():
                details.pack_forget()
                btn_details.config(text="Chi tiết kỹ thuật ▸")
            else:
                details.pack(fill="x", pady=(0, 8), after=btn_details)
                btn_details.config(text="Chi tiết kỹ thuật ▾")

        make_link(btn_details, _toggle_details, bg)
        btn_details.pack(anchor="w", pady=(0, 8))

        btn_action_box = tk.Frame(card, bg=bg)
        btn_action_box.pack(fill="x", side="bottom")

        self._wrap_targets.extend([(lbl_name, 70), (lbl_cat, 0), (lbl_desc, 0), (lbl_spec, 24), (lbl_tech, 24)])
        self._rendered_cards[profile.profile_id] = {
            "card": card, "top_row": top_row, "name_col": name_col, "lbl_name": lbl_name, "lbl_lang": lbl_lang,
            "badges": badges, "lbl_cat": lbl_cat, "lbl_desc": lbl_desc, "pill_status": pill_status,
            "btn_details": btn_details, "btn_action_box": btn_action_box, "avatar_widget": avatar_widget,
            "profile": profile,
        }
        self._update_card_state(profile.profile_id)

    def _update_card_state(self, p_id: str) -> None:
        """Update one card's active look, status pill and action in place."""
        data = self._rendered_cards.get(p_id)
        if data is None:
            return
        profile = data["profile"]
        av = availability(profile, self._active_profile_id)
        is_active = av.key == "active"
        bg = Colors.BG_CARD_ACTIVE if is_active else Colors.BG_CARD
        data["card"].config(bg=bg, highlightbackground=Colors.PRIMARY if is_active else Colors.BORDER,
                            highlightthickness=2 if is_active else 1)
        for key in ("top_row", "name_col", "lbl_name", "lbl_lang", "badges", "lbl_cat", "lbl_desc", "btn_details",
                    "btn_action_box", "avatar_widget"):
            try:
                data[key].config(bg=bg)
            except tk.TclError:
                pass
        data["btn_details"].config(highlightbackground=bg)
        data["lbl_desc"].config(text=readiness_text(profile, self._active_profile_id))
        data["pill_status"].config(text=av.label, bg=av.bg, fg=av.fg)
        self._render_action(data["btn_action_box"], profile, av, bg)

    def _render_action(self, box: tk.Frame, profile: ProfileSummary, av: Any, bg: str) -> None:
        for w in box.winfo_children():
            w.destroy()
        if av.key == "active":
            tk.Label(box, text="✓ Đang dùng trong Studio", font=Fonts.BUTTON, bg=Colors.SUCCESS_BG,
                     fg=Colors.SUCCESS_TEXT, pady=7).pack(fill="x")
        elif not av.loadable:
            reason = {"withdrawn": "Đã thu hồi", "planned": "Chưa có mô hình", "no_preset": "Chưa có preset giọng"}
            tk.Label(box, text=f"{reason.get(av.key, 'Thiếu tệp')} · Không thể nạp", font=Fonts.BUTTON,
                     bg=Colors.BG_ELEVATED, fg=Colors.TEXT_MUTED, pady=7).pack(fill="x")
        elif self._is_loading:
            tk.Label(box, text="⏳ Đang nạp giọng…", font=Fonts.BUTTON, bg=Colors.BG_ELEVATED,
                     fg=Colors.TEXT_MUTED, pady=7).pack(fill="x")
        else:
            tk.Button(box, text="Dùng trong Studio", font=Fonts.BUTTON, bg=Colors.PRIMARY,
                      fg=Colors.TEXT_ON_PRIMARY, activebackground=Colors.PRIMARY_HOVER,
                      activeforeground=Colors.TEXT_ON_PRIMARY, borderwidth=0, cursor="hand2", pady=7,
                      command=lambda pid=profile.profile_id: self._on_card_load_clicked(pid)).pack(fill="x")

    # ----------------------------------------------------------------- catalog cards
    def _render_catalog_card(self, resolved: ResolvedCatalogEntry) -> None:
        bg = Colors.BG_CARD
        entry = resolved.entry
        role = self._catalog.role(entry.role)
        card = tk.Frame(self.catalog_frame, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1,
                        padx=12, pady=10)
        row = tk.Frame(card, bg=bg)
        row.pack(fill="x")
        avatar_widget = create_avatar(
            row, AvatarSpec(monogram(entry.name), role.color, character_avatar_image(entry.catalog_id)), 36, bg)
        avatar_widget.pack(side="left", padx=(0, 10))
        col = tk.Frame(row, bg=bg)
        col.pack(side="left", fill="x", expand=True)
        tk.Label(col, text=entry.name, font=Fonts.BODY_BOLD, bg=bg, fg=Colors.TEXT_PRIMARY, anchor="w").pack(fill="x")
        tk.Label(col, text=role.label, font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED, anchor="w").pack(fill="x")
        pill_status = tk.Label(card, text="", font=Fonts.BADGE, padx=8, pady=3)
        pill_status.pack(anchor="w", pady=(8, 6))
        btn_action_box = tk.Frame(card, bg=bg)
        btn_action_box.pack(fill="x")
        self._catalog_cards[entry.catalog_id] = {
            "card": card, "pill_status": pill_status, "btn_action_box": btn_action_box,
            "avatar_widget": avatar_widget, "resolved": resolved,
        }
        self._update_catalog_card(entry.catalog_id)

    def _update_catalog_card(self, cid: str) -> None:
        data = self._catalog_cards.get(cid)
        if data is None:
            return
        profile = data["resolved"].profile
        av = availability(profile, self._active_profile_id) if profile is not None else NO_PRESET
        data["pill_status"].config(text=av.label, bg=av.bg, fg=av.fg)
        if profile is None:
            for w in data["btn_action_box"].winfo_children():
                w.destroy()  # no preset: no action of any kind
        else:
            self._render_action(data["btn_action_box"], profile, av, Colors.BG_CARD)

    # ----------------------------------------------------------------- actions
    def _on_card_load_clicked(self, profile_id: str) -> None:
        """Hand a registered, selectable profile to the app's selection flow; anything else is refused."""
        profile = next((p for p in self._all_profiles if p.profile_id == profile_id), None)
        if profile is None or not getattr(profile, "selectable", False):
            logger.warning("Prevented use of unknown or unready profile '%s'", profile_id)
            return
        self.on_profile_selected(profile_id)
