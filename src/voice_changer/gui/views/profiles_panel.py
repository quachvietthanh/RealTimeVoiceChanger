"""Voice profile selection and metadata panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from voice_changer.gui.theme import Colors, Fonts
from voice_changer.service import ProfileSummary, VoiceChangerService


class ProfilesPanel(ttk.Frame):
    """Clean, balanced panel for selecting voice models without redundant labeling."""

    def __init__(
        self,
        parent: tk.Widget,
        service: VoiceChangerService,
        on_profile_selected: Callable[[str], None],
    ) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self.service = service
        self.on_profile_selected = on_profile_selected

        self._profiles: list[ProfileSummary] = []
        self._selected_id: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Card container with subtle border
        card = tk.Frame(
            self,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=16,
            pady=14,
        )
        card.pack(fill="x", expand=True)

        # Header Row: Title & Single Status Pill
        header_row = tk.Frame(card, bg=Colors.BG_SURFACE)
        header_row.pack(fill="x", pady=(0, 10))

        lbl_title = tk.Label(
            header_row,
            text="Hồ sơ giọng nói (Voice Profile)",
            font=Fonts.SUBTITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_title.pack(side="left")

        self.pill_status = tk.Label(
            header_row,
            text="",
            font=Fonts.BADGE,
            padx=10,
            pady=3,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
            relief="flat",
        )
        self.pill_status.pack(side="right")

        # Combobox selector
        self.profile_combo = ttk.Combobox(card, state="readonly", style="TCombobox")
        self.profile_combo.pack(fill="x", ipady=2, pady=(0, 10))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_combo_changed)

        # Details box
        self.info_box = tk.Frame(
            card,
            bg=Colors.BG_ELEVATED,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        self.info_box.pack(fill="x", pady=(0, 12))

        # Profile description with responsive wrapping
        self.lbl_desc = tk.Label(
            self.info_box,
            text="",
            font=Fonts.BODY,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_SECONDARY,
            anchor="w",
            justify="left",
        )
        self.lbl_desc.pack(fill="x", pady=(0, 6))

        # Bind configure for dynamic text wrapping
        self.info_box.bind(
            "<Configure>",
            lambda e: self.lbl_desc.config(wraplength=max(200, e.width - 24)),
        )

        # Readiness row (Model + Index status)
        readiness_row = tk.Frame(self.info_box, bg=Colors.BG_ELEVATED)
        readiness_row.pack(fill="x", pady=(2, 0))

        self.lbl_model_readiness = tk.Label(
            readiness_row,
            text="Mô hình (.pth): --",
            font=Fonts.CAPTION,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
        )
        self.lbl_model_readiness.pack(side="left")

        self.lbl_index_readiness = tk.Label(
            readiness_row,
            text="Chỉ mục (.index): --",
            font=Fonts.CAPTION,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
        )
        self.lbl_index_readiness.pack(side="right")

        # Load / Activate button
        self.btn_load = ttk.Button(
            card,
            text="Nạp hồ sơ vào GPU",
            style="Primary.TButton",
            command=self._on_apply_clicked,
        )
        self.btn_load.pack(fill="x")

    def set_profiles(self, profiles: list[ProfileSummary], current_id: Optional[str] = None) -> None:
        """Populate the profile list and select the active/default one."""
        self._profiles = profiles
        display_texts = []
        select_idx = 0

        target_id = current_id or (profiles[0].profile_id if profiles else None)

        for i, p in enumerate(profiles):
            lang_label = "Tiếng Việt" if p.language == "vi" else "Tiếng Anh"
            txt = f"{p.display_name} — {lang_label}"
            display_texts.append(txt)
            if p.profile_id == target_id:
                select_idx = i

        self.profile_combo["values"] = display_texts
        if display_texts:
            self.profile_combo.current(select_idx)
            self._update_display(profiles[select_idx])

    def _on_combo_changed(self, event: tk.Event) -> None:
        idx = self.profile_combo.current()
        if 0 <= idx < len(self._profiles):
            self._update_display(self._profiles[idx])

    def _update_display(self, profile: ProfileSummary) -> None:
        self._selected_id = profile.profile_id
        self.lbl_desc.config(text=profile.description or "Không có mô tả chi tiết.")

        # Single authoritative status pill (no duplicate labels elsewhere!)
        if profile.status == "approved":
            self.pill_status.config(
                text="✓ ĐÃ PHÊ DUYỆT",
                bg=Colors.SUCCESS_BG,
                fg=Colors.SUCCESS_TEXT,
            )
        elif profile.status == "preview":
            self.pill_status.config(
                text="⚠️ THỬ NGHIỆM (CHỜ DUYỆT)",
                bg=Colors.WARNING_BG,
                fg=Colors.WARNING_TEXT,
            )
        else:
            self.pill_status.config(
                text="ℹ️ BẢN MẪU KỸ THUẬT",
                bg=Colors.BG_ELEVATED,
                fg=Colors.TEXT_MUTED,
            )

        # Asset readiness
        if profile.files_present:
            self.lbl_model_readiness.config(text="✓ Mô hình (.pth): Đã sẵn sàng", fg=Colors.SUCCESS)
            self.lbl_index_readiness.config(text="✓ Chỉ mục (.index): Đầy đủ", fg=Colors.SUCCESS)
        else:
            self.lbl_model_readiness.config(text="✗ Thiếu tệp mô hình", fg=Colors.DANGER)
            self.lbl_index_readiness.config(text="✗ Thiếu tệp chỉ mục", fg=Colors.DANGER)

        # Button state
        if profile.is_active:
            self.btn_load.config(text="✓ Hồ sơ này đang hoạt động trong GPU", state="disabled")
        else:
            self.btn_load.config(text=f"Nạp '{profile.display_name}' vào GPU", state="normal")

    def _on_apply_clicked(self) -> None:
        if self._selected_id:
            self.on_profile_selected(self._selected_id)

    def set_loading(self, is_loading: bool) -> None:
        """Update button state while profile is loading into GPU."""
        if is_loading:
            self.profile_combo.config(state="disabled")
            self.btn_load.config(text="⏳ Đang nạp mô hình vào GPU (2–8s)...", state="disabled")
        else:
            self.profile_combo.config(state="readonly")
            idx = self.profile_combo.current()
            if 0 <= idx < len(self._profiles):
                self._update_display(self._profiles[idx])
