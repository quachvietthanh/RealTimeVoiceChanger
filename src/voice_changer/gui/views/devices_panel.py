"""Audio device selection and configuration panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from voice_changer.audio.devices import AudioDevice
from voice_changer.gui.theme import Colors, Fonts
from voice_changer.service import DeviceInventory


class DevicesPanel(ttk.Frame):
    """Clean, balanced panel for selecting input microphone and output playback devices."""

    def __init__(
        self,
        parent: tk.Widget,
        on_devices_changed: Callable[[Optional[AudioDevice], Optional[AudioDevice]], None],
        on_refresh_requested: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self.on_devices_changed = on_devices_changed
        self.on_refresh_requested = on_refresh_requested

        self._inventory: Optional[DeviceInventory] = None
        self._selected_input: Optional[AudioDevice] = None
        self._selected_output: Optional[AudioDevice] = None

        self._build_ui()

    def _build_ui(self) -> None:
        card = tk.Frame(
            self,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=16,
            pady=14,
        )
        card.pack(fill="x", expand=True)

        # Header with refresh button
        header_row = tk.Frame(card, bg=Colors.BG_SURFACE)
        header_row.pack(fill="x", pady=(0, 10))

        lbl_title = tk.Label(
            header_row,
            text="Thiết bị âm thanh (Audio Endpoints)",
            font=Fonts.SUBTITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_title.pack(side="left")

        self.btn_refresh = ttk.Button(
            header_row,
            text="🔄 Làm mới",
            style="Secondary.TButton",
            command=self.on_refresh_requested,
        )
        self.btn_refresh.pack(side="right")

        # 1. Microphone Input Section
        lbl_in = tk.Label(
            card,
            text="Microphone đầu vào (Input):",
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_in.pack(anchor="w", pady=(0, 3))

        self.input_combo = ttk.Combobox(card, state="readonly", style="TCombobox")
        self.input_combo.pack(fill="x", ipady=2, pady=(0, 2))
        self.input_combo.bind("<<ComboboxSelected>>", self._on_input_selected)

        self.lbl_input_hint = tk.Label(
            card,
            text="",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
            anchor="w",
        )
        self.lbl_input_hint.pack(fill="x", pady=(0, 8))

        # 2. Output Device Section
        lbl_out = tk.Label(
            card,
            text="Ngõ ra âm thanh (Output / Cáp ảo VB-Cable):",
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_out.pack(anchor="w", pady=(0, 3))

        self.output_combo = ttk.Combobox(card, state="readonly", style="TCombobox")
        self.output_combo.pack(fill="x", ipady=2, pady=(0, 2))
        self.output_combo.bind("<<ComboboxSelected>>", self._on_output_selected)

        self.lbl_output_hint = tk.Label(
            card,
            text="",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
            anchor="w",
            justify="left",
        )
        self.lbl_output_hint.pack(fill="x", pady=(0, 4))

        # Responsive label wrapping
        card.bind(
            "<Configure>",
            lambda e: self.lbl_output_hint.config(wraplength=max(200, e.width - 32)),
        )

    def set_inventory(
        self,
        inventory: DeviceInventory,
        selected_input_id: Optional[int] = None,
        selected_output_id: Optional[int] = None,
    ) -> None:
        """Populate device options from backend inventory."""
        self._inventory = inventory

        # Inputs
        input_texts = []
        target_in_idx = 0
        target_in_id = selected_input_id if selected_input_id is not None else inventory.default_input_id

        for i, dev in enumerate(inventory.inputs):
            is_def = " (Mặc định)" if dev.device_id == inventory.default_input_id else ""
            txt = f"[{dev.host_api}] {dev.name}{is_def}"
            input_texts.append(txt)
            if dev.device_id == target_in_id:
                target_in_idx = i

        self.input_combo["values"] = input_texts
        if input_texts:
            self.input_combo.current(target_in_idx)
            self._selected_input = inventory.inputs[target_in_idx]
            self._update_input_hint(self._selected_input)

        # Outputs
        output_texts = []
        target_out_idx = 0
        target_out_id = selected_output_id if selected_output_id is not None else inventory.default_output_id

        for i, dev in enumerate(inventory.outputs):
            is_def = " (Mặc định)" if dev.device_id == inventory.default_output_id else ""
            cable_prefix = "⭐ [Cáp ảo] " if dev.is_virtual_cable else ""
            txt = f"{cable_prefix}[{dev.host_api}] {dev.name}{is_def}"
            output_texts.append(txt)
            if dev.device_id == target_out_id:
                target_out_idx = i

        self.output_combo["values"] = output_texts
        if output_texts:
            self.output_combo.current(target_out_idx)
            self._selected_output = inventory.outputs[target_out_idx]
            self._update_output_hint(self._selected_output)

        self._notify_changed()

    def _on_input_selected(self, event: tk.Event) -> None:
        if not self._inventory:
            return
        idx = self.input_combo.current()
        if 0 <= idx < len(self._inventory.inputs):
            self._selected_input = self._inventory.inputs[idx]
            self._update_input_hint(self._selected_input)
            self._notify_changed()

    def _on_output_selected(self, event: tk.Event) -> None:
        if not self._inventory:
            return
        idx = self.output_combo.current()
        if 0 <= idx < len(self._inventory.outputs):
            self._selected_output = self._inventory.outputs[idx]
            self._update_output_hint(self._selected_output)
            self._notify_changed()

    def _update_input_hint(self, dev: AudioDevice) -> None:
        if dev.host_api.upper() in ("WASAPI", "WDM-KS"):
            self.lbl_input_hint.config(
                text="⚠️ WASAPI có thể từ chối 40 kHz. Khuyên chọn cổng [MME] hoặc [DirectSound].",
                fg=Colors.WARNING_TEXT,
            )
        else:
            self.lbl_input_hint.config(
                text=f"✓ Cổng {dev.host_api} hỗ trợ 40 kHz | Kênh: {dev.max_input_channels}",
                fg=Colors.TEXT_MUTED,
            )

    def _update_output_hint(self, dev: AudioDevice) -> None:
        if dev.is_virtual_cable:
            self.lbl_output_hint.config(
                text="⭐ Đã chọn Cáp ảo VB-Cable. Giọng chuyển đổi sẽ truyền trực tiếp vào Discord/OBS.",
                fg=Colors.SUCCESS_TEXT,
            )
        elif dev.host_api.upper() in ("WASAPI", "WDM-KS"):
            self.lbl_output_hint.config(
                text="⚠️ WASAPI có thể từ chối 40 kHz. Khuyên chọn cổng [MME] hoặc [DirectSound].",
                fg=Colors.WARNING_TEXT,
            )
        else:
            self.lbl_output_hint.config(
                text="ℹ️ Phát ra loa/tai nghe để nghe thử. Để truyền vào Discord, hãy chọn 'CABLE Input'.",
                fg=Colors.TEXT_MUTED,
            )

    def _notify_changed(self) -> None:
        self.on_devices_changed(self._selected_input, self._selected_output)

    def get_selected_devices(self) -> tuple[Optional[AudioDevice], Optional[AudioDevice]]:
        return self._selected_input, self._selected_output

    def set_streaming(self, is_streaming: bool) -> None:
        state = "disabled" if is_streaming else "readonly"
        self.input_combo.config(state=state)
        self.output_combo.config(state=state)
        self.btn_refresh.config(state="disabled" if is_streaming else "normal")
