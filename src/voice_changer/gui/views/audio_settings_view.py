"""Audio & Connections: the single place to choose the microphone and the output device.

The guidance under the device lists is built from the devices actually selected (presentation.route_summary).
The app only plays the converted voice into the chosen output; other applications receive it when that
output is a virtual audio cable and they select the cable's recording side as their microphone. There is no
direct integration with any game or chat application. Longer setup guides are in collapsible help sections.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from voice_changer.audio.devices import AudioDevice
from voice_changer.gui.presentation import route_summary
from voice_changer.gui.theme import Colors, Fonts
from voice_changer.gui.views.devices_panel import WASAPI_WARNING, may_reject_40k
from voice_changer.gui.views.widgets import CollapsibleSection
from voice_changer.service import AudioSettings, DeviceInventory

logger = logging.getLogger(__name__)


def device_label(dev: AudioDevice, default_id: Optional[int]) -> str:
    """Combobox text: name first, then the host API, then default / virtual-cable markers."""
    suffix = " · mặc định" if dev.device_id == default_id else ""
    cable = " · cáp ảo" if dev.is_virtual_cable else ""
    return f"{dev.name}  ({dev.host_api}){suffix}{cable}"


class AudioSettingsView(ttk.Frame):
    """Device selection, contextual routing guidance, and optional help for Discord / OBS / games."""

    def __init__(
        self,
        parent: tk.Widget,
        on_devices_changed: Optional[Callable[[Optional[AudioDevice], Optional[AudioDevice]], None]] = None,
        on_refresh_requested: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, style="TFrame")
        self.on_devices_changed = on_devices_changed
        self.on_refresh_requested = on_refresh_requested

        self._inventory: Optional[DeviceInventory] = None
        self._selected_input: Optional[AudioDevice] = None
        self._selected_output: Optional[AudioDevice] = None
        self._missing_input: Optional[str] = None
        self._missing_output: Optional[str] = None
        # Side the user changed in the event being reported ('input'/'output'); read by the app to save only that side.
        self.last_user_change: Optional[str] = None

        self._last_warn_box_width: int = 0
        self._last_trouble_width: int = 0
        self._trouble_labels: list[tk.Label] = []
        self._discord_labels: list[tk.Label] = []
        self._obs_labels: list[tk.Label] = []
        self._game_labels: list[tk.Label] = []
        self._cable_labels: list[tk.Label] = []

        self._build_ui()

    def _build_ui(self) -> None:
        canvas = tk.Canvas(self, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        page = tk.Frame(canvas, bg=Colors.BG_DARK)
        page.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=page, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=14)
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=14)

        # 1. Devices
        bg = Colors.BG_SURFACE
        card_dev = tk.Frame(page, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1, padx=18, pady=16)
        card_dev.pack(fill="x", pady=(0, 12))
        header_row = tk.Frame(card_dev, bg=bg)
        header_row.pack(fill="x", pady=(0, 10))
        self.btn_refresh = ttk.Button(header_row, text="Làm mới danh sách", style="Secondary.TButton",
                                      command=self._on_refresh_clicked)
        self.btn_refresh.pack(side="right")
        tk.Label(header_row, text="Thiết bị", font=Fonts.HEADING, bg=bg, fg=Colors.TEXT_PRIMARY).pack(side="left")

        tk.Label(card_dev, text="Microphone (giọng thật của bạn)", font=Fonts.BODY_BOLD, bg=bg,
                 fg=Colors.TEXT_PRIMARY).pack(anchor="w", pady=(0, 4))
        self.input_combo = ttk.Combobox(card_dev, state="readonly", style="TCombobox")
        self.input_combo.pack(fill="x", ipady=2, pady=(0, 4))
        self.input_combo.bind("<<ComboboxSelected>>", self._on_input_selected)
        self.lbl_in_specs = tk.Label(card_dev, text="Chưa chọn thiết bị", font=Fonts.CAPTION, bg=bg,
                                     fg=Colors.TEXT_MUTED, anchor="w", justify="left", wraplength=320)
        self.lbl_in_specs.pack(fill="x", pady=(0, 12))

        tk.Label(card_dev, text="Ngõ ra (nơi phát giọng đã chuyển)", font=Fonts.BODY_BOLD, bg=bg,
                 fg=Colors.TEXT_PRIMARY).pack(anchor="w", pady=(0, 4))
        self.output_combo = ttk.Combobox(card_dev, state="readonly", style="TCombobox")
        self.output_combo.pack(fill="x", ipady=2, pady=(0, 4))
        self.output_combo.bind("<<ComboboxSelected>>", self._on_output_selected)
        self.lbl_out_specs = tk.Label(card_dev, text="Chưa chọn thiết bị", font=Fonts.CAPTION, bg=bg,
                                      fg=Colors.TEXT_MUTED, anchor="w", justify="left", wraplength=320)
        self.lbl_out_specs.pack(fill="x")

        self.lbl_streaming_note = tk.Label(card_dev, text="", font=Fonts.CAPTION, bg=bg, fg=Colors.WARNING_TEXT,
                                           anchor="w", justify="left")
        self.lbl_streaming_note.pack(fill="x", pady=(8, 0))

        def _wrap_specs(e: tk.Event) -> None:
            wrap = max(160, e.width - 40)
            for lab in (self.lbl_in_specs, self.lbl_out_specs, self.lbl_streaming_note):
                lab.config(wraplength=wrap)

        card_dev.bind("<Configure>", _wrap_specs)

        # 2. What the selection means for other applications (from the actual output device)
        card_route = tk.Frame(page, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1, padx=18,
                              pady=14)
        card_route.pack(fill="x", pady=(0, 12))
        tk.Label(card_route, text="Dùng giọng đã chuyển trong ứng dụng khác", font=Fonts.HEADING, bg=bg,
                 fg=Colors.TEXT_PRIMARY, anchor="w").pack(fill="x", pady=(0, 6))
        self.lbl_route = tk.Label(card_route, text="", font=Fonts.BODY_BOLD, bg=bg, fg=Colors.TEXT_SECONDARY,
                                  anchor="w", justify="left")
        self.lbl_route.pack(fill="x")
        self.lbl_route_hint = tk.Label(card_route, text="", font=Fonts.BODY, bg=bg, fg=Colors.TEXT_SECONDARY,
                                       anchor="w", justify="left")
        self.lbl_route_hint.pack(fill="x", pady=(4, 8))

        self.warn_box = tk.Frame(card_route, bg=Colors.WARNING_BG, padx=10, pady=6)
        self.warn_box.pack(fill="x")
        self.warn_box.bind("<Configure>", lambda e: self._on_warn_box_configure(e.width))
        self.lbl_feedback_warn = tk.Label(
            self.warn_box,
            text="⚠ Không chọn Loa ngoài (Speakers) làm ngõ ra khi micro đang mở mà không đeo tai nghe: "
                 "loa phát lại vào micro gây tiếng hú (acoustic feedback loop).",
            font=Fonts.CAPTION, bg=Colors.WARNING_BG, fg=Colors.WARNING_TEXT, anchor="w", justify="left",
            wraplength=600,
        )
        self.lbl_feedback_warn.pack(fill="x")
        def _wrap_route(e: tk.Event) -> None:
            for lab in (self.lbl_route, self.lbl_route_hint):
                lab.config(wraplength=max(200, e.width - 40))

        card_route.bind("<Configure>", _wrap_route)

        # 3. Optional help
        tk.Label(page, text="Trợ giúp", font=Fonts.HEADING, bg=Colors.BG_DARK, fg=Colors.TEXT_PRIMARY,
                 anchor="w").pack(fill="x", pady=(6, 6))
        self.help_cable = self._help_section(page, "Cài cáp âm thanh ảo (VB-Cable)", [
            "1. Tải VB-CABLE Driver từ trang chính thức vb-audio.com/Cable (phần mềm donationware của VB-Audio, "
            "không đi kèm ứng dụng này).",
            "2. Giải nén, chạy 'VBCABLE_Setup_x64.exe' bằng quyền Administrator, rồi khởi động lại Windows.",
            "3. Đừng đặt 'CABLE Input' làm thiết bị phát mặc định của Windows (âm thanh hệ thống sẽ lọt vào cáp).",
            "4. Quay lại đây, nhấn 'Làm mới danh sách' và chọn ngõ ra 'CABLE Input'.",
        ], self._cable_labels)
        self.help_discord = self._help_section(page, "Discord", [
            "1. Mở Cài đặt người dùng → Giọng nói & Video (Voice & Video).",
            "2. Thiết bị đầu vào (Input Device): chọn 'CABLE Output (VB-Audio Virtual Cable)'.",
            "3. Thiết bị đầu ra: loa hoặc tai nghe thật của bạn.",
            "4. Nên tắt khử tiếng ồn (Krisp) và khử vọng của Discord khi thử lần đầu, để chúng không làm méo giọng.",
        ], self._discord_labels)
        self.help_obs = self._help_section(page, "OBS Studio", [
            "1. Thêm nguồn 'Audio Input Capture' (hoặc Settings → Audio → Mic/Auxiliary Audio).",
            "2. Chọn thiết bị 'CABLE Output (VB-Audio Virtual Cable)'.",
            "3. Ghi thử 30 giây và nghe lại trước khi phát trực tiếp.",
        ], self._obs_labels)
        self.help_games = self._help_section(page, "Game và ứng dụng khác", [
            "Ứng dụng này không tích hợp trực tiếp với game nào. Game hoặc ứng dụng nhận giọng đã chuyển "
            "khi bạn chọn 'CABLE Output' làm micro trong phần cài đặt âm thanh / voice chat của nó.",
            "Nếu game không cho chọn micro, nó thường dùng micro mặc định của Windows. Khi đó có thể tạm đặt "
            "'CABLE Output' làm thiết bị ghi âm mặc định, và nhớ đổi lại sau khi chơi.",
            f"Giọng đã chuyển đến chậm hơn giọng thật (xử lý theo khối "
            f"{AudioSettings().chunk_duration_s * 1000:.0f} ms), nên có độ trễ có thể cảm nhận được.",
        ], self._game_labels)

        self.help_trouble = self._help_section(page, "Khắc phục sự cố", [
            "• Ứng dụng khác không nghe thấy giọng: kiểm tra đã nhấn 'Bắt đầu chuyển giọng', ngõ ra ở đây là "
            "'CABLE Input', và ứng dụng kia chọn micro 'CABLE Output'.",
            "• Lỗi 'rejects 40000 Hz' khi bắt đầu: cổng WASAPI / WDM-KS không hỗ trợ 40 kHz. Chọn cùng thiết bị "
            "ở cổng (MME) hoặc (Windows DirectSound).",
            "• Âm thanh giật hoặc rè: xem 'Nâng cao' để biết có gián đoạn (underflow) hay chunk bị rơi không.",
            "• Micro không có tiếng: bật quyền Microphone trong Windows Settings → Privacy & Security → "
            "Microphone.",
        ], self._trouble_labels)
        self.help_trouble.body.bind("<Configure>", lambda e: self._on_card_trouble_configure(e.width))
        for section, labels, key in ((self.help_discord, self._discord_labels, "discord"),
                                     (self.help_obs, self._obs_labels, "obs"),
                                     (self.help_cable, self._cable_labels, "cable"),
                                     (self.help_games, self._game_labels, "games")):
            section.body.bind("<Configure>", lambda e, lb=labels, k=key: self._on_step_card_configure(e.width, lb, k))

        self._update_route()

    def _help_section(self, parent: tk.Widget, title: str, lines: list[str],
                      store: list[tk.Label]) -> CollapsibleSection:
        section = CollapsibleSection(parent, title, expanded=False)
        section.pack(fill="x", pady=(0, 6))
        for text in lines:
            lbl = tk.Label(section.body, text=text, font=Fonts.CAPTION, bg=Colors.BG_SURFACE,
                           fg=Colors.TEXT_SECONDARY, justify="left", anchor="w", wraplength=600)
            lbl.pack(fill="x", padx=16, pady=2)
            store.append(lbl)
        tk.Frame(section.body, bg=Colors.BG_SURFACE, height=8).pack(fill="x")
        return section

    # ----------------------------------------------------------------- public API
    def set_inventory(
        self,
        inventory: DeviceInventory,
        selected_input_id: Optional[int] = None,
        selected_output_id: Optional[int] = None,
    ) -> None:
        """Show `inventory` and select the given ids; a missing id falls back to the system default, then to the
        first device. Never reports a change: loading an inventory is not a user choice."""
        self._inventory = inventory
        self._missing_input = self._missing_output = None
        self.input_combo["values"] = [device_label(d, inventory.default_input_id) for d in inventory.inputs]
        self.output_combo["values"] = [device_label(d, inventory.default_output_id) for d in inventory.outputs]
        self._selected_input = self._pick(self.input_combo, inventory.inputs, selected_input_id,
                                          inventory.default_input_id)
        self._selected_output = self._pick(self.output_combo, inventory.outputs, selected_output_id,
                                           inventory.default_output_id)
        self._update_specs_labels()

    @staticmethod
    def _pick(combo: ttk.Combobox, devices: tuple[AudioDevice, ...], wanted: Optional[int],
              default_id: Optional[int]) -> Optional[AudioDevice]:
        if not devices:
            combo.set("")
            return None
        target = wanted if wanted is not None else default_id
        idx = next((i for i, d in enumerate(devices) if d.device_id == target), 0)
        combo.current(idx)
        return devices[idx]

    def get_selected_devices(self) -> tuple[Optional[AudioDevice], Optional[AudioDevice]]:
        return self._selected_input, self._selected_output

    def set_streaming(self, is_streaming: bool) -> None:
        """Lock device choices while a session is open (devices are fixed when it starts)."""
        state = "disabled" if is_streaming else "readonly"
        self.input_combo.config(state=state)
        self.output_combo.config(state=state)
        self.btn_refresh.config(state="disabled" if is_streaming else "normal")
        self.lbl_streaming_note.config(
            text="Đang chuyển giọng: dừng phiên ở Studio để đổi thiết bị." if is_streaming else "")

    def show_saved_device_unavailable(self, input_name: Optional[str], output_name: Optional[str]) -> None:
        """Explain that a saved device is unplugged and the current selection is only for this session.

        A side passed as None gets its normal label back, so a notice that no longer applies is cleared.
        """
        self._missing_input, self._missing_output = input_name, output_name
        self._update_specs_labels()

    # ----------------------------------------------------------------- internal callbacks
    def _on_input_selected(self, event: Optional[tk.Event]) -> None:
        if not self._inventory:
            return
        idx = self.input_combo.current()
        if 0 <= idx < len(self._inventory.inputs):
            self._selected_input = self._inventory.inputs[idx]
            self._missing_input = None
            self._update_specs_labels()
            self.last_user_change = "input"
            if self.on_devices_changed:
                self.on_devices_changed(self._selected_input, self._selected_output)

    def _on_output_selected(self, event: Optional[tk.Event]) -> None:
        if not self._inventory:
            return
        idx = self.output_combo.current()
        if 0 <= idx < len(self._inventory.outputs):
            self._selected_output = self._inventory.outputs[idx]
            self._missing_output = None
            self._update_specs_labels()
            self.last_user_change = "output"
            if self.on_devices_changed:
                self.on_devices_changed(self._selected_input, self._selected_output)

    def _on_refresh_clicked(self) -> None:
        if self.on_refresh_requested:
            self.on_refresh_requested()

    @staticmethod
    def _unavailable_note(name: str) -> str:
        return (f"⚠️ Thiết bị đã lưu '{name}' không khả dụng; tạm dùng thiết bị trên cho phiên này. "
                "Cắm lại rồi nhấn 'Làm mới danh sách' để khôi phục.")

    def _update_specs_labels(self) -> None:
        if self._missing_input:
            self.lbl_in_specs.config(text=self._unavailable_note(self._missing_input), fg=Colors.WARNING_TEXT)
        elif self._selected_input:
            d = self._selected_input
            text = f"Cổng {d.host_api} · {d.max_input_channels} kênh · {d.default_sample_rate:.0f} Hz"
            if may_reject_40k(d):
                text += f"\n{WASAPI_WARNING}"
            self.lbl_in_specs.config(text=text, fg=Colors.WARNING_TEXT if may_reject_40k(d) else Colors.TEXT_MUTED)
        else:
            self.lbl_in_specs.config(text="Không có microphone nào.", fg=Colors.WARNING_TEXT)

        if self._missing_output:
            self.lbl_out_specs.config(text=self._unavailable_note(self._missing_output), fg=Colors.WARNING_TEXT)
        elif self._selected_output:
            d = self._selected_output
            cable_badge = " · ✓ CÁP ẢO" if d.is_virtual_cable else ""
            text = f"Cổng {d.host_api} · {d.max_output_channels} kênh · {d.default_sample_rate:.0f} Hz{cable_badge}"
            if may_reject_40k(d):
                text += f"\n{WASAPI_WARNING}"
                fg = Colors.WARNING_TEXT
            else:
                fg = Colors.SUCCESS_TEXT if d.is_virtual_cable else Colors.TEXT_MUTED
            self.lbl_out_specs.config(text=text, fg=fg)
        else:
            self.lbl_out_specs.config(text="Không có thiết bị phát nào.", fg=Colors.WARNING_TEXT)
        self._update_route()

    def _update_route(self) -> None:
        summary = route_summary(self._selected_input, self._selected_output)
        self.lbl_route.config(text=summary.text)
        hint = summary.hint
        has_cable = bool(self._inventory and any(d.is_virtual_cable for d in self._inventory.outputs))
        if self._inventory is not None and not has_cable:
            hint += " Chưa thấy cáp âm thanh ảo nào trên máy: xem “Cài cáp âm thanh ảo” bên dưới."
        self.lbl_route_hint.config(text=hint)

    def _on_warn_box_configure(self, width: int) -> None:
        """Update acoustic feedback warning wraplength based on warn_box container width."""
        if width <= 1 or abs(width - self._last_warn_box_width) < 4:
            return
        self._last_warn_box_width = width
        self.lbl_feedback_warn.config(wraplength=max(200, width - 28))

    def _on_card_trouble_configure(self, width: int) -> None:
        """Update troubleshooting tips wraplength based on the help section width."""
        if width <= 1 or abs(width - self._last_trouble_width) < 4:
            return
        self._last_trouble_width = width
        for lbl in self._trouble_labels:
            lbl.config(wraplength=max(200, width - 32))

    def _on_step_card_configure(self, width: int, labels: list[tk.Label], card_id: str) -> None:
        """Update step labels wraplength based on the help section width."""
        last_width = getattr(self, f"_last_{card_id}_width", 0)
        if width <= 1 or abs(width - last_width) < 4:
            return
        setattr(self, f"_last_{card_id}_width", width)
        for lbl in labels:
            lbl.config(wraplength=max(160, width - 32))
