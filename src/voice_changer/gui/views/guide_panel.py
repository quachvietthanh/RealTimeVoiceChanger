"""VB-Cable, Discord, and OBS routing setup guide panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from voice_changer.gui.theme import Colors, Fonts


class GuidePanel(ttk.Frame):
    """Panel providing visual setup guidance for VB-Cable, Discord, OBS, and troubleshooting."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self._build_ui()

    def _build_ui(self) -> None:
        canvas = tk.Canvas(self, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(xscrollcommand=None, yscrollcommand=scrollbar.set)

        # Mousewheel scroll handler
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=12)
        scrollbar.pack(side="right", fill="y", padx=(0, 16), pady=12)

        # Header Title
        lbl_title = tk.Label(
            scrollable_frame,
            text="Hướng dẫn kết nối Cáp ảo VB-Cable (Discord & OBS)",
            font=Fonts.TITLE,
            bg=Colors.BG_DARK,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_title.pack(anchor="w", pady=(0, 4))

        lbl_sub = tk.Label(
            scrollable_frame,
            text="Định tuyến âm thanh chuyển đổi từ Micro sang Discord/OBS theo thời gian thực mà không gây trễ tiếng.",
            font=Fonts.BODY,
            bg=Colors.BG_DARK,
            fg=Colors.TEXT_MUTED,
        )
        lbl_sub.pack(anchor="w", pady=(0, 14))

        # Visual Signal Flow Banner
        flow_card = tk.Frame(
            scrollable_frame,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.PRIMARY,
            highlightthickness=1,
            padx=16,
            pady=12,
        )
        flow_card.pack(fill="x", pady=(0, 14))

        tk.Label(
            flow_card,
            text="SƠ ĐỒ ĐƯỜNG TRUYỀN TÍN HIỆU ÂM THANH:",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.PRIMARY_LIGHT,
            anchor="w",
        ).pack(fill="x")

        flow_diag = (
            "🎙️ Microphone của bạn  ──▶  [ RealTimeVoiceChanger ]  ──▶  🔊 CABLE Input (VB-Cable)\n"
            "                                                                  │\n"
            "                                                                  ▼ Cáp ảo Windows\n"
            "🎮 Discord / OBS Studio  ◀──────────────────────────────────  🎙️ CABLE Output"
        )
        tk.Label(
            flow_card,
            text=flow_diag,
            font=("Consolas", 10, "bold"),
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
            justify="left",
            pady=6,
        ).pack(fill="x")

        # Step Grid: 2 Columns of Step Cards
        steps_grid = ttk.Frame(scrollable_frame, style="TFrame")
        steps_grid.pack(fill="x", pady=(0, 14))
        steps_grid.columnconfigure(0, weight=1, uniform="step")
        steps_grid.columnconfigure(1, weight=1, uniform="step")

        # Step 1
        self._create_step_card(
            steps_grid,
            row=0,
            col=0,
            badge="BƯỚC 1",
            title="Cài đặt Driver VB-Audio CABLE",
            items=[
                "Tải driver miễn phí từ trang chủ: vb-audio.com/Cable/",
                "Giải nén tệp zip, nhấp chuột phải 'VBCABLE_Setup_x64.exe' chọn Run as administrator.",
                "Khởi động lại máy tính Windows sau khi cài đặt hoàn tất.",
            ],
        )

        # Step 2
        self._create_step_card(
            steps_grid,
            row=0,
            col=1,
            badge="BƯỚC 2",
            title="Cấu hình trong RealTimeVoiceChanger",
            items=[
                "Microphone đầu vào: Chọn Micro thật của bạn (ưu tiên cổng [MME] hoặc [DirectSound]).",
                "Ngõ ra âm thanh: Chọn đúng 'CABLE Input (VB-Audio Virtual Cable)'.",
                "Nhấn nút 'Bắt đầu chuyển đổi' để khởi chạy luồng xử lý giọng nói.",
            ],
        )

        # Step 3
        self._create_step_card(
            steps_grid,
            row=1,
            col=0,
            badge="BƯỚC 3",
            title="Cấu hình trong Discord",
            items=[
                "Mở Discord ➔ Cài đặt người dùng (Settings) ➔ Thoại & Video (Voice & Video).",
                "Thiết bị đầu vào (Input Device): Chọn 'CABLE Output (VB-Audio Virtual Cable)'.",
                "Thiết bị đầu ra (Output Device): Chọn Loa hoặc Tai nghe thật của bạn.",
                "⚠️ QUAN TRỌNG: Tắt 'Khử tiếng ồn Krisp' trong Discord để không bị méo âm sắc giọng nữ.",
            ],
        )

        # Step 4
        self._create_step_card(
            steps_grid,
            row=1,
            col=1,
            badge="BƯỚC 4",
            title="Cấu hình trong OBS Studio",
            items=[
                "Trong OBS, thêm nguồn Audio mới: 'Audio Input Capture' (Thu lại âm thanh đầu vào).",
                "Mục Device: Chọn 'CABLE Output (VB-Audio Virtual Cable)'.",
                "Kiểm tra thanh âm lượng trên OBS xem tín hiệu đã nhảy đều theo giọng nói chưa.",
            ],
        )

        # Troubleshooting Section
        trouble_card = tk.Frame(
            scrollable_frame,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=16,
            pady=14,
        )
        trouble_card.pack(fill="x", pady=(0, 20))

        tk.Label(
            trouble_card,
            text="❓ Khắc phục sự cố thường gặp (Troubleshooting)",
            font=Fonts.SUBTITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.WARNING_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        faqs = [
            (
                "Lỗi 'rejects 40000 Hz' khi nhấn Bắt đầu:",
                "Nguyên nhân do chọn cổng WASAPI không hỗ trợ 40 kHz. Khắc phục: Chuyển sang cổng [MME] hoặc [DirectSound] của cùng thiết bị.",
            ),
            (
                "Discord hoặc OBS không nghe thấy âm thanh gì:",
                "Kiểm tra lại xem RealTimeVoiceChanger đã chọn ngõ ra là 'CABLE Input' chưa, và Discord đã chọn ngõ vào là 'CABLE Output' chưa.",
            ),
            (
                "Âm thanh bị giật, vấp tiếng hoặc Gián đoạn âm tăng cao:",
                "GPU bị quá tải khiến RTF > 1.0. Hãy đóng bớt các game nặng hoặc ứng dụng 3D đang chạy ngầm.",
            ),
            (
                "Âm lượng của Voice B nhỏ hơn Voice A:",
                "Mô hình Voice B thử nghiệm có âm lượng đầu ra nhỏ hơn ~9 dB. Hãy tăng nhẹ mức âm lượng Micro trong Windows hoặc Discord.",
            ),
        ]

        for q, a in faqs:
            q_frame = tk.Frame(trouble_card, bg=Colors.BG_SURFACE)
            q_frame.pack(fill="x", pady=3)

            tk.Label(
                q_frame,
                text=f"• {q}",
                font=Fonts.BODY_BOLD,
                bg=Colors.BG_SURFACE,
                fg=Colors.TEXT_PRIMARY,
                anchor="w",
            ).pack(fill="x")

            tk.Label(
                q_frame,
                text=f"  {a}",
                font=Fonts.CAPTION,
                bg=Colors.BG_SURFACE,
                fg=Colors.TEXT_SECONDARY,
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=(1, 0))

    def _create_step_card(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        badge: str,
        title: str,
        items: list[str],
    ) -> None:
        card = tk.Frame(
            parent,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        hdr = tk.Frame(card, bg=Colors.BG_SURFACE)
        hdr.pack(fill="x", pady=(0, 6))

        tk.Label(
            hdr,
            text=f" {badge} ",
            font=Fonts.BADGE,
            bg=Colors.PRIMARY,
            fg=Colors.TEXT_ON_PRIMARY,
            padx=6,
            pady=2,
        ).pack(side="left")

        tk.Label(
            hdr,
            text=f" {title}",
            font=Fonts.HEADING,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
            anchor="w",
        ).pack(side="left")

        for item in items:
            tk.Label(
                card,
                text=f"• {item}",
                font=Fonts.CAPTION,
                bg=Colors.BG_SURFACE,
                fg=Colors.TEXT_SECONDARY,
                anchor="w",
                justify="left",
                wraplength=380,
            ).pack(fill="x", pady=1)

        # Dynamic wrapping on resize
        card.bind(
            "<Configure>",
            lambda e: [
                w.config(wraplength=max(180, e.width - 24))
                for w in card.winfo_children()
                if isinstance(w, tk.Label) and w.cget("text").startswith("•")
            ],
        )
