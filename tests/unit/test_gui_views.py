"""Unit tests for newly added commercial GUI views (Sidebar, VoiceLibraryView, DiagnosticsPanel)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import tkinter as tk

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.conversion.base import ConversionConfig
from voice_changer.gui.theme import Colors, setup_theme
from voice_changer.gui.views.diagnostics_panel import DiagnosticsPanel
from voice_changer.gui.views.library_view import VoiceLibraryView
from voice_changer.gui.views.profiles_panel import ProfilesPanel
from voice_changer.gui.views.sidebar import Sidebar
from voice_changer.service import EngineState, EngineStatus, ErrorInfo, ProfileSummary


class TestCommercialViews(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        setup_theme(self.root)

        self.mock_profiles = [
            ProfileSummary(
                profile_id="vctk_p231",
                display_name="VCTK Female p231",
                description="English Prototype",
                status="prototype",
                language="en",
                defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.75, filter_radius=3, protect_rate=0.33),
                files_present=True,
                is_active=True,
                notes="Standard test model",
                readiness="files_present",
            ),
            ProfileSummary(
                profile_id="voice_a_preview",
                display_name="Speaker A (Northern)",
                description="Northern Vietnamese Female",
                status="preview",
                language="vi",
                defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=2, protect_rate=0.40),
                files_present=True,
                is_active=False,
                notes="Preview e50",
                readiness="files_present",
            ),
            ProfileSummary(
                profile_id="jett",
                display_name="Jett",
                description="Planned profile. No dataset, model or index yet.",
                status="planned",
                language="en",
                defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
                files_present=False,
                is_active=False,
                notes="Planned game character",
                readiness="no_model",
                setup_hint="Chưa có dataset và checkpoint trong hệ thống.",
            ),
        ]

    def tearDown(self) -> None:
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        self.root.destroy()

    def test_sidebar_navigation_and_status(self) -> None:
        """Verify sidebar navigation callbacks and live engine status updates."""
        nav_target = []
        sidebar = Sidebar(self.root, on_navigate=lambda v: nav_target.append(v), current_view="studio")
        sidebar.pack()

        # Click navigation items
        sidebar.set_active_view("library")
        self.assertEqual(sidebar._current_view, "library")

        sidebar.set_active_view("diagnostics")
        self.assertEqual(sidebar._current_view, "diagnostics")

        # Navigation only: engine state is shown by the header pill and the Controls card, not repeated here.
        self.assertFalse(hasattr(sidebar, "lbl_engine_state"))
        self.assertFalse(hasattr(sidebar, "update_status"))

    def test_voice_library_view_rendering_and_filtering(self) -> None:
        """Verify Voice Library card rendering, search filtering, and planned model hints."""
        selected_profiles = []
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: selected_profiles.append(pid))
        library.pack()

        library.set_profiles(self.mock_profiles, active_id="vctk_p231")

        # Initial profile count
        self.assertIn("3 / 3", library.lbl_count.cget("text"))

        # Filter by language 'vi'
        library._on_filter_chip_clicked("vi")
        self.assertEqual(len(library._filtered_profiles), 1)
        self.assertEqual(library._filtered_profiles[0].profile_id, "voice_a_preview")

        # Filter by planned
        library._on_filter_chip_clicked("planned")
        self.assertEqual(len(library._filtered_profiles), 1)
        self.assertEqual(library._filtered_profiles[0].profile_id, "jett")

        # Search query
        library._on_filter_chip_clicked("all")
        library.var_search.set("Northern")
        self.assertEqual(len(library._filtered_profiles), 1)
        self.assertEqual(library._filtered_profiles[0].profile_id, "voice_a_preview")

        # Clear search
        library.var_search.set("")
        self.assertEqual(len(library._filtered_profiles), 3)

    def test_diagnostics_panel_metrics_and_error(self) -> None:
        """Verify DiagnosticsPanel telemetry updates and error notifications."""
        diag = DiagnosticsPanel(self.root, log_path="test_diagnostics.log")
        diag.pack()

        sample_metrics = {
            "rtf": 0.380,
            "avg_inference_ms": 190.0,
            "avg_software_e2e_ms": 520.0,
            "current_buffer_ms": 150.0,
            "chunks_converted": 120,
            "chunks_dropped": 0,
            "steady_underflows": 0,
        }
        diag.update_metrics(sample_metrics)

        self.assertEqual(diag.val_diag_rtf.cget("text"), "0.380")
        self.assertEqual(diag.val_diag_gpu.cget("text"), "190 ms")
        self.assertEqual(diag.val_diag_e2e.cget("text"), "~520 ms")
        self.assertEqual(diag.val_diag_chunks.cget("text"), "120 / 0")

        # Error notification
        err_status = EngineStatus(
            state=EngineState.ERROR,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=True,
            config=None,
            audio=None,
            session_id=1,
            health_problems=(),
            last_error=ErrorInfo(
                category="audio_device",
                code="driver_disconnected",
                message="Microphone bị ngắt kết nối",
                hint="Kiểm tra lại cáp kết nối",
                details="PortAudio err -9985",
                timestamp="2026-10-22T10:00:00",
            ),
        )
        diag.update_status(err_status)
        self.assertIn("Microphone bị ngắt kết nối", diag.lbl_diag_err_msg.cget("text"))
        self.assertIn("Kiểm tra lại cáp kết nối", diag.lbl_diag_err_hint.cget("text"))


    def test_profiles_panel_planned_jett_and_sage(self) -> None:
        """Planned Jett and Sage say they have no model; the Studio action refuses to load them."""
        from voice_changer.gui.presentation import primary_action

        selected_ids = []
        panel = ProfilesPanel(
            self.root,
            service=mock.MagicMock(),
            on_profile_selected=lambda pid: selected_ids.append(pid),
        )
        panel.pack()

        sage_profile = ProfileSummary(
            profile_id="sage",
            display_name="Sage",
            description="Planned Valorant agent.",
            status="planned",
            language="en",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=False,
            is_active=False,
            notes="Planned profile",
            readiness="no_model",
            setup_hint="Chưa có dataset và checkpoint trong hệ thống.",
        )
        profiles = list(self.mock_profiles) + [sage_profile]
        for pid in ("jett", "sage"):
            panel.set_profiles(profiles, current_id=pid)
            self.assertEqual(panel.get_selected_profile().profile_id, pid)
            self.assertEqual(panel.pill_status.cget("text"), "🔒 Chưa có mô hình")
            self.assertIn("Chưa thể chuyển giọng: giọng này chưa có mô hình.", panel.lbl_readiness_msg.cget("text"))
            self.assertIn("Chưa có dataset và checkpoint", panel.lbl_readiness_msg.cget("text"))
            action = primary_action(EngineState.READY, panel.get_selected_profile(), "vctk_p231")
            self.assertEqual(action.kind, "none")
            self.assertFalse(action.enabled)

        # The picker offers only loadable voices (plus the current choice), so planned voices are not listed
        # beside them; the Library is where they are browsed.
        panel.set_profiles(profiles, current_id="vctk_p231")
        self.assertEqual(panel._combo_ids, ["vctk_p231", "voice_a_preview"])
        self.assertEqual(selected_ids, [])

    def test_profiles_panel_ready_profile_and_switching_state(self) -> None:
        """A loadable voice is offered for loading; switching never leaves a stale status; loaded comes from the engine."""
        selected_ids = []
        panel = ProfilesPanel(
            self.root,
            service=mock.MagicMock(),
            on_profile_selected=lambda pid: selected_ids.append(pid),
        )
        panel.pack()

        panel.set_profiles(self.mock_profiles, current_id="voice_a_preview")
        self.assertEqual(panel.pill_status.cget("text"), "Thử nghiệm")
        self.assertIn("sẽ được kiểm tra đầy đủ khi nạp", panel.lbl_readiness_msg.cget("text"))
        self.assertEqual(str(panel.profile_combo.cget("state")), "readonly")

        # Choosing in the picker reports the choice to the app (which validates and selects); nothing loads here.
        panel.profile_combo.current(0)
        panel._on_combo_changed(tk.Event())
        self.assertEqual(selected_ids, ["vctk_p231"])

        # Planned Jett chosen (e.g. from the Library), then back: no stale planned warning.
        panel.select_profile("jett")
        self.assertEqual(panel.pill_status.cget("text"), "🔒 Chưa có mô hình")
        panel.select_profile("voice_a_preview")
        self.assertEqual(panel.pill_status.cget("text"), "Thử nghiệm")
        self.assertNotIn("Chưa thể", panel.lbl_readiness_msg.cget("text"))

        # Only the engine reporting it loaded marks it in use.
        panel.set_active_profile("voice_a_preview")
        self.assertEqual(panel.pill_status.cget("text"), "● Đang dùng")
        self.assertIn("Đã nạp và kiểm tra tệp thành công", panel.lbl_readiness_msg.cget("text"))
        panel.select_profile("jett")
        self.assertEqual(panel.pill_status.cget("text"), "🔒 Chưa có mô hình")
        panel.select_profile("voice_a_preview")
        self.assertEqual(panel.pill_status.cget("text"), "● Đang dùng")

        # While loading the picker is locked.
        panel.set_loading(True)
        self.assertEqual(str(panel.profile_combo.cget("state")), "disabled")
        panel.set_loading(False)
        self.assertEqual(str(panel.profile_combo.cget("state")), "readonly")

    def test_profiles_panel_malformed_readiness_fails_safely(self) -> None:
        """Verify profile with missing/None readiness fails safely and is never offered for loading."""
        from voice_changer.gui.presentation import primary_action

        selected_ids = []
        panel = ProfilesPanel(
            self.root,
            service=mock.MagicMock(),
            on_profile_selected=lambda pid: selected_ids.append(pid),
        )
        panel.pack()

        # Malformed profile: files_present is False, readiness is None
        bad_profile = ProfileSummary(
            profile_id="corrupt_voice",
            display_name="Corrupt Profile",
            description="Malformed profile with missing files.",
            status="prototype",
            language="vi",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=False,
            is_active=False,
            notes="",
        )
        # Force readiness to None to simulate malformed/missing backend field
        object.__setattr__(bad_profile, "readiness", None)

        panel.set_profiles([bad_profile], current_id="corrupt_voice")

        # Must NOT assume ready: warning shown, no load offered
        self.assertEqual(panel.pill_status.cget("text"), "⚠ Thiếu tệp mô hình")
        self.assertIn("Chưa thể chuyển giọng", panel.lbl_readiness_msg.cget("text"))
        action = primary_action(EngineState.IDLE, panel.get_selected_profile(), None)
        self.assertFalse(action.enabled)
        self.assertEqual(selected_ids, [])

    def test_voice_library_view_set_active_profile_does_not_flicker_or_destroy_cards(self) -> None:
        """Verify that calling set_active_profile repeatedly (e.g. from polling loop) does not destroy cards."""
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()

        library.set_profiles(self.mock_profiles, active_id="vctk_p231")

        # Initial card widgets
        vctk_card = library._rendered_cards["vctk_p231"]["card"]
        vctk_pill = library._rendered_cards["vctk_p231"]["pill_status"]
        voice_a_card = library._rendered_cards["voice_a_preview"]["card"]

        self.assertEqual(vctk_pill.cget("text"), "● Đang dùng")

        # Simulate 10 metric polling cycles (300ms each) with the same active profile
        for _ in range(10):
            library.set_active_profile("vctk_p231")

        # Card widgets MUST be identical objects (not destroyed and recreated!)
        self.assertIs(library._rendered_cards["vctk_p231"]["card"], vctk_card)
        self.assertIs(library._rendered_cards["vctk_p231"]["pill_status"], vctk_pill)
        self.assertIs(library._rendered_cards["voice_a_preview"]["card"], voice_a_card)

        # Switch active profile to voice_a_preview
        library.set_active_profile("voice_a_preview")

        # In-place update: cards still survive without full DOM rebuild
        self.assertIs(library._rendered_cards["voice_a_preview"]["card"], voice_a_card)
        self.assertEqual(library._rendered_cards["voice_a_preview"]["pill_status"].cget("text"), "● Đang dùng")
        self.assertNotEqual(library._rendered_cards["vctk_p231"]["pill_status"].cget("text"), "● Đang dùng")


    def test_audio_settings_view_components_and_callbacks(self) -> None:
        """Verify AudioSettingsView controls, specifications labels, and callbacks."""
        from voice_changer.audio.devices import AudioDevice
        from voice_changer.gui.views.audio_settings_view import AudioSettingsView
        from voice_changer.service import DeviceInventory

        changed_devices = []
        refreshed = []
        view = AudioSettingsView(
            self.root,
            on_devices_changed=lambda inp, out: changed_devices.append((inp, out)),
            on_refresh_requested=lambda: refreshed.append(True),
        )
        view.pack()

        mic = AudioDevice(10, "Focusrite Scarlett Solo", "WASAPI", 2, 0, 48000.0)
        cable = AudioDevice(11, "CABLE Input (VB-Audio)", "MME", 0, 2, 44100.0)
        inv = DeviceInventory(
            inputs=(mic,),
            outputs=(cable,),
            default_input_id=10,
            default_output_id=11,
        )

        view.set_inventory(inv, selected_input_id=10, selected_output_id=11)

        # Inspect specs labels
        self.assertIn("WASAPI", view.lbl_in_specs.cget("text"))
        self.assertIn("48000 Hz", view.lbl_in_specs.cget("text"))
        self.assertIn("CÁP ẢO", view.lbl_out_specs.cget("text"))

        # Test refresh button
        view.btn_refresh.invoke()
        self.assertEqual(len(refreshed), 1)

        # Test streaming mode lock
        view.set_streaming(True)
        self.assertEqual(str(view.input_combo.cget("state")), "disabled")
        self.assertEqual(str(view.output_combo.cget("state")), "disabled")
        self.assertEqual(str(view.btn_refresh.cget("state")), "disabled")

        view.set_streaming(False)
        self.assertEqual(str(view.input_combo.cget("state")), "readonly")
        self.assertEqual(str(view.output_combo.cget("state")), "readonly")
        self.assertEqual(str(view.btn_refresh.cget("state")), "normal")

    def test_voice_library_truthful_specifications(self) -> None:
        """Verify Voice Library cards display honest specs and don't fabricate RVC v2 for unverified profiles."""
        withdrawn_profile = ProfileSummary(
            profile_id="voice_withdrawn",
            display_name="Withdrawn Voice",
            description="Withdrawn test model",
            status="prototype",
            language="vi",
            defaults=ConversionConfig(pitch_shift=0.0, index_rate=0.5, filter_radius=3, protect_rate=0.33),
            files_present=True,
            is_active=False,
            notes="",
            withdrawn_reason="Known distortion issue",
        )
        profiles = list(self.mock_profiles) + [withdrawn_profile]
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles(profiles, active_id="vctk_p231")

        # 1. Ready profile (vctk_p231) displays only what ProfileSummary carries (its default config);
        # the checkpoint's RVC version and sample rate are not in the summary, so they are not claimed.
        vctk_card = library._rendered_cards["vctk_p231"]["card"]
        specs_vctk = [w for w in vctk_card.winfo_children() if isinstance(w, tk.Frame) and any(isinstance(c, tk.Label) and "F0:" in c.cget("text") for c in w.winfo_children())]
        self.assertTrue(len(specs_vctk) > 0)
        lbl_vctk_spec = [c for c in specs_vctk[0].winfo_children() if isinstance(c, tk.Label) and "F0:" in c.cget("text")][0]
        self.assertEqual(lbl_vctk_spec.cget("text"), "F0: RMVPE • Pitch mặc định: +12")
        self.assertNotIn("RVC v2", lbl_vctk_spec.cget("text"))
        self.assertNotIn("kHz", lbl_vctk_spec.cget("text"))

        # 2. Planned profile without model (jett) must display honest state, NOT fabricated RVC v2
        jett_card = library._rendered_cards["jett"]["card"]
        specs_jett = [w for w in jett_card.winfo_children() if isinstance(w, tk.Frame) and any(isinstance(c, tk.Label) and "Mô hình:" in c.cget("text") for c in w.winfo_children())]
        self.assertTrue(len(specs_jett) > 0)
        lbl_jett_spec = [c for c in specs_jett[0].winfo_children() if isinstance(c, tk.Label) and "Mô hình:" in c.cget("text")][0]
        self.assertIn("Chưa cài đặt", lbl_jett_spec.cget("text"))
        self.assertNotIn("RVC v2", lbl_jett_spec.cget("text"))

        # 3. Withdrawn profile displays withdrawal status
        withdrawn_card = library._rendered_cards["voice_withdrawn"]["card"]
        specs_w = [w for w in withdrawn_card.winfo_children() if isinstance(w, tk.Frame) and any(isinstance(c, tk.Label) and "Trạng thái:" in c.cget("text") for c in w.winfo_children())]
        self.assertTrue(len(specs_w) > 0)
        lbl_w_spec = [c for c in specs_w[0].winfo_children() if isinstance(c, tk.Label) and "Trạng thái:" in c.cget("text")][0]
        self.assertIn("Đã thu hồi", lbl_w_spec.cget("text"))

    def test_voice_library_dynamic_card_wraplength(self) -> None:
        """Verify Voice Library card dynamically updates label wraplength on resize without truncating text."""
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles(self.mock_profiles, active_id="vctk_p231")

        card_data = library._rendered_cards["vctk_p231"]
        desc_label = card_data["lbl_desc"]

        # Directly verify calculated wrap behavior
        calc_wrap = max(180, 500 - 32)
        desc_label.config(wraplength=calc_wrap)
        self.assertEqual(int(desc_label.cget("wraplength")), 468)

    def test_sidebar_keyboard_accessibility(self) -> None:
        """Verify sidebar navigation buttons have keyboard focus and bindings for Return/Space."""
        nav_target = []
        sidebar = Sidebar(self.root, on_navigate=lambda v: nav_target.append(v), current_view="studio")
        sidebar.pack()

        # Check takefocus and keyboard bindings on navigation buttons
        for view_id in ("studio", "library", "audio_settings", "diagnostics"):
            btn_frame = sidebar._buttons[view_id][0]
            self.assertTrue(btn_frame.cget("takefocus") in (1, "1", True))
            self.assertTrue(bool(btn_frame.bind("<Return>")))
            self.assertTrue(bool(btn_frame.bind("<space>")))
            self.assertTrue(bool(btn_frame.bind("<FocusIn>")))
            self.assertTrue(bool(btn_frame.bind("<FocusOut>")))

    def test_library_filter_chips_and_search_clear_keyboard_accessibility(self) -> None:
        """Verify filter chips and search clear have keyboard focus and bindings for Return/Space."""
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles(self.mock_profiles, active_id="vctk_p231")

        # Filter chip focus & Return/Space bindings
        for cat_id, chip in library._chip_buttons.items():
            self.assertTrue(chip.cget("takefocus") in (1, "1", True))
            self.assertTrue(bool(chip.bind("<Return>")))
            self.assertTrue(bool(chip.bind("<space>")))
            self.assertTrue(bool(chip.bind("<FocusIn>")))
            self.assertTrue(bool(chip.bind("<FocusOut>")))

        # Search query and clear via keyboard binding
        library.var_search.set("female")
        self.assertTrue(library.btn_clear.cget("takefocus") in (1, "1", True))
        self.assertTrue(bool(library.btn_clear.bind("<Return>")))
        self.assertTrue(bool(library.btn_clear.bind("<space>")))
        self.assertTrue(bool(library.btn_clear.bind("<FocusIn>")))
        self.assertTrue(bool(library.btn_clear.bind("<FocusOut>")))
        library._on_clear_search()
        self.assertEqual(library.var_search.get(), "")

    def test_tuning_slider_keyboard_debounced_apply(self) -> None:
        """Verify tuning slider applies changes on debounced KeyRelease and immediately on FocusOut."""
        from voice_changer.gui.views.tuning_panel import TuningPanel
        applied_configs = []
        panel = TuningPanel(
            self.root,
            on_config_changed=lambda cfg: applied_configs.append(cfg),
            on_reset_requested=lambda: None,
        )
        panel.pack()

        initial_cfg = ConversionConfig(pitch_shift=0.0, index_rate=0.5, filter_radius=3, protect_rate=0.33)
        panel.set_config(initial_cfg)

        # Change slider value via keyboard
        panel.val_pitch.set(5.0)
        panel._on_slider_released()
        self.assertEqual(len(applied_configs), 1)
        self.assertEqual(applied_configs[0]["pitch_shift"], 5.0)

        # Test immediate apply on FocusOut
        panel.val_index.set(0.85)
        panel._debounce_timer = panel.after(1000, lambda: None)
        panel._apply_immediate_if_pending()
        self.assertIsNone(panel._debounce_timer)
        self.assertEqual(len(applied_configs), 2)
        self.assertEqual(applied_configs[1]["index_rate"], 0.85)

    def test_profiles_panel_active_profile_sync_does_not_clobber_user_browsing(self) -> None:
        """Verify that repeated active profile status synchronization does not reset combobox selection."""
        panel = ProfilesPanel(
            self.root,
            service=mock.MagicMock(),
            on_profile_selected=lambda pid: None,
        )
        panel.pack()
        panel.set_profiles(self.mock_profiles, current_id="vctk_p231")
        panel.set_active_profile("vctk_p231")

        # User browses to Jett (e.g. chosen in the Library; the picker lists only loadable voices)
        panel.select_profile("jett")
        self.assertEqual(panel.get_selected_profile().profile_id, "jett")

        # Metric polling calls set_active_profile("vctk_p231") repeatedly
        panel.set_active_profile("vctk_p231")
        panel.set_active_profile("vctk_p231")

        # Selection MUST remain on Jett (not clobbered by poll)
        self.assertEqual(panel.get_selected_profile().profile_id, "jett")

        # But when active profile changes to voice_a_preview, it synchronizes!
        panel.set_active_profile("voice_a_preview")
        self.assertEqual(panel.get_selected_profile().profile_id, "voice_a_preview")

    def test_noise_panel_buttons_keyboard_focus_and_activation(self) -> None:
        """Verify NoiseSuppressionPanel buttons have visible focus outlines and Return activation."""
        from voice_changer.gui.views.noise_panel import NoiseSuppressionPanel
        from voice_changer.service import NoiseSuppressionStatus

        noise_changes = []
        rearmed = []
        panel = NoiseSuppressionPanel(
            self.root,
            on_noise_changed=lambda en, mode: noise_changes.append((en, mode)),
            on_rearm_requested=lambda: rearmed.append(True),
        )
        panel.pack()

        # btn_toggle focus styling
        btn_toggle = panel.btn_toggle
        self.assertEqual(int(btn_toggle.cget("highlightthickness")), 2)
        self.assertEqual(btn_toggle.cget("highlightbackground"), Colors.BG_SURFACE)

        btn_toggle._focus_in()
        self.assertEqual(btn_toggle.cget("highlightbackground"), Colors.ACCENT_CYAN)

        btn_toggle._focus_out()
        self.assertEqual(btn_toggle.cget("highlightbackground"), Colors.BG_SURFACE)

        # Return activation
        self.assertTrue(bool(btn_toggle.bind("<Return>")))
        self.assertTrue(bool(btn_toggle.bind("<FocusIn>")))
        self.assertTrue(bool(btn_toggle.bind("<FocusOut>")))
        btn_toggle.invoke()
        self.assertEqual(len(noise_changes), 1)

        # Show error banner to expose btn_rearm
        status_err = NoiseSuppressionStatus(
            enabled=True,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=False,
            failed=True,
            last_error="Crash in filter",
        )
        panel.set_status(status_err)
        btn_rearm = panel.btn_rearm
        self.assertEqual(int(btn_rearm.cget("highlightthickness")), 2)
        self.assertEqual(btn_rearm.cget("highlightbackground"), Colors.DANGER_BG)

        btn_rearm._focus_in()
        self.assertEqual(btn_rearm.cget("highlightbackground"), Colors.ACCENT_CYAN)

        btn_rearm._focus_out()
        self.assertEqual(btn_rearm.cget("highlightbackground"), Colors.DANGER_BG)

        self.assertTrue(bool(btn_rearm.bind("<Return>")))
        self.assertTrue(bool(btn_rearm.bind("<FocusIn>")))
        self.assertTrue(bool(btn_rearm.bind("<FocusOut>")))
        btn_rearm.invoke()
        self.assertEqual(len(rearmed), 1)

    def test_controls_panel_error_banner_focus_and_wrapping(self) -> None:
        """Verify ControlsPanel btn_err_details focus outline and responsive error text wrapping."""
        from voice_changer.gui.views.controls_panel import ControlsPanel
        from voice_changer.service import EngineState, EngineStatus, ErrorInfo

        panel = ControlsPanel(
            self.root,
            on_start_requested=lambda: None,
            on_stop_requested=lambda: None,
        )
        panel.pack()

        # Display an error to map error banner
        err = ErrorInfo(
            code="DEV_NOT_FOUND",
            category="audio_device",
            message="Microphone USB Audio Device is disconnected or unreachable by the system.",
            hint="Please reconnect your microphone USB cable and select it again.",
            details="Traceback (most recent call last):\n  PortAudio error -9996",
            timestamp="2026-10-23T09:00:00",
        )
        status = EngineStatus(
            state=EngineState.ERROR,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=False,
            config=None,
            audio=None,
            session_id=1,
            health_problems=(),
            last_error=err,
        )
        panel.update_status(status)

        # P2: Test btn_err_details focus indicator
        btn_details = panel.btn_err_details
        self.assertEqual(int(btn_details.cget("highlightthickness")), 2)
        self.assertEqual(btn_details.cget("highlightbackground"), Colors.DANGER_BG)

        btn_details._focus_in()
        self.assertEqual(btn_details.cget("highlightbackground"), Colors.ACCENT_CYAN)

        btn_details._focus_out()
        self.assertEqual(btn_details.cget("highlightbackground"), Colors.DANGER_BG)

        self.assertTrue(bool(btn_details.bind("<Return>")))
        self.assertTrue(bool(btn_details.bind("<FocusIn>")))
        self.assertTrue(bool(btn_details.bind("<FocusOut>")))

        # Return activation triggers dialog
        with mock.patch("tkinter.messagebox.showerror") as mock_box:
            btn_details.invoke()
            mock_box.assert_called_once()
            self.assertIn("DEV_NOT_FOUND", mock_box.call_args[1]["detail"])

        # P4: Test responsive wrapping on resize
        # Simulate card configure with width 400
        panel._on_card_configure(400)
        self.assertEqual(int(panel.lbl_error_msg.cget("wraplength")), 400 - 58)
        self.assertEqual(int(panel.lbl_error_hint.cget("wraplength")), 400 - 58)

        # Simulate banner configure with width 350
        panel._on_banner_configure(350)
        self.assertEqual(int(panel.lbl_error_msg.cget("wraplength")), 350 - 28)
        self.assertEqual(int(panel.lbl_error_hint.cget("wraplength")), 350 - 28)

        # Minimum clamp bound check (width 100 clamps to 180)
        panel._on_banner_configure(100)
        self.assertEqual(int(panel.lbl_error_msg.cget("wraplength")), 180)
        self.assertEqual(int(panel.lbl_error_hint.cget("wraplength")), 180)

    def test_p3_engine_status_pill_harmony(self) -> None:
        """Verify consistent status pill wording and color styling across sidebar, controls, and theme mappings."""
        from voice_changer.gui.theme import ENGINE_STATE_VISUALS, get_engine_state_visual
        from voice_changer.gui.views.controls_panel import ControlsPanel
        from voice_changer.service import EngineState, EngineStatus

        # 1. Authoritative dictionary validation for all 7 EngineState members
        all_states = [
            EngineState.IDLE,
            EngineState.LOADING,
            EngineState.READY,
            EngineState.STARTING,
            EngineState.RUNNING,
            EngineState.STOPPING,
            EngineState.ERROR,
        ]
        for state in all_states:
            self.assertIn(state, ENGINE_STATE_VISUALS)
            text, bg, fg = get_engine_state_visual(state)
            self.assertTrue(text.startswith("● "))
            self.assertTrue(bool(bg))
            self.assertTrue(bool(fg))

        # Check specific expected states
        self.assertEqual(get_engine_state_visual(EngineState.IDLE)[0], "● CHƯA NẠP GIỌNG")
        self.assertEqual(get_engine_state_visual(EngineState.STARTING)[0], "● ĐANG KHỞI ĐỘNG…")
        self.assertEqual(get_engine_state_visual(EngineState.RUNNING)[0], "● ĐANG CHUYỂN GIỌNG")
        self.assertEqual(get_engine_state_visual(EngineState.RUNNING)[1], Colors.SUCCESS)
        self.assertEqual(get_engine_state_visual(EngineState.RUNNING)[2], Colors.TEXT_ON_PRIMARY)
        self.assertEqual(get_engine_state_visual(EngineState.ERROR)[0], "● CÓ LỖI")

        # 2. ControlsPanel pill (beside the action) follows the authoritative mapping for every state
        panel = ControlsPanel(self.root, on_start_requested=lambda: None, on_stop_requested=lambda: None)
        panel.pack()
        self.assertEqual(panel.pill_state.cget("text"), "● CHƯA NẠP GIỌNG")

        for state in all_states:
            status = EngineStatus(
                state=state,
                profile_id="vctk_p231" if state != EngineState.IDLE else None,
                profile_status="prototype" if state != EngineState.IDLE else None,
                model_loaded=state not in (EngineState.IDLE, EngineState.ERROR),
                config=None,
                audio=None,
                session_id=1,
                health_problems=(),
                last_error=None,
            )
            panel.update_status(status)
            expected_text, expected_bg, expected_fg = get_engine_state_visual(state)
            self.assertEqual(panel.pill_state.cget("text"), expected_text)
            self.assertEqual(panel.pill_state.cget("bg"), expected_bg)
            self.assertEqual(panel.pill_state.cget("fg"), expected_fg)

        # Verify button action semantics preserved
        # STARTING: disabled
        status_start = EngineStatus(state=EngineState.STARTING, profile_id="p", profile_status="p", model_loaded=True, config=None, audio=None, session_id=1, health_problems=(), last_error=None)
        panel.update_status(status_start)
        self.assertEqual(str(panel.btn_action.cget("state")), "disabled")

        # RUNNING: normal and shows STOP
        status_run = EngineStatus(state=EngineState.RUNNING, profile_id="p", profile_status="p", model_loaded=True, config=None, audio=None, session_id=1, health_problems=(), last_error=None)
        panel.update_status(status_run)
        self.assertEqual(str(panel.btn_action.cget("state")), "normal")
        self.assertIn("Dừng", panel.btn_action.cget("text"))

    def test_p5_diagnostics_responsive_wrapping(self) -> None:
        """Verify DiagnosticsPanel error card labels update wrap lengths responsively and avoid configure loops."""
        diag = DiagnosticsPanel(self.root, log_path="test_diagnostics.log")
        diag.pack()

        err_text = "Detailed stack trace showing PortAudio error -9985: Device unavailable or unplugged during live conversion."
        err_status = EngineStatus(
            state=EngineState.ERROR,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=True,
            config=None,
            audio=None,
            session_id=1,
            health_problems=(),
            last_error=ErrorInfo(
                category="audio_device",
                code="driver_disconnected",
                message="Microphone bị ngắt kết nối trong lúc đang chạy luồng âm thanh",
                hint="Kiểm tra lại cáp kết nối USB của thiết bị",
                details=err_text,
                timestamp="2026-10-23T10:00:00",
            ),
        )
        diag.update_status(err_status)

        # Full original text must be preserved
        self.assertIn("Microphone bị ngắt kết nối", diag.lbl_diag_err_msg.cget("text"))
        self.assertIn("Kiểm tra lại cáp kết nối", diag.lbl_diag_err_hint.cget("text"))
        self.assertEqual(diag.lbl_diag_err_detail.cget("text"), err_text)

        # Simulate resize to container width 600
        diag._on_card_error_configure(600)
        self.assertEqual(int(diag.lbl_diag_err_msg.cget("wraplength")), 600 - 36)
        self.assertEqual(int(diag.lbl_diag_err_hint.cget("wraplength")), 600 - 36)
        self.assertEqual(int(diag.lbl_diag_err_detail.cget("wraplength")), 600 - 52)

        # Simulate resize to wider container width 900
        diag._on_card_error_configure(900)
        self.assertEqual(int(diag.lbl_diag_err_msg.cget("wraplength")), 900 - 36)
        self.assertEqual(int(diag.lbl_diag_err_hint.cget("wraplength")), 900 - 36)
        self.assertEqual(int(diag.lbl_diag_err_detail.cget("wraplength")), 900 - 52)

        # Fluctuations < 4px should be ignored to avoid configure loops
        diag._on_card_error_configure(902)
        self.assertEqual(int(diag.lbl_diag_err_detail.cget("wraplength")), 900 - 52)

        # Narrow container clamps to minimum bound 200
        diag._on_card_error_configure(150)
        self.assertEqual(int(diag.lbl_diag_err_msg.cget("wraplength")), 200)
        self.assertEqual(int(diag.lbl_diag_err_detail.cget("wraplength")), 200)

        # Text remained unchanged throughout all resizes
        self.assertEqual(diag.lbl_diag_err_detail.cget("text"), err_text)

    def test_p5_audio_settings_responsive_wrapping(self) -> None:
        """Verify AudioSettingsView feedback warning and troubleshooting tips wrap responsively without loops."""
        from voice_changer.gui.views.audio_settings_view import AudioSettingsView

        settings = AudioSettingsView(self.root)
        settings.pack()

        orig_warn = settings.lbl_feedback_warn.cget("text")
        self.assertIn("Không chọn Loa ngoài (Speakers)", orig_warn)
        self.assertIn("acoustic feedback loop", orig_warn)

        # 1. Feedback warning wrapping on warn_box resize
        settings._on_warn_box_configure(700)
        self.assertEqual(int(settings.lbl_feedback_warn.cget("wraplength")), 700 - 28)

        # Wider container width 950
        settings._on_warn_box_configure(950)
        self.assertEqual(int(settings.lbl_feedback_warn.cget("wraplength")), 950 - 28)

        # Minor oscillation < 4px ignored
        settings._on_warn_box_configure(952)
        self.assertEqual(int(settings.lbl_feedback_warn.cget("wraplength")), 950 - 28)

        # Minimum clamp to 200
        settings._on_warn_box_configure(100)
        self.assertEqual(int(settings.lbl_feedback_warn.cget("wraplength")), 200)

        # Full warning text preserved
        self.assertEqual(settings.lbl_feedback_warn.cget("text"), orig_warn)

        # 2. Troubleshooting tips wrapping on card_trouble resize
        self.assertTrue(len(settings._trouble_labels) >= 3)
        orig_tip0 = settings._trouble_labels[0].cget("text")

        settings._on_card_trouble_configure(750)
        for lbl in settings._trouble_labels:
            self.assertEqual(int(lbl.cget("wraplength")), 750 - 32)

        # Minor oscillation < 4px ignored
        settings._on_card_trouble_configure(753)
        for lbl in settings._trouble_labels:
            self.assertEqual(int(lbl.cget("wraplength")), 750 - 32)

        # Narrow container clamps to 200
        settings._on_card_trouble_configure(120)
        for lbl in settings._trouble_labels:
            self.assertEqual(int(lbl.cget("wraplength")), 200)

        self.assertEqual(settings._trouble_labels[0].cget("text"), orig_tip0)

        # 3. Step cards (Discord & OBS) wrapping
        self.assertTrue(len(settings._discord_labels) >= 1)
        orig_step0 = settings._discord_labels[0].cget("text")
        settings._on_step_card_configure(380, settings._discord_labels, "discord")
        for lbl in settings._discord_labels:
            self.assertEqual(int(lbl.cget("wraplength")), 380 - 32)
        self.assertEqual(settings._discord_labels[0].cget("text"), orig_step0)

    def test_profiles_panel_hero_avatar_and_category_update(self) -> None:
        """Verify ProfilesPanel updates hero avatar canvas and category tag on profile switch."""
        panel = ProfilesPanel(self.root, service=mock.MagicMock())  # no app callback: the picker selects directly
        panel.pack()
        panel.set_profiles(self.mock_profiles, current_id="vctk_p231")

        # Initial profile is vctk_p231
        self.assertEqual(panel.lbl_hero_name.cget("text"), "VCTK Female p231")
        self.assertIn("tiếng Anh", panel.lbl_hero_category.cget("text"))
        self.assertTrue(len(panel.avatar_canvas.find_all()) > 0)
        self.assertEqual(panel.avatar_canvas.avatar_spec.glyph, "EN")

        # Switch to voice_a_preview
        panel.profile_combo.current(1)
        panel._on_combo_changed(tk.Event())
        self.assertEqual(panel.lbl_hero_name.cget("text"), "Speaker A (Northern)")
        self.assertIn("Bắc", panel.lbl_hero_category.cget("text"))
        self.assertEqual(panel.avatar_canvas.avatar_spec.glyph, "VA")

    def test_voice_library_view_avatar_and_action_button(self) -> None:
        """Verify VoiceLibraryView card avatars and 'Dùng trong Studio' action buttons."""
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles(self.mock_profiles, active_id="vctk_p231")

        # Active profile vctk_p231
        active_card = library._rendered_cards["vctk_p231"]
        active_btn = active_card["btn_action_box"].winfo_children()[0]
        self.assertIn("Đang dùng trong Studio", active_btn.cget("text"))

        # Inactive selectable profile voice_a_preview
        voice_a_card = library._rendered_cards["voice_a_preview"]
        voice_a_btn = voice_a_card["btn_action_box"].winfo_children()[0]
        self.assertIn("Dùng trong Studio", voice_a_btn.cget("text"))

        # Avatar widget is present in the card data
        self.assertIn("avatar_widget", active_card)
        self.assertIsNotNone(active_card["avatar_widget"])

    def _withdrawn_voice_a(self) -> ProfileSummary:
        return ProfileSummary(
            profile_id="voice_a_preview", display_name="Voice A (preview)", description="Speaker A",
            status="preview", language="vi",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.5, filter_radius=2, protect_rate=0.4),
            files_present=True, is_active=False, notes="", readiness="files_present",
            withdrawn_reason="Withdrawn 2026-10-22 at the owner's request.")

    def test_voice_library_withdrawn_profile_is_distinct_and_cannot_be_loaded(self) -> None:
        selected = []
        library = VoiceLibraryView(self.root, on_profile_selected=selected.append)
        library.pack()
        preview = ProfileSummary(
            profile_id="voice_b_preview", display_name="Voice B (preview)", description="Speaker B",
            status="preview", language="vi", defaults=self.mock_profiles[1].defaults, files_present=True,
            is_active=False, notes="", readiness="files_present")
        missing = ProfileSummary(
            profile_id="missing_voice", display_name="Missing", description="", status="preview", language="en",
            defaults=self.mock_profiles[1].defaults, files_present=False, is_active=False, notes="",
            readiness="files_missing")
        library.set_profiles([self._withdrawn_voice_a(), preview, missing, self.mock_profiles[2]], active_id=None)

        pills = {pid: library._rendered_cards[pid]["pill_status"] for pid in
                 ("voice_a_preview", "voice_b_preview", "missing_voice", "jett")}
        looks = {pid: (p.cget("text"), p.cget("bg"), p.cget("fg")) for pid, p in pills.items()}
        self.assertIn("thu hồi", looks["voice_a_preview"][0])
        # Withdrawn, preview, missing-file and planned profiles differ in color, not only in wording.
        self.assertEqual(len({(bg, fg) for _, bg, fg in looks.values()}), 4)

        action = library._rendered_cards["voice_a_preview"]["btn_action_box"].winfo_children()[0]
        self.assertNotIsInstance(action, tk.Button)
        self.assertIn("Không thể nạp", action.cget("text"))
        library._on_card_load_clicked("voice_a_preview")
        self.assertEqual(selected, [])

    def test_studio_panel_shows_withdrawn_voice_a_as_not_loadable(self) -> None:
        selected = []
        panel = ProfilesPanel(self.root, service=mock.MagicMock(), on_profile_selected=selected.append)
        panel.pack()
        panel.set_profiles([self._withdrawn_voice_a()], current_id="voice_a_preview")
        self.assertIn("thu hồi", panel.pill_status.cget("text"))
        self.assertIn("không thể nạp", panel.lbl_readiness_msg.cget("text"))
        self.assertIn("Withdrawn 2026-10-22", panel.lbl_readiness_msg.cget("text"))
        self.assertEqual(panel._combo_ids, ["voice_a_preview"])  # shown only because it is the current choice
        from voice_changer.gui.presentation import primary_action
        for state in (EngineState.IDLE, EngineState.READY, EngineState.ERROR):
            self.assertFalse(primary_action(state, panel.get_selected_profile(), None).enabled, state)
        self.assertEqual(selected, [])

    def test_unknown_profile_gets_a_generic_local_avatar(self) -> None:
        from voice_changer.gui.presentation import DEFAULT_AVATAR_COLOR, monogram

        unknown = ProfileSummary(
            profile_id="brand_new_voice", display_name="New", description="", status="prototype", language="en",
            defaults=self.mock_profiles[0].defaults, files_present=True, is_active=False, notes="")
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles([unknown])
        avatar = library._rendered_cards["brand_new_voice"]["avatar_widget"]
        texts = [avatar.itemcget(i, "text") for i in avatar.find_all() if avatar.type(i) == "text"]
        self.assertEqual(texts, [monogram("New")])
        self.assertEqual(avatar.avatar_spec.color, DEFAULT_AVATAR_COLOR)
        self.assertIsNone(avatar.avatar_spec.image)  # never a remote or bundled third-party image
        self.assertEqual(library._rendered_cards["brand_new_voice"]["lbl_cat"].cget("text"), "Giọng tiếng anh")

    def test_voice_presentation_matches_the_registry_metadata(self) -> None:
        from voice_changer.gui.presentation import load_voice_presentation

        table = load_voice_presentation()
        # Registry: voice B is "warm/soft female (English-leaning source)"; no regional accent is recorded.
        self.assertNotIn("miền", table["voice_b_preview"].tagline.lower())
        # Jett and Sage are registered third-party preview models, not planned profiles.
        for pid in ("jett", "sage"):
            self.assertNotIn("dự kiến", table[pid].tagline.lower())
            self.assertIn("bên thứ ba", table[pid].tagline)
        # Voice A is described as withdrawn.
        self.assertIn("thu hồi", table["voice_a_preview"].tagline)

    def test_voice_library_search_matches_category_and_initial_chip_looks_selected(self) -> None:
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles(self.mock_profiles)
        all_chip = library._chip_buttons["all"]
        self.assertEqual((all_chip.cget("bg"), all_chip.cget("fg")), (Colors.PRIMARY, Colors.TEXT_ON_PRIMARY))

        library.var_search.set("nhân vật")  # only in jett's tagline
        self.assertEqual([p.profile_id for p in library._filtered_profiles], ["jett"])
        library.var_search.set("")
        library._on_filter_chip_clicked("vi")
        self.assertEqual([p.profile_id for p in library._filtered_profiles], ["voice_a_preview"])
        library._on_filter_chip_clicked("available")
        self.assertEqual([p.profile_id for p in library._filtered_profiles], ["vctk_p231", "voice_a_preview"])

    def test_voice_library_card_details_come_only_from_profile_summary_fields(self) -> None:
        from types import SimpleNamespace

        known = ProfileSummary(
            profile_id="jett", display_name="Jett", description="", status="preview", language="en",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.5, filter_radius=3, protect_rate=0.33),
            files_present=True, is_active=False, notes="", readiness="files_present", data_rights="unverified")
        self.assertEqual(VoiceLibraryView._verified_spec_text(known),
                         "F0: RMVPE • Pitch mặc định: +12 • Bản quyền: chưa xác minh (chỉ dùng cá nhân)")

        # "unspecified" rights make no claim, so nothing is shown for them.
        unspecified = ProfileSummary(
            profile_id="voice_b_preview", display_name="Voice B", description="", status="preview", language="vi",
            defaults=ConversionConfig(pitch_shift=14.0, index_rate=0.25, filter_radius=2, protect_rate=0.4),
            files_present=True, is_active=False, notes="", readiness="files_present")
        self.assertEqual(VoiceLibraryView._verified_spec_text(unspecified), "F0: RMVPE • Pitch mặc định: +14")

        # No usable metadata at all: say so instead of falling back to a generic architecture.
        missing = SimpleNamespace(profile_id="x", defaults=None, data_rights="unspecified")
        text = VoiceLibraryView._verified_spec_text(missing)
        self.assertEqual(text, "Thông số mô hình: không có dữ liệu")

        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles([known, unspecified])
        for pid in ("jett", "voice_b_preview"):
            card = library._rendered_cards[pid]["card"]
            texts = [c.cget("text") for w in card.winfo_children() for c in w.winfo_children()
                     if isinstance(c, tk.Label)]
            self.assertFalse(any("RVC v2" in t or "40 kHz" in t for t in texts), pid)

    def test_voice_library_card_layout_settles(self) -> None:
        # A card wraplength derived from the card's own width once made its size oscillate forever.
        long_category = ProfileSummary(
            profile_id="jett", display_name="Jett", description="", status="preview", language="en",
            defaults=ConversionConfig(), files_present=True, is_active=False, notes="", readiness="files_present",
            data_rights="unverified")
        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles([long_category])
        widths = []
        library._rendered_cards["jett"]["card"].bind("<Configure>", lambda e: widths.append(e.width), add="+")
        self.root.after(1000, self.root.quit)
        self.root.mainloop()
        self.assertLess(len(widths), 20, widths[:10])

        # Wrapping follows the window-driven canvas width: half of it minus grid and card padding.
        library._canvas_width = 900
        library._apply_card_wraps()
        card = library._rendered_cards["jett"]
        self.assertEqual(int(card["lbl_desc"].cget("wraplength")), 900 // 2 - 52)
        self.assertEqual(int(card["lbl_cat"].cget("wraplength")), 900 // 2 - 52)

    def test_rtf_color_is_the_same_in_studio_and_diagnostics(self) -> None:
        from voice_changer.gui.views.controls_panel import ControlsPanel

        controls = ControlsPanel(self.root, on_start_requested=lambda: None, on_stop_requested=lambda: None)
        diag = DiagnosticsPanel(self.root, log_path="x.log")
        for rtf in (0.5, 0.7, 0.95, 1.2):
            m = {"live": True, "rtf": rtf, "avg_inference_ms": rtf * 500, "estimated_software_latency_ms": 600.0,
                 "current_buffer_ms": 150.0, "chunks_converted": 10, "chunks_dropped": 0, "steady_underflows": 0}
            controls.update_performance(m)
            diag.update_metrics(m)
            # Studio's one-line verdict uses the same thresholds as the Advanced RTF value.
            self.assertEqual(controls.lbl_perf.cget("fg"), diag.val_diag_rtf.cget("fg"), rtf)

    def test_device_hints_wrap_to_the_card_width(self) -> None:
        from voice_changer.audio.devices import AudioDevice
        from voice_changer.gui.views.audio_settings_view import AudioSettingsView
        from voice_changer.gui.views.devices_panel import DevicesPanel
        from voice_changer.service import DeviceInventory

        mic = AudioDevice(1, "Mic", "Windows WASAPI", 1, 0, 48000.0)
        out = AudioDevice(2, "Speakers", "MME", 0, 2, 48000.0)
        inv = DeviceInventory(inputs=(mic,), outputs=(out,), default_input_id=1, default_output_id=2)

        studio = DevicesPanel(self.root, on_devices_changed=lambda i, o: None, on_refresh_requested=lambda: None)
        studio.pack()
        studio.set_inventory(inv, 1, 2)
        # PortAudio reports "Windows WASAPI"; the warning used to require the exact name "WASAPI".
        self.assertIn("WASAPI có thể từ chối 40 kHz", studio.lbl_input_hint.cget("text"))
        studio.winfo_children()[0].event_generate("<Configure>", width=300, height=200, when="now")
        self.assertEqual(int(studio.lbl_input_hint.cget("wraplength")), 300 - 32)
        self.assertEqual(int(studio.lbl_output_hint.cget("wraplength")), 300 - 32)

        settings = AudioSettingsView(self.root)
        settings.pack()
        settings.set_inventory(inv, 1, 2)
        # The WASAPI warning sits beside the input list on this page too.
        self.assertIn("WASAPI có thể từ chối 40 kHz", settings.lbl_in_specs.cget("text"))
        self.assertEqual(settings.lbl_in_specs.cget("fg"), Colors.WARNING_TEXT)
        self.assertNotIn("từ chối", settings.lbl_out_specs.cget("text"))
        self.assertGreater(int(settings.lbl_in_specs.cget("wraplength")), 0)
        self.assertTrue(any("rejects 40000 Hz" in lbl.cget("text") for lbl in settings._trouble_labels))


if __name__ == "__main__":
    unittest.main()


