"""Unit tests for the GUI presentation layer: state-to-action mapping, truthful wording, the character catalog,
local-only assets, route and telemetry text (headless; no backend, audio device, model or network)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tkinter as tk
from tkinter import ttk

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.audio.devices import AudioDevice
from voice_changer.conversion.base import ConversionConfig
from voice_changer.gui import presentation as pres
from voice_changer.gui.app import VoiceChangerApp, noise_summary, tuning_summary
from voice_changer.gui.theme import setup_theme
from voice_changer.gui.views.diagnostics_panel import DiagnosticsPanel
from voice_changer.gui.views.library_view import VoiceLibraryView
from voice_changer.gui.views.sidebar import Sidebar, logo_file_for
from voice_changer.gui.views.widgets import create_avatar
from voice_changer.service import (
    AudioSettings,
    DeviceInventory,
    EngineState,
    EngineStatus,
    NoiseSuppressionStatus,
    ProfileSummary,
    VoiceChangerService,
)

DEFAULTS = ConversionConfig(pitch_shift=12.0, index_rate=0.5, filter_radius=3, protect_rate=0.33)


def profile(pid: str, **kw) -> ProfileSummary:
    fields = dict(profile_id=pid, display_name=pid.title(), description="", status="preview", language="en",
                  defaults=DEFAULTS, files_present=True, is_active=False, notes="", readiness="files_present")
    fields.update(kw)
    return ProfileSummary(**fields)


READY_B = profile("voice_b_preview", display_name="Voice B (preview)", language="vi")
VCTK = profile("vctk_p231", display_name="VCTK Female p231", status="prototype")
WITHDRAWN_A = profile("voice_a_preview", display_name="Voice A (preview)", language="vi",
                      withdrawn_reason="Withdrawn at the owner's request.")
PLANNED = profile("planned_voice", status="planned", files_present=False, readiness="no_model")
MISSING = profile("missing_voice", files_present=False, readiness="files_missing")


class TestPrimaryAction(unittest.TestCase):
    def test_selecting_a_voice_offers_load_never_start(self) -> None:
        for state in (EngineState.IDLE, EngineState.READY):
            action = pres.primary_action(state, READY_B, loaded_id="vctk_p231" if state == EngineState.READY else None)
            self.assertEqual((action.kind, action.text, action.enabled), ("load", "Nạp giọng", True), state)

    def test_start_only_when_the_backend_reports_the_selected_voice_loaded_and_ready(self) -> None:
        action = pres.primary_action(EngineState.READY, READY_B, loaded_id="voice_b_preview")
        self.assertEqual(action.kind, "start")
        self.assertEqual(action.style, "Success")
        # IDLE with the voice "selected" but not loaded is not startable.
        self.assertNotEqual(pres.primary_action(EngineState.IDLE, READY_B, loaded_id=None).kind, "start")

    def test_running_offers_stop_even_with_another_voice_selected(self) -> None:
        action = pres.primary_action(EngineState.RUNNING, VCTK, loaded_id="voice_b_preview")
        self.assertEqual((action.kind, action.style, action.enabled), ("stop", "Danger", True))

    def test_busy_states_and_loading_flag_disable_the_action(self) -> None:
        for state in (EngineState.LOADING, EngineState.STARTING, EngineState.STOPPING):
            action = pres.primary_action(state, READY_B, loaded_id="voice_b_preview")
            self.assertEqual(action.kind, "busy", state)
            self.assertFalse(action.enabled)
        self.assertFalse(pres.primary_action(EngineState.READY, READY_B, "voice_b_preview", loading=True).enabled)

    def test_error_offers_retry_for_the_loaded_voice_and_load_for_another(self) -> None:
        self.assertEqual(pres.primary_action(EngineState.ERROR, READY_B, "voice_b_preview").kind, "retry")
        self.assertEqual(pres.primary_action(EngineState.ERROR, VCTK, "voice_b_preview").kind, "load")

    def test_unloadable_voices_never_get_an_enabled_action(self) -> None:
        for p in (WITHDRAWN_A, PLANNED, MISSING):
            for state in (EngineState.IDLE, EngineState.READY, EngineState.ERROR):
                action = pres.primary_action(state, p, loaded_id="vctk_p231")
                self.assertFalse(action.enabled, (p.profile_id, state))
                self.assertNotIn(action.kind, ("load", "start"), (p.profile_id, state))

    def test_nothing_selected_and_nothing_loaded_is_disabled(self) -> None:
        action = pres.primary_action(EngineState.IDLE, None, None)
        self.assertFalse(action.enabled)


class TestAvailabilityAndWording(unittest.TestCase):
    def test_withdrawn_planned_and_missing_are_never_loadable(self) -> None:
        for p, key in ((WITHDRAWN_A, "withdrawn"), (PLANNED, "planned"), (MISSING, "files_missing")):
            av = pres.availability(p, active_id=p.profile_id)  # even if (wrongly) reported active
            self.assertEqual(av.key, key)
            self.assertFalse(av.loadable)

    def test_active_comes_only_from_the_live_engine_id(self) -> None:
        stale = profile("vctk_p231", is_active=True, status="prototype")
        self.assertNotEqual(pres.availability(stale, active_id=None).key, "active")
        self.assertEqual(pres.availability(stale, active_id="vctk_p231").key, "active")

    def test_readiness_text_does_not_claim_validation_before_loading(self) -> None:
        text = pres.readiness_text(READY_B, active_id=None)
        self.assertIn("sẽ được kiểm tra đầy đủ khi nạp", text)
        self.assertNotIn("thành công", text)
        self.assertIn("thành công", pres.readiness_text(READY_B, active_id="voice_b_preview"))
        self.assertIn("thu hồi", pres.readiness_text(WITHDRAWN_A))
        self.assertIn("chưa có mô hình", pres.readiness_text(PLANNED))
        self.assertIn("thiếu tệp", pres.readiness_text(MISSING))

    def test_rights_label_makes_no_claim_for_unspecified(self) -> None:
        self.assertEqual(pres.rights_label(profile("x", data_rights="unspecified")), "")
        self.assertIn("chưa xác minh", pres.rights_label(profile("x", data_rights="unverified")))


class TestCharacterCatalog(unittest.TestCase):
    def test_bundled_catalog_lists_the_checked_roster_with_its_source(self) -> None:
        catalog = pres.load_character_catalog()
        self.assertEqual(len(catalog.entries), 29)
        self.assertEqual(catalog.source_url, "https://playvalorant.com/en-us/agents/")
        self.assertEqual(catalog.checked_on, "2026-10-23")
        self.assertIn("không liên kết với Riot Games", catalog.disclaimer)
        linked = {e.catalog_id: e.linked_profile_id for e in catalog.entries if e.linked_profile_id}
        self.assertEqual(linked, {"valorant.jett": "jett", "valorant.sage": "sage"})
        self.assertEqual({e.role for e in catalog.entries}, {"duelist", "initiator", "sentinel", "controller"})

    def test_links_to_unregistered_profiles_are_rejected(self) -> None:
        catalog = pres.load_character_catalog()
        with self.assertLogs("voice_changer.gui.presentation", level="WARNING"):
            resolved, rejected = pres.resolve_catalog(catalog, [VCTK])  # jett/sage not registered here
        self.assertEqual(sorted(rejected), ["jett", "sage"])
        self.assertTrue(all(r.profile is None for r in resolved))

    def test_malformed_entries_are_skipped(self) -> None:
        data = {"roles": {"duelist": {"label_vi": "Duelist", "color": "#be185d"}},
                "entries": [{"id": "a", "name": "A", "role": "duelist", "linked_profile_id": None},
                            {"id": "a", "name": "Dup", "role": "duelist"},
                            {"id": "b", "name": "B", "role": "unknown_role"},
                            {"id": "c", "name": "C", "role": "duelist", "linked_profile_id": 42},
                            "not a dict"]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "catalog.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertLogs("voice_changer.gui.presentation", level="WARNING"):
                catalog = pres.load_character_catalog(path)
        self.assertEqual([(e.catalog_id, e.linked_profile_id) for e in catalog.entries], [("a", None), ("c", None)])

    def test_missing_catalog_file_gives_an_empty_catalog(self) -> None:
        with self.assertLogs("voice_changer.gui.presentation", level="WARNING"):
            catalog = pres.load_character_catalog(Path(tempfile.gettempdir()) / "no_such_catalog_file.json")
        self.assertEqual(catalog.entries, ())


class TestAssetsAndFallbacks(unittest.TestCase):
    def test_only_local_pngs_inside_the_assets_folder_are_accepted(self) -> None:
        for bad in ("https://example.com/jett.png", "//cdn/x.png", "C:/art/jett.png", "../../secret.png",
                    "icons/logo.svg", "", None, 3):
            self.assertIsNone(pres._local_asset(bad), bad)
        self.assertEqual(pres._local_asset("icons/app_icon_32.png"), pres.ICONS_DIR / "app_icon_32.png")

    def test_presentation_has_no_remote_or_image_avatars(self) -> None:
        for pid, entry in pres.load_voice_presentation().items():
            self.assertIsNone(entry.avatar.image, pid)
            self.assertTrue(entry.avatar.glyph, pid)

    def test_generated_icons_exist(self) -> None:
        for size in (16, 32, 48, 256):
            self.assertTrue((pres.ICONS_DIR / f"app_icon_{size}.png").is_file(), size)
        self.assertTrue((pres.ICONS_DIR / "app_icon.ico").is_file())
        self.assertEqual(Path(logo_file_for(48)).name, "app_icon_48.png")
        self.assertEqual(Path(logo_file_for(30)).name, "app_icon_32.png")

    def test_missing_icons_fall_back_to_a_text_logo(self) -> None:
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        setup_theme(root)
        with mock.patch("voice_changer.gui.views.sidebar.ICONS_DIR", Path(tempfile.gettempdir()) / "no_icons_here"):
            self.assertIsNone(logo_file_for(48))
            sidebar = Sidebar(root, on_navigate=lambda v: None)
        self.assertIsNone(sidebar.logo_image)
        self.assertEqual(sidebar.lbl_logo.cget("text"), "RT")
        self.assertEqual(sidebar.lbl_app_title.cget("text"), "RealTimeVoiceChanger")

    def test_unreadable_avatar_image_falls_back_to_the_monogram(self) -> None:
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "broken.png"
            bad.write_bytes(b"not a png")
            with self.assertLogs("voice_changer.gui.views.widgets", level="WARNING"):
                canvas = create_avatar(root, pres.AvatarSpec("JT", "#be185d", bad), 40)
        texts = [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]
        self.assertEqual(texts, ["JT"])

    def test_monogram(self) -> None:
        self.assertEqual(pres.monogram("Voice B (preview)"), "VB")
        self.assertEqual(pres.monogram("KAY/O"), "KO")
        self.assertEqual(pres.monogram("Jett"), "JE")
        self.assertEqual(pres.monogram("!!"), "?")

    def test_app_version_comes_from_project_metadata(self) -> None:
        import tomllib

        import voice_changer

        version = pres.app_version()
        self.assertTrue(version)
        self.assertNotEqual(version, "1.2.0")  # the old hardcoded sidebar text
        self.assertEqual(version, voice_changer.__version__)
        # pyproject.toml must not declare a second, independent version string.
        with open(Path(__file__).resolve().parents[2] / "pyproject.toml", "rb") as f:
            meta = tomllib.load(f)
        self.assertNotIn("version", meta["project"])
        self.assertIn("version", meta["project"]["dynamic"])
        self.assertEqual(meta["tool"]["setuptools"]["dynamic"]["version"], {"attr": "voice_changer.__version__"})


class TestRouteAndTelemetry(unittest.TestCase):
    MIC = AudioDevice(1, "USB Microphone", "MME", 1, 0, 48000.0)
    CABLE = AudioDevice(2, "CABLE Input (VB-Audio Virtual Cable)", "MME", 0, 2, 48000.0)
    SPEAKERS = AudioDevice(3, "Speakers (Realtek)", "MME", 0, 2, 48000.0)

    def test_route_text_names_the_selected_devices(self) -> None:
        r = pres.route_summary(self.MIC, self.CABLE)
        self.assertIn("USB Microphone", r.text)
        self.assertIn("CABLE Input", r.text)
        self.assertTrue(r.to_virtual_cable)
        self.assertIn("CABLE Output", r.hint)

    def test_route_to_speakers_does_not_claim_other_apps_receive_the_voice(self) -> None:
        r = pres.route_summary(self.MIC, self.SPEAKERS)
        self.assertFalse(r.to_virtual_cable)
        self.assertIn("nghe thử", r.hint)
        self.assertNotIn("CABLE Output", r.hint)
        self.assertIn("Chưa chọn", pres.route_summary(None, None).text)

    def test_telemetry_label_distinguishes_live_from_previous_session(self) -> None:
        self.assertIn("Chưa có dữ liệu", pres.telemetry_label(None))
        self.assertEqual(pres.telemetry_label({"live": True, "session_id": 4}), "Trực tiếp · phiên #4")
        self.assertIn("Phiên trước (#4)", pres.telemetry_label({"live": False, "session_id": 4}))

    def test_performance_summary_only_while_live_and_measured(self) -> None:
        base = {"rtf": 0.5, "chunks_converted": 10, "chunks_dropped": 0, "steady_underflows": 0}
        self.assertIsNone(pres.performance_summary(base))
        self.assertIsNone(pres.performance_summary(dict(base, live=False)))
        self.assertIsNone(pres.performance_summary(dict(base, live=True, chunks_converted=0)))
        self.assertIn("ổn định", pres.performance_summary(dict(base, live=True))[0])
        self.assertIn("gián đoạn", pres.performance_summary(dict(base, live=True, chunks_dropped=1))[0])

    def test_stream_parameters_come_from_audio_settings(self) -> None:
        defaults = dict(pres.stream_parameters())
        a = AudioSettings()
        self.assertEqual(defaults["Độ dài chunk"], f"{a.chunk_duration_s * 1000:.0f} ms")
        custom = dict(pres.stream_parameters(AudioSettings(chunk_duration_s=0.25)))
        self.assertEqual(custom["Độ dài chunk"], "250 ms")

    def test_summaries_for_collapsed_studio_sections(self) -> None:
        self.assertEqual(tuning_summary(None), "Chưa nạp giọng")
        self.assertEqual(tuning_summary(DEFAULTS), "Cao độ +12 · Index 0.50")
        ns = NoiseSuppressionStatus(enabled=False, mode="medium", available_modes=("light", "medium", "strong"),
                                    max_attenuation_db=-12.0, latency_ms=20.0, active=False, failed=False,
                                    last_error="")
        self.assertEqual(noise_summary(ns)[0], "Đang tắt")
        self.assertIn("Vừa", noise_summary(_replace(ns, enabled=True))[0])
        self.assertIn("Đang lọc", noise_summary(_replace(ns, enabled=True, active=True))[0])
        self.assertIn("tự ngắt", noise_summary(_replace(ns, enabled=True, failed=True))[0])


def _replace(obj, **changes):
    import dataclasses

    return dataclasses.replace(obj, **changes)


class TestDiagnosticsParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        setup_theme(self.root)

    def test_parameters_follow_the_session_or_the_defaults(self) -> None:
        diag = DiagnosticsPanel(self.root, log_path="x.log")
        self.assertEqual(diag.lbl_params_source.cget("text"), "Mặc định cho phiên tiếp theo")
        status = EngineStatus(state=EngineState.RUNNING, profile_id="p", profile_status="preview",
                              model_loaded=True, config=ConversionConfig(f0_method="rmvpe"),
                              audio=AudioSettings(chunk_duration_s=0.25), session_id=2, health_problems=(),
                              last_error=None)
        diag.update_status(status)
        self.assertEqual(diag.lbl_params_source.cget("text"), "Phiên hiện tại")
        self.assertEqual(diag._param_values["Độ dài chunk"].cget("text"), " 250 ms")
        self.assertEqual(diag._param_values["Thuật toán F0"].cget("text"), " RMVPE")
        diag.update_metrics({"live": True, "session_id": 2, "chunks_converted": 3, "algorithmic_delay_ms": 70.0})
        self.assertEqual(diag._param_values["Độ trễ ghép chunk + khử ồn"].cget("text"), " 70 ms")
        self.assertEqual(diag.lbl_telemetry_source.cget("text"), "Trực tiếp · phiên #2")

    def test_open_log_folder_reports_failure_instead_of_raising(self) -> None:
        diag = DiagnosticsPanel(self.root, log_path=str(Path(tempfile.gettempdir()) / "logs" / "backend.log"))
        with mock.patch("voice_changer.gui.views.diagnostics_panel.os.startfile", side_effect=OSError("nope"),
                        create=True), \
                mock.patch("voice_changer.gui.views.diagnostics_panel.messagebox.showwarning") as warn:
            diag._open_log_folder()
        warn.assert_called_once()


class TestLibraryCatalogAndNoNetwork(unittest.TestCase):
    """Builds the Library and the whole app with sockets disabled: nothing may be fetched remotely."""

    def setUp(self) -> None:
        def _no_network(*args, **kwargs):
            raise AssertionError("the GUI attempted a network connection")

        for target in ("socket.socket", "socket.create_connection", "urllib.request.urlopen"):
            patcher = mock.patch(target, side_effect=_no_network)
            patcher.start()
            self.addCleanup(patcher.stop)
        settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(settings_dir.cleanup)
        env = mock.patch.dict(os.environ, {"LOCALAPPDATA": settings_dir.name})
        env.start()
        self.addCleanup(env.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        setup_theme(self.root)

    def test_unlinked_catalog_entries_have_no_action_and_linked_ones_follow_the_registry(self) -> None:
        jett = profile("jett", display_name="Jett", data_rights="unverified")
        selected = []
        library = VoiceLibraryView(self.root, on_profile_selected=selected.append)
        library.pack()
        library.set_profiles([VCTK, jett, WITHDRAWN_A])
        self.assertEqual(library.rejected_catalog_links, ["sage"])  # sage is not registered in this test
        self.assertEqual(len(library._catalog_cards), 29)
        for cid, data in library._catalog_cards.items():
            linked = data["resolved"].profile
            buttons = [w for w in data["btn_action_box"].winfo_children() if isinstance(w, tk.Button)]
            if linked is None:
                self.assertEqual(data["pill_status"].cget("text"), "Chưa có preset giọng", cid)
                self.assertEqual(buttons, [], cid)
            else:
                self.assertEqual(cid, "valorant.jett")
                self.assertEqual(len(buttons), 1)
                buttons[0].invoke()
        self.assertEqual(selected, ["jett"])
        self.assertIn("29 nhân vật", library.section_catalog.lbl_summary.cget("text"))

    def test_personal_character_pictures_come_from_the_user_avatars_folder(self) -> None:
        avatars = pres.user_avatars_dir()
        self.assertEqual(avatars, Path(os.environ["LOCALAPPDATA"]) / "RealTimeVoiceChanger" / "avatars")
        self.assertIsNone(pres.character_avatar_image("valorant.jett"))  # nothing there yet
        (avatars / "valorant").mkdir(parents=True)
        picture = tk.PhotoImage(master=self.root, width=200, height=200)
        picture.put("#be185d", to=(0, 0, 200, 200))
        picture.write(str(avatars / "valorant" / "jett.png"), format="png")
        self.assertEqual(pres.character_avatar_image("valorant.jett"), avatars / "valorant" / "jett.png")
        for bad in ("valorant.sage", "../valorant.jett", "valorant/jett", "jett", "valorant.jett.png"):
            self.assertIsNone(pres.character_avatar_image(bad), bad)

        library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
        library.pack()
        library.set_profiles([VCTK, profile("jett", display_name="Jett")])
        jett_avatar = library._catalog_cards["valorant.jett"]["avatar_widget"]
        self.assertEqual(jett_avatar.avatar_spec.image, avatars / "valorant" / "jett.png")
        self.assertTrue(hasattr(jett_avatar, "image_ref"))  # drawn as a picture, not the monogram
        self.assertLessEqual(jett_avatar.image_ref.width(), int(jett_avatar.cget("width")))
        self.assertIsNone(library._catalog_cards["valorant.sage"]["avatar_widget"].avatar_spec.image)
        # The registered voice the Jett entry links to wears the same picture; unlinked voices keep monograms.
        self.assertEqual(library._rendered_cards["jett"]["avatar_widget"].avatar_spec.image,
                         avatars / "valorant" / "jett.png")
        self.assertIsNone(library._rendered_cards["vctk_p231"]["avatar_widget"].avatar_spec.image)

    def test_empty_filter_result_reserves_no_space_for_previously_shown_cards(self) -> None:
        def _build(cards_shown_first: bool) -> VoiceLibraryView:
            library = VoiceLibraryView(self.root, on_profile_selected=lambda pid: None)
            library.pack()
            if not cards_shown_first:
                library._on_filter_chip_clicked("planned")
            library.set_profiles([VCTK, READY_B, WITHDRAWN_A])  # no planned profile
            self.root.update_idletasks()
            if cards_shown_first:
                self.assertGreater(library.cards_frame.winfo_reqheight(), 100)  # cards were laid out
                library._on_filter_chip_clicked("planned")
                self.root.update_idletasks()
            return library

        after_cards = _build(cards_shown_first=True)
        self.assertEqual(after_cards.content.pack_slaves(),
                         [after_cards.lbl_profiles_title, after_cards.empty_state, after_cards.section_catalog])
        self.assertEqual(after_cards.content.winfo_reqheight(),
                         _build(cards_shown_first=False).content.winfo_reqheight())
        # Clearing the filter brings the cards back between the heading and the catalog.
        after_cards._reset_filters()
        self.assertEqual(after_cards.content.pack_slaves(),
                         [after_cards.lbl_profiles_title, after_cards.cards_frame, after_cards.section_catalog])
        self.assertEqual(len(after_cards._rendered_cards), 3)

    def test_app_starts_without_network_and_library_selection_does_not_load(self) -> None:
        service = mock.MagicMock(spec=VoiceChangerService)
        service.list_profiles.return_value = [VCTK, READY_B, WITHDRAWN_A]
        service.resolve_profile_id.return_value = None  # no startup load in this test
        service.list_devices.return_value = DeviceInventory(
            inputs=(TestRouteAndTelemetry.MIC,), outputs=(TestRouteAndTelemetry.CABLE,),
            default_input_id=1, default_output_id=2)
        service.get_status.return_value = EngineStatus(
            state=EngineState.IDLE, profile_id=None, profile_status=None, model_loaded=False, config=None,
            audio=None, session_id=0, health_problems=(), last_error=None)
        service.get_metrics.return_value = None
        service.get_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False, mode="medium", available_modes=("light", "medium", "strong"), max_attenuation_db=-12.0,
            latency_ms=20.0, active=False, failed=False, last_error="")
        with mock.patch.object(VoiceChangerApp, "_run_async", lambda app, func, *args: func(*args)), \
                mock.patch("voice_changer.gui.app.configure_logging", return_value=Path("test.log")):
            app = VoiceChangerApp(self.root, service=service)
            app._process_queue()
            app._navigate_to("library")
            app.panel_library._on_card_load_clicked("voice_b_preview")
            service.select_profile.assert_not_called()
            self.assertEqual(app._current_view, "studio")
            self.assertEqual(app.panel_controls.action.kind, "load")
            # Studio has no device selectors; the route summary names the selected devices.
            studio_widgets = _descendants(app.studio_column)
            combos = {w for w in studio_widgets if isinstance(w, ttk.Combobox)}
            self.assertEqual(combos, {app.panel_profiles.profile_combo, app.panel_noise.combo_mode})
            self.assertIn("USB Microphone", app.panel_controls.lbl_route.cget("text"))
            self.assertIn("CABLE Input", app.panel_controls.lbl_route.cget("text"))
            # Withdrawn voice A is listed in the Library but cannot be chosen.
            app.panel_library._on_card_load_clicked("voice_a_preview")
            self.assertEqual(app.panel_profiles.get_selected_profile().profile_id, "voice_b_preview")
            self.assertEqual(self.root.title(), "RealTimeVoiceChanger")

    def test_resizing_keeps_the_studio_column_bounded(self) -> None:
        service = mock.MagicMock(spec=VoiceChangerService)
        service.list_profiles.return_value = [VCTK]
        service.resolve_profile_id.return_value = None
        service.list_devices.return_value = DeviceInventory(inputs=(), outputs=(), default_input_id=None,
                                                            default_output_id=None)
        service.get_status.return_value = EngineStatus(
            state=EngineState.IDLE, profile_id=None, profile_status=None, model_loaded=False, config=None,
            audio=None, session_id=0, health_problems=(), last_error=None)
        service.get_metrics.return_value = None
        service.get_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False, mode="medium", available_modes=("light", "medium", "strong"), max_attenuation_db=-12.0,
            latency_ms=20.0, active=False, failed=False, last_error="")
        with mock.patch.object(VoiceChangerApp, "_run_async", lambda app, func, *args: func(*args)), \
                mock.patch("voice_changer.gui.app.configure_logging", return_value=Path("test.log")):
            app = VoiceChangerApp(self.root, service=service)
        app._process_queue()
        self.assertIn("Chưa chọn", app.panel_controls.lbl_route.cget("text"))  # no devices: said plainly
        canvas = app.studio_column.master
        from voice_changer.gui.dpi import px
        for width in (500, 1400):
            canvas.event_generate("<Configure>", width=width, height=600, when="now")
            window = canvas.find_all()[0]
            self.assertLessEqual(int(float(canvas.itemcget(window, "width"))), px(760))
            self.assertLessEqual(int(float(canvas.itemcget(window, "width"))), width)


def _descendants(widget: tk.Misc) -> list:
    out = []
    for child in widget.winfo_children():
        out.append(child)
        out.extend(_descendants(child))
    return out


if __name__ == "__main__":
    unittest.main()
