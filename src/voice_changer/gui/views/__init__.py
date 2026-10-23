"""GUI View components."""

from voice_changer.gui.views.audio_settings_view import AudioSettingsView
from voice_changer.gui.views.controls_panel import ControlsPanel
from voice_changer.gui.views.devices_panel import DevicesPanel
from voice_changer.gui.views.diagnostics_panel import DiagnosticsPanel
from voice_changer.gui.views.library_view import VoiceLibraryView
from voice_changer.gui.views.noise_panel import NoiseSuppressionPanel
from voice_changer.gui.views.profiles_panel import ProfilesPanel
from voice_changer.gui.views.sidebar import Sidebar
from voice_changer.gui.views.tuning_panel import TuningPanel

__all__ = [
    "AudioSettingsView",
    "ControlsPanel",
    "DevicesPanel",
    "DiagnosticsPanel",
    "NoiseSuppressionPanel",
    "ProfilesPanel",
    "Sidebar",
    "TuningPanel",
    "VoiceLibraryView",
]
