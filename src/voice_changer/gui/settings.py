"""Local settings persistence for RealTimeVoiceChanger GUI.

Stores preferences in %LOCALAPPDATA%\\RealTimeVoiceChanger\\gui_settings.json.
Remembers last used profile and audio endpoints by (name, host_api) pair,
as device IDs are PortAudio indices that can change across reboots/plugs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from voice_changer.audio.devices import AudioDevice
from voice_changer.service import DeviceInventory

logger = logging.getLogger(__name__)


def _get_settings_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base_dir = Path(local_app_data) / "RealTimeVoiceChanger"
    else:
        base_dir = Path.home() / ".real_time_voice_changer"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "gui_settings.json"


@dataclass
class GuiSettings:
    """Persistent user preferences for the desktop GUI."""

    profile_id: Optional[str] = None
    input_device_name: Optional[str] = None
    input_device_host_api: Optional[str] = None
    output_device_name: Optional[str] = None
    output_device_host_api: Optional[str] = None

    pitch_shift: Optional[float] = None
    index_rate: Optional[float] = None
    filter_radius: Optional[int] = None
    protect_rate: Optional[float] = None

    @classmethod
    def load(cls) -> GuiSettings:
        path = _get_settings_path()
        if not path.is_file():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                profile_id=data.get("profile_id"),
                input_device_name=data.get("input_device_name"),
                input_device_host_api=data.get("input_device_host_api"),
                output_device_name=data.get("output_device_name"),
                output_device_host_api=data.get("output_device_host_api"),
                pitch_shift=data.get("pitch_shift"),
                index_rate=data.get("index_rate"),
                filter_radius=data.get("filter_radius"),
                protect_rate=data.get("protect_rate"),
            )
        except Exception as err:
            logger.warning("Could not read GUI settings from %s: %s", path, err)
            return cls()

    def save(self) -> None:
        path = _get_settings_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        except Exception as err:
            logger.warning("Could not write GUI settings to %s: %s", path, err)

    def resolve_input_device_id(self, inventory: DeviceInventory) -> Optional[int]:
        """Match stored (name, host_api) against current inputs, fallback to default."""
        if self.input_device_name and self.input_device_host_api:
            for dev in inventory.inputs:
                if dev.name == self.input_device_name and dev.host_api == self.input_device_host_api:
                    return dev.device_id
        return inventory.default_input_id

    def resolve_output_device_id(self, inventory: DeviceInventory) -> Optional[int]:
        """Match stored (name, host_api) against current outputs, fallback to default."""
        if self.output_device_name and self.output_device_host_api:
            for dev in inventory.outputs:
                if dev.name == self.output_device_name and dev.host_api == self.output_device_host_api:
                    return dev.device_id
        return inventory.default_output_id

    def missing_saved_input(self, inventory: DeviceInventory) -> Optional[str]:
        """Name of the saved input device if it is not in `inventory` (unplugged), else None."""
        if not (self.input_device_name and self.input_device_host_api):
            return None
        found = any(d.name == self.input_device_name and d.host_api == self.input_device_host_api
                    for d in inventory.inputs)
        return None if found else self.input_device_name

    def missing_saved_output(self, inventory: DeviceInventory) -> Optional[str]:
        """Name of the saved output device if it is not in `inventory` (unplugged), else None."""
        if not (self.output_device_name and self.output_device_host_api):
            return None
        found = any(d.name == self.output_device_name and d.host_api == self.output_device_host_api
                    for d in inventory.outputs)
        return None if found else self.output_device_name

    def update_input(self, device: Optional[AudioDevice]) -> None:
        if device is not None:
            self.input_device_name = device.name
            self.input_device_host_api = device.host_api
        else:
            self.input_device_name = None
            self.input_device_host_api = None

    def update_output(self, device: Optional[AudioDevice]) -> None:
        if device is not None:
            self.output_device_name = device.name
            self.output_device_host_api = device.host_api
        else:
            self.output_device_name = None
            self.output_device_host_api = None
