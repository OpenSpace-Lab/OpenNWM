"""Allowlisted scene sources. Private paths are never accepted from HTTP."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEBSITE = REPO / "website"
DEFAULT_RUN = os.environ.get("NWM_NAV15_STAGE2_RUN", "/file_system/nas/algorithm/dujun.nie/nwm/opennwm/runs/imported-nav15-stage2")
DEFAULT_OUTPUT = "/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/redesign_20260929/playground"
DATA_ROOT = Path(os.environ.get("NWM_DATA_ROOT", "/file_system/nas/algorithm/dujun.nie/nwm/data"))

SCENES = [
    dict(id="garden-recon", title="Garden path", titleZh="Garden path", dataset="RECON", domain="ID", folder="recon", trajectory="jackal_2019-10-24-13-35-07_2_r02", frame=91, waypointSpacing=.25, description="A ground robot moving through a sunlit garden.", descriptionZh="A ground robot moving through a sunlit garden."),
    dict(id="hall-huron", title="Coral hallway", titleZh="Coral hallway", dataset="HuRoN", domain="ID", folder="sacson", trajectory="Feb-27-2023-soda3-intloss_00000025_0", frame=51, waypointSpacing=.255, description="A wheeled robot approaching a colorful hallway.", descriptionZh="A wheeled robot approaching a colorful hallway."),
    dict(id="indoor-huron", title="Glass corridor", titleZh="Glass corridor", dataset="HuRoN", domain="ID", folder="sacson", trajectory="Feb-06-2023-bww8-intloss_00000010_3", frame=51, waypointSpacing=.255, description="A wheeled robot's indoor view.", descriptionZh="A wheeled robot's indoor view."),
    dict(id="outdoor-tartan", title="Gravel trail", titleZh="Gravel trail", dataset="TartanDrive", domain="ID", folder="tartan", trajectory="20210903_heightmaps_2_20210903_36_1", frame=235, waypointSpacing=.72, description="An off-road view across uneven terrain.", descriptionZh="An off-road view across uneven terrain."),
    dict(id="office-go2", title="Robot dog · office", titleZh="Robot dog · office", dataset="Office-Go2", domain="OOD", folder="unitree_go2", trajectory="episode_20260909T122302_105f3d57__seg01", frame=88, waypointSpacing=.07610250112606785, description="A held-out quadruped camera view.", descriptionZh="A held-out quadruped camera view."),
    dict(id="moon-yutu2", title="Moon · Yutu-2", titleZh="Moon · Yutu-2", dataset="Planetary Rover", domain="OOD", folder="planetary_rover", trajectory="moon_yutu2_pcam_color_0000", frame=1, waypointSpacing=None, description="Real Chang'e-4 / Yutu-2 lunar imagery.", descriptionZh="Real Chang'e-4 / Yutu-2 lunar imagery.", planetary=True),
    dict(id="mars-zhurong", title="Mars · Zhurong", titleZh="Mars · Zhurong", dataset="Planetary Rover", domain="OOD", folder="planetary_rover", trajectory="mars_zhurong_natecam_color_0000", frame=3, waypointSpacing=None, description="Real Tianwen-1 / Zhurong Martian imagery.", descriptionZh="Real Tianwen-1 / Zhurong Martian imagery.", planetary=True),
    dict(id="moon-lusnar", title="Moon · simulation", titleZh="Moon · simulation", dataset="LuSNAR", domain="OOD", trajectory="Moon_1/image0/color", frame="1693274724086398208", waypointSpacing=None, description="A simulated lunar environment from LuSNAR.", descriptionZh="A simulated lunar environment from LuSNAR.", simulated=True,
         sourcePath="/file_system/nas/algorithm/dujun.nie/datasets/LuSNAR/Moon_1/image0/color/1693274724086398208.png"),
]
for scene in SCENES:
    if scene.get("planetary") or scene.get("simulated"):
        scene["qualityNote"] = "Out-of-domain stress test: the current checkpoint does not preserve the space environment and hallucinates terrestrial structures. These are unmodified model predictions."
        scene["qualityNoteZh"] = "Out-of-domain stress test: the current checkpoint does not preserve the space environment and hallucinates terrestrial structures. These are unmodified model predictions."
SCENE_MAP = {scene["id"]: scene for scene in SCENES}
# Keep the historical sources available for reproduction and old local jobs;
# only the visually reviewed scenes below appear in the interactive demo.
PLAYGROUND_SCENE_IDS = ("outdoor-tartan", "garden-recon", "hall-huron")
DEFAULT_SCENE = PLAYGROUND_SCENE_IDS[0]


def source_path(scene):
    if "sourcePath" in scene:
        return Path(scene["sourcePath"])
    return DATA_ROOT / scene["folder"] / scene["trajectory"] / f"{scene['frame']}.jpg"
