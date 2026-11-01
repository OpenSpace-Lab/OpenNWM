"""Small, ROS-free Isaac Sim 4.1 adapter for the original MarsLab assets.

The upstream application targets Isaac Sim 5.x. This adapter deliberately uses
4.1 World/ArticulationView/Camera APIs while retaining the authored rigid bodies,
collision geometry, mass properties, passive rocker coupling and Ackermann
geometry. Only the initial spawn is authored; subsequent poses come from PhysX.
Imports of Kit/USD stay inside functions so configuration tools need no GPU.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

LOG = logging.getLogger(__name__)


class _ArticulationBodyState:
    """Read the root rigid body without introducing a second USD transform view."""

    def __init__(self, articulation: Any):
        self.articulation = articulation

    def get_world_pose(self) -> tuple:
        position, quaternion = self.articulation.get_world_poses()
        return position[0], quaternion[0]

    def get_linear_velocity(self) -> Any:
        return self.articulation.get_linear_velocities()[0]

    def get_angular_velocity(self) -> Any:
        return self.articulation.get_angular_velocities()[0]


def boot(args: Any) -> Any:
    from omni.isaac.kit import SimulationApp

    return SimulationApp({
        "headless": True, "renderer": "RaytracedLighting", "multi_gpu": False,
        "active_gpu": args.gpu_id,
        "physics_gpu": args.physics_gpu_id if args.physics_gpu_id is not None else args.gpu_id,
        "width": args.resolution, "height": args.resolution,
        "fast_shutdown": False, "anti_aliasing": 2,
    })


def _load_reference(stage: Any, path: str, asset: str) -> None:
    import omni.kit.app
    from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading

    add_reference_to_stage(usd_path=asset, prim_path=path)
    app = omni.kit.app.get_app()
    while is_stage_loading():
        app.update()
    if not stage.GetPrimAtPath(path).IsValid():
        raise RuntimeError(f"Asset failed to load: {asset}")


def _spawn_position(stage: Any, terrain: str, spawn: dict) -> tuple[tuple, dict]:
    import numpy as np
    from pxr import Usd, UsdGeom

    visible = collision = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath(terrain), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        invisible = UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
        if "Collision" in prim.GetName() or invisible:
            if collision is None:
                collision = prim
        elif visible is None:
            visible = prim
    mesh = collision if collision is not None else visible
    if mesh is None:
        raise RuntimeError("No terrain mesh to resolve the initial spawn")
    reference = visible if visible is not None else mesh
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = bounds.ComputeWorldBound(reference).ComputeAlignedRange()
    center = [(float(box.GetMin()[i]) + float(box.GetMax()[i])) / 2 for i in (0, 1)]
    mode = spawn.get("mode", "dem_relative")
    xy = spawn.get("xy", [0, 0])
    if mode == "dem_relative":
        xy = [center[i] + float(xy[i]) for i in (0, 1)]
    elif mode == "dem_center":
        xy = center
    elif mode != "absolute":
        raise ValueError(f"Unknown spawn mode: {mode}")
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    if not len(points):
        raise RuntimeError("Terrain mesh contains no points")
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh), dtype=np.float64)
    points = points @ transform[:3, :3] + transform[3, :3]
    distances = np.sum((points[:, :2] - np.asarray(xy)) ** 2, axis=1)
    nearest = np.argpartition(distances, min(8, len(points)) - 1)[:8]
    surface = float(np.median(points[nearest, 2]))
    position = (float(xy[0]), float(xy[1]), surface + 0.1)
    return position, {"dem_center_xy_m": center, "spawn_surface_z_m": surface,
                      "spawn_dem_mesh": str(mesh.GetPath()), "spawn_mode": mode}


def _set_spawn(stage: Any, root: str, xyz: tuple, yaw: float) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(root))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    xform.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))


def _configure_authored_physics(stage: Any, chassis: str, body: str, config: dict) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    body_prim = stage.GetPrimAtPath(body)
    if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI) or not body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(f"Expected a physical articulated chassis at {body}")
    if UsdPhysics.RigidBodyAPI(body_prim).GetKinematicEnabledAttr().Get():
        raise RuntimeError("Refusing to record a kinematic rover")
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
    rigid.CreateAngularDampingAttr(float(config.get("angular_damping", 12)))
    rigid.CreateLinearDampingAttr(float(config.get("linear_damping", 0.5)))
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(body_prim)
    articulation.CreateSolverPositionIterationCountAttr(16)
    articulation.CreateSolverVelocityIterationCountAttr(4)
    wheels = config["wheels"]
    material = UsdShade.Material.Define(stage, f"{chassis}/RecordingWheelMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(float(wheels["friction_static"]))
    physics_material.CreateDynamicFrictionAttr(float(wheels["friction_dynamic"]))
    physics_material.CreateRestitutionAttr(float(wheels["restitution"]))
    for name in wheels["link_names"]:
        wheel = stage.GetPrimAtPath(f"{chassis}/{name}")
        if not wheel or not wheel.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing physical wheel {name}")
        colliders = [p for p in Usd.PrimRange(wheel, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        if not colliders:
            raise RuntimeError(f"Wheel has no enabled collision geometry: {name}")
        UsdShade.MaterialBindingAPI.Apply(wheel).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
    joints = f"{chassis}/{config.get('chassis', {}).get('joints_prim_name', 'joints')}"
    control = config["control"]
    for kind in ("drive", "steer"):
        for name in control[f"{kind}_joint_names"]:
            prim = stage.GetPrimAtPath(f"{joints}/{name}")
            if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"Missing rover revolute joint: {name}")
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(float(control["steer_stiffness"]) if kind == "steer" else 0.0)
            drive.CreateDampingAttr(float(control[f"{kind}_damping"]))
            drive.CreateMaxForceAttr(float(control[f"{kind}_max_force"]))
            drive.CreateTypeAttr(control.get("drive_type", "acceleration"))
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateTargetPositionAttr(0.0)
            if kind == "steer":
                joint = UsdPhysics.RevoluteJoint(prim)
                limit = math.degrees(float(control["max_steer_angle"]))
                joint.CreateLowerLimitAttr(-limit)
                joint.CreateUpperLimitAttr(limit)
    suspension = config["suspension"]
    for kind in ("rocker", "bogie"):
        for name in suspension[f"{kind}_joint_names"]:
            prim = stage.GetPrimAtPath(f"{joints}/{name}")
            if prim:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateDampingAttr(float(suspension[f"{kind}_damping"]))
                drive.CreateStiffnessAttr(0.0)


def _lighting(stage: Any, args: Any) -> dict:
    from pxr import Gf, UsdGeom, UsdLux

    light_types = (UsdLux.DistantLight, UsdLux.DomeLight, UsdLux.SphereLight, UsdLux.RectLight)
    if any(any(p.IsA(kind) for kind in light_types) for p in stage.Traverse()):
        return {"mode": "authored scene lighting"}
    sun = UsdLux.DistantLight.Define(stage, "/World/RecordingSun")
    sun.CreateIntensityAttr(args.sun_intensity)
    # Color and apparent diameter match MarsLab's rendering defaults. Energy
    # is fixed explicitly: this adapter does not run its atmosphere model.
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    sun.CreateAngleAttr(0.35)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(35, -40, 25))
    sky = UsdLux.DomeLight.Define(stage, "/World/RecordingSky")
    sky.CreateIntensityAttr(args.dome_intensity)
    sky.CreateColorAttr(Gf.Vec3f(0.85, 0.75, 0.60))
    return {"mode": "fixed sun and dome; no atmospheric scattering model",
            "sun_intensity": args.sun_intensity, "sun_color_rgb": [1.0, 0.95, 0.85],
            "sun_angular_diameter_deg": 0.35, "sun_rotation_xyz_deg": [35, -40, 25],
            "dome_intensity": args.dome_intensity, "dome_color_rgb": [0.85, 0.75, 0.60]}


def release_scene(world: Any, camera: Any) -> None:
    """Release sensor graphs and PhysX objects before Isaac 4.1 unloads plugins."""
    import omni.kit.app
    import omni.timeline
    from omni.isaac.core import World

    omni.timeline.get_timeline_interface().stop()
    if camera is not None:
        camera.pause()
        product_path = camera.get_render_product_path()
        # Camera.__del__ in 4.1 omits these two built-in annotators. Detaching
        # explicitly also avoids keeping the RGB OmniGraph alive during exit.
        for attribute in ("_rgb_annotator", "_fabric_time_annotator"):
            annotator = getattr(camera, attribute, None)
            if annotator is not None and product_path:
                annotator.detach([product_path])
                setattr(camera, attribute, None)
        for annotator in camera.supported_annotators:
            getattr(camera, f"remove_{annotator}_from_frame")()
        if camera._render_product is not None:
            camera._render_product.destroy()
            camera._render_product = None
        camera._stage_open_callback = None
        camera._timer_reset_callback = None
    for _ in range(3):
        omni.kit.app.get_app().update()
    if world is not None:
        world.clear()
        World.clear_instance()
    for _ in range(3):
        omni.kit.app.get_app().update()
    LOG.info("Released recording camera and physics scene before app.close")


def build_scene(args: Any, episode: dict, config: dict, rover_usd: Any) -> tuple:
    import numpy as np
    import omni.timeline
    import omni.usd
    from omni.isaac.core import World
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.sensor import Camera
    from pxr import Usd, UsdPhysics
    from marslab.robots.differential import configure_rocker_differential
    from marslab.sim.physics_scene import normalize_reference_physics, validate_world_physics
    from marslab.sim.point_instancer import normalize_external_prototypes
    from marslab.sim.spawn_clearance import resolve_spawn_clearance

    world = World(physics_dt=1 / args.physics_hz, rendering_dt=1 / args.physics_hz,
                  stage_units_in_meters=1.0, backend="numpy", device="cpu")
    physics = world.get_physics_context()
    physics.set_gravity(-abs(float(args.gravity)))
    physics.set_solver_type("TGS")
    physics.enable_gpu_dynamics(args.gpu_physics)
    physics.set_broadphase_type("GPU" if args.gpu_physics else "MBP")
    stage = omni.usd.get_context().get_stage()
    terrain, root = "/World/Terrain", config.get("prim_path", "/World/Rover")
    omni.timeline.get_timeline_interface().stop()
    _load_reference(stage, terrain, str(args.scene.resolve()))
    normalize_reference_physics(stage, terrain, physics.prim_path, args.gravity)
    normalize_external_prototypes(stage, terrain)
    terrain_colliders = sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath(terrain), Usd.TraverseInstanceProxies())
                           if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
    if not terrain_colliders:
        raise RuntimeError("Scene has no enabled physical collision geometry")
    initial, details = _spawn_position(stage, terrain, episode["spawn"])
    _load_reference(stage, root, str(rover_usd))
    # Normalization validates all scenes, so the asset must not add an independent solver.
    normalize_reference_physics(stage, root, physics.prim_path, args.gravity)
    yaw = float(episode["spawn"].get("yaw_rad", 0))
    _set_spawn(stage, root, initial, yaw)
    chassis = f"{root}/{config.get('chassis', {}).get('container_prim_name', 'Body_Chassis')}"
    body_path = f"{chassis}/{config.get('chassis', {}).get('rigid_body_prim_name', 'Body_Chassis')}"
    _configure_authored_physics(stage, chassis, body_path, config)
    configure_rocker_differential(stage, chassis)
    resolved = resolve_spawn_clearance(stage, terrain, root, initial, 0.1)
    _set_spawn(stage, root, resolved, yaw)
    validate_world_physics(stage, physics.prim_path, args.gravity)
    details.update(spawn_root_position_m=list(resolved), spawn_root_yaw_rad=yaw,
                   chassis_body_path=body_path, terrain_enabled_collider_count=terrain_colliders,
                   lighting=_lighting(stage, args), runtime="minimal Isaac 4.1 adapter; upstream MarsLab targets 5.x")
    articulation = world.scene.add(ArticulationView(prim_paths_expr=body_path, name="recording_rover",
                                                  reset_xform_properties=False))
    body = _ArticulationBodyState(articulation)
    pitch = math.radians(args.camera_pitch)
    camera = Camera(prim_path=f"{body_path}/RecordingCamera", name="recording_camera", frequency=-1,
                    resolution=(args.resolution, args.resolution), translation=np.asarray(args.camera_translation),
                    orientation=np.asarray([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
    world.reset()
    camera.initialize()
    camera.set_frequency(-1)
    aperture = camera.get_horizontal_aperture()
    camera.set_vertical_aperture(aperture)
    camera.set_focal_length(aperture / (2 * math.tan(math.radians(args.camera_fov) / 2)))
    camera.set_clipping_range(0.3, 200.0)
    details["camera_intrinsics"] = camera.get_intrinsics_matrix().tolist()
    details["articulation_dof_names"] = list(articulation.dof_names)
    LOG.info("Scene ready: spawn=%s body=%s DOFs=%s", resolved, body_path, articulation.dof_names)
    return world, articulation, body, camera, details


def configure_control(articulation: Any, config: dict) -> dict:
    import numpy as np

    names = list(articulation.dof_names)
    drive = np.asarray([names.index(name) for name in config["drive_joint_names"]], dtype=np.int32)
    steer = np.asarray([names.index(name) for name in config["steer_joint_names"]], dtype=np.int32)
    articulation.set_gains(kps=np.zeros((1, len(drive)), dtype=np.float32),
                           kds=np.full((1, len(drive)), config["drive_damping"], dtype=np.float32), joint_indices=drive)
    articulation.set_gains(kps=np.full((1, len(steer)), config["steer_stiffness"], dtype=np.float32),
                           kds=np.full((1, len(steer)), config["steer_damping"], dtype=np.float32), joint_indices=steer)
    return {"config": config, "drive_indices": drive, "steer_indices": steer,
            "drive_target": np.zeros(6, dtype=np.float32), "steer_target": np.zeros(4, dtype=np.float32)}


def apply_control(articulation: Any, state: dict, requested: tuple, dt: float) -> dict:
    import numpy as np
    from marslab.robots.rover_control import ackermann_command, constrain_ackermann_twist

    cfg = state["config"]
    v = float(np.clip(requested[0], -cfg["max_linear_velocity"], cfg["max_linear_velocity"]))
    w = float(np.clip(requested[1], -cfg["max_angular_velocity"], cfg["max_angular_velocity"]))
    v, w = constrain_ackermann_twist(v, w, cfg["wheelbase"], cfg["track_steer"],
                                    cfg["max_steer_angle"], cfg.get("steering_axle_offset", 0))
    steer, drive = ackermann_command(v, w, cfg["wheelbase"], cfg["track_steer"], cfg["track_middle"],
                                    cfg["wheel_radius"], cfg.get("steering_axle_offset", 0))
    if cfg.get("negate_steer", True):
        steer = -steer
    step = float(cfg["steer_ramp_rate"]) * dt
    state["steer_target"] += np.clip(steer - state["steer_target"], -step, step)
    measured = articulation.get_joint_positions(joint_indices=state["steer_indices"])[0]
    aligned = bool(np.max(np.abs(measured - steer)) <= cfg["steering_alignment_tolerance"])
    if not aligned:
        drive = np.zeros_like(drive)
    delta = drive - state["drive_target"]
    slowing = np.abs(drive) < np.abs(state["drive_target"])
    step = float(cfg["max_wheel_accel_rate"]) * dt * np.where(slowing, cfg["decel_multiplier"], 1.0)
    state["drive_target"] += np.clip(delta, -step, step)
    articulation.set_joint_position_targets(state["steer_target"][None, :], joint_indices=state["steer_indices"])
    articulation.set_joint_velocity_targets(state["drive_target"][None, :], joint_indices=state["drive_indices"])
    return {"limited_command_vw": [v, w], "steering_aligned": aligned,
            "wheel_velocity_targets_rad_s": state["drive_target"].tolist(),
            "steering_position_targets_rad": state["steer_target"].tolist()}


def sample_camera(world: Any, camera: Any, size: int, minimum_render_frames: int = 1,
                  render_stats: dict | None = None) -> tuple:
    """Flush rendering until RGB is stamped with the current physical state."""
    import numpy as np

    if minimum_render_frames < 1:
        raise ValueError("minimum_render_frames must be positive")
    expected = float(world.current_time)
    last = None
    for attempt in range(minimum_render_frames + 29):
        world.render()  # 4.1 disables playSimulations around app.update: no physics tick.
        if abs(float(world.current_time) - expected) > 1e-8:
            raise RuntimeError("Render-only acquisition unexpectedly advanced physics")
        frame = camera.get_current_frame()
        last = frame.get("rendering_time")
        rgba = np.asarray(frame.get("rgba"))
        if last is not None and abs(float(last) - expected) < 1e-5 and rgba.shape == (size, size, 4):
            if rgba.dtype != np.uint8:
                raise RuntimeError(f"Unexpected camera dtype: {rgba.dtype}")
            if np.any(rgba[:, :, :3]) and attempt + 1 >= minimum_render_frames:
                if render_stats is not None:
                    render_stats["render_updates"] = attempt + 1
                return rgba.copy(), float(last), int(frame["rendering_frame"])
    raise RuntimeError(f"Camera could not synchronize to physics time {expected}; last rendering_time={last}")


def warmup_camera(world: Any, body: Any, camera: Any, size: int, frames: int) -> dict:
    """Load rendering resources at the recording pose without advancing physics."""
    import numpy as np

    if frames < 1:
        raise ValueError("render warmup frames must be positive")
    started = time.monotonic()
    before_time = float(world.current_time)
    before_step = int(world.current_time_step_index)
    before_pose = [np.asarray(value).copy() for value in body.get_world_pose()]
    before_velocity = [np.asarray(body.get_linear_velocity()).copy(),
                       np.asarray(body.get_angular_velocity()).copy()]
    before_render = camera.get_current_frame().get("rendering_frame")
    stats = {}
    _, camera_time, last_render = sample_camera(world, camera, size, frames, stats)
    after_pose = [np.asarray(value).copy() for value in body.get_world_pose()]
    after_velocity = [np.asarray(body.get_linear_velocity()).copy(),
                      np.asarray(body.get_angular_velocity()).copy()]
    unchanged = (float(world.current_time) == before_time
                 and int(world.current_time_step_index) == before_step
                 and all(np.array_equal(before, after) for before, after in
                         zip(before_pose + before_velocity, after_pose + after_velocity)))
    if not unchanged:
        raise RuntimeError("Render warmup changed physics time, step, body pose or velocity")
    result = {"requested_render_frames": frames, **stats,
              "elapsed_wall_seconds": time.monotonic() - started,
              "simulation_time_before": before_time, "simulation_time_after": float(world.current_time),
              "physics_step_before": before_step, "physics_step_after": int(world.current_time_step_index),
              "body_position_before_m": before_pose[0].tolist(), "body_position_after_m": after_pose[0].tolist(),
              "body_quaternion_before_wxyz": before_pose[1].tolist(), "body_quaternion_after_wxyz": after_pose[1].tolist(),
              "physics_state_unchanged": unchanged, "camera_timestamp": camera_time,
              "rendering_frame_before": int(before_render) if before_render is not None else None,
              "rendering_frame_after": last_render}
    LOG.info("render warmup finished: %d render updates, sim_time=%.9f unchanged, body pose/velocity unchanged, wall_seconds=%.3f",
             result["render_updates"], before_time, result["elapsed_wall_seconds"])
    return result
