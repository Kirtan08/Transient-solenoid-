"""Everything that drives the solenoid transient analysis: FEMM geometry
construction, the magnetostatic solve, and the coupled electrical/mechanical
time-stepping loop.

FEMM uses millimetres for geometry and SI units for electrical and
mechanical quantities. The armature position is reported as travel toward
the stopper.

All user-tunable parameters live in `config.py` -- nothing here should need
to change to try a different design. Run this file directly to build the
model, inspect the geometry in FEMM, then run the transient:

    python simulation.py

To inspect geometry only, without running the transient, call
`build_model()` from a REPL, or comment out the transient section in
`main()` below. Note that `build_model()`/`solve_femm()` called this way,
without `main()` having called `start_new_run()` first, write to the flat,
non-timestamped default paths in the project folder (see `PATHS` below).

Each full run through `main()` writes its outputs -- the FEMM model, results
CSV, saved frames, and videos -- into their own results/<timestamp>/ folder
(via `start_new_run()`), so nothing from a previous run gets overwritten.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, List, Optional, Tuple

import cv2
import femm
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG, SimulationConfig

PROJECT_DIR: Final = Path(__file__).resolve().parent


@dataclass
class RunPaths:
    """Every file/folder one run writes to. Built fresh by `start_new_run()`
    so each run's outputs land in their own results/<timestamp>/ folder
    instead of overwriting the previous run's."""
    root: Path
    model: Path
    results_csv: Path
    frames_dir: Path
    video: Path
    airgap_frames_dir: Path
    airgap_video: Path
    comparison_video: Path
    comparison_plot_only_video: Path

    @classmethod
    def under(cls, root: Path) -> "RunPaths":
        return cls(
            root=root,
            model=root / "solenoid_v1.fem",
            results_csv=root / "transient_results.csv",
            frames_dir=root / "flux_frames",
            video=root / "flux_animation.mp4",
            airgap_frames_dir=root / "airgap_frames",
            airgap_video=root / "airgap_animation.mp4",
            comparison_video=root / "comparison_animation.mp4",
            comparison_plot_only_video=root / "comparison_plot_only_animation.mp4",
        )


# Flat, non-timestamped paths in the project folder -- the default until
# start_new_run() is called, so build_model()/solve_femm() etc. still work
# standalone (e.g. from a REPL) without requiring a full run first.
PATHS: RunPaths = RunPaths.under(PROJECT_DIR)


def start_new_run() -> RunPaths:
    """Point every output (model file, CSV, frames, videos) at a fresh
    results/<timestamp>/ folder under the project directory. Call this once,
    before build_model(), at the start of a full run."""
    global PATHS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_DIR / "results" / timestamp
    root.mkdir(parents=True, exist_ok=True)
    PATHS = RunPaths.under(root)
    return PATHS


CIRCUIT_NAME: Final = "coil"
ARMATURE_GROUP: Final = 1
COIL_GROUP: Final = 2
INNER_REGION_GROUP: Final = 3  # tags the near-field air block so it's a distinct,
                                # selectable block from the coarse far-field air

STEEL_MATERIAL: Final = "1010 Steel"  # FEMM materials-library name (matlib.dat)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def draw_rectangle(r1: float, z1: float, r2: float, z2: float, *, group: int = 0) -> None:
    """Draw a closed rectangular region in FEMM's r-z plane.

    `mi_setblockprop(..., group=...)` on a block label only tags that label
    point, not the boundary. A group needs to include the nodes/segments too
    for `mi_selectgroup` + `mi_movetranslate` to actually move the shape, so
    when `group` is given here the freshly drawn boundary is selected and
    tagged with it via `mi_selectrectangle(..., 4)` + `mi_setgroup`.
    """
    femm.mi_drawline(r1, z1, r2, z1)
    femm.mi_drawline(r2, z1, r2, z2)
    femm.mi_drawline(r2, z2, r1, z2)
    femm.mi_drawline(r1, z2, r1, z1)
    if group:
        femm.mi_selectrectangle(r1, z1, r2, z2, 4)
        femm.mi_setgroup(group)
        femm.mi_clearselected()


def draw_rectangle_by_size(
    position_r: float,
    position_z: float,
    width: float,
    height: float,
    *,
    group: int = 0,
) -> None:
    """Draw a rectangle from its lower-left position, width, and height."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    draw_rectangle(position_r, position_z,
                   position_r + width, position_z + height, group=group)


def draw_circle(center_r: float, center_z: float, diameter: float) -> None:
    """Draw a closed circle from its center position and diameter."""
    if diameter <= 0:
        raise ValueError("diameter must be positive")

    radius = diameter / 2
    left_r = center_r - radius
    right_r = center_r + radius
    femm.mi_addnode(left_r, center_z)
    femm.mi_addnode(right_r, center_z)
    femm.mi_addarc(right_r, center_z, left_r, center_z, 180, 1)
    femm.mi_addarc(left_r, center_z, right_r, center_z, 180, 1)


def add_label(
    r: float,
    z: float,
    material: str,
    mesh_size: float,
    *,
    group: int = 0,
    circuit: str = "<none>",
    turns: int = 0,
) -> None:
    femm.mi_addblocklabel(r, z)
    femm.mi_selectlabel(r, z)
    femm.mi_setblockprop(material, 0, mesh_size, circuit, 0, group, turns)
    femm.mi_clearselected()


def build_model(config: SimulationConfig = CONFIG) -> None:
    """Create and save the initial FEMM model at zero armature travel.

    Geometry is drawn in the r-z half-plane with the origin (0, 0) at the
    centerline (r = 0) and the bottom face of the stationary yoke (z = 0).
    """
    config.validate()
    inner_radius = config.coil_inner_radius
    outer_radius = config.coil_outer_radius
    yoke_radius = config.yoke_outer_radius
    top_flange = config.yoke_flange_thickness
    bottom_flange = config.yoke_height - config.yoke_flange_thickness
    armature_radius = config.armature_diameter / 2
    armature_bottom = config.armature_bottom_z
    armature_top = config.armature_top_z

    femm.openfemm()
    femm.newdocument(0)
    femm.mi_probdef(0, "millimeters", "axi", 1e-8, 0, 30)
    femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    femm.mi_addmaterial("SWG 38", 1, 1, 0, 0, 58, 0, 0, 0, 3, 0, 0, 1, 0.1524)
    femm.mi_getmaterial(STEEL_MATERIAL)  # pulls the full nonlinear BH curve from FEMM's library
    femm.mi_addcircprop(CIRCUIT_NAME, 0.0, 1)

    draw_rectangle(0, 0, yoke_radius, top_flange)
    draw_rectangle(0, top_flange, armature_radius, config.stopper_height)
    draw_rectangle(0, armature_bottom, armature_radius, armature_top, group=ARMATURE_GROUP)
    draw_rectangle(inner_radius, top_flange, outer_radius, bottom_flange)
    draw_rectangle(outer_radius, top_flange, yoke_radius, bottom_flange)
    # Inner bound is inner_radius (not armature_radius) so the top flange
    # keeps the same radial clearance off the armature as the coil bore
    # does, instead of touching the armature where it passes through.
    draw_rectangle(inner_radius, bottom_flange, yoke_radius, config.yoke_height)

    # A static rectangle around the whole device -- reusing actuator_zoom_box,
    # the same box used to frame video frames, rather than redefining "the
    # region around the device" a second time -- splits the surrounding air
    # into a near-field region (meshed densely: it includes the working air
    # gap) and a far-field region out to the ABC boundary (meshed coarsely,
    # since it barely affects the solution but would otherwise dominate the
    # element count). Fixed for the whole run, same reasoning as
    # actuator_zoom_box: travel only ever shrinks the armature within this
    # box, so it never needs to move or be redrawn.
    inner_r0, inner_z0, inner_r1, inner_z1 = actuator_zoom_box(config)
    draw_rectangle(inner_r0, inner_z0, inner_r1, inner_z1)

    add_label(armature_radius / 2, (armature_bottom + armature_top) / 2,
              STEEL_MATERIAL, config.mesh_size_armature, group=ARMATURE_GROUP)
    add_label(armature_radius / 2, (top_flange + config.stopper_height) / 2,
              STEEL_MATERIAL, config.mesh_size_stopper)
    add_label((inner_radius + outer_radius) / 2, (top_flange + bottom_flange) / 2,
              "SWG 38", config.mesh_size_coil, group=COIL_GROUP,
              circuit=CIRCUIT_NAME, turns=config.turns)
    add_label((outer_radius + yoke_radius) / 2, (top_flange + bottom_flange) / 2,
              STEEL_MATERIAL, config.mesh_size_yoke)
    add_label(yoke_radius / 2, top_flange / 2, STEEL_MATERIAL, config.mesh_size_yoke)
    add_label(yoke_radius / 2, (bottom_flange + config.yoke_height) / 2,
              STEEL_MATERIAL, config.mesh_size_yoke)
    # With the top flange no longer touching the armature, the armature's
    # radial clearance annulus (armature_radius..inner_radius) runs
    # uninterrupted from the bottom flange up past the top flange, merging
    # with the void beside/above the yoke can (including the notch beside a
    # protruding armature, if armature_top_z > yoke_height) into one
    # connected region, still just one label -- now bounded above by the
    # inner rectangle, so it gets the dense (not coarse) mesh size. The
    # label point sits halfway between the yoke's outer radius and the
    # rectangle's edge, so it lands inside the rectangle regardless of how
    # `clearance` or the rectangle's margin are configured. Tagged with its
    # own group (distinct from the coarse outer-air block below, even though
    # both share the "Air" material) so it is its own selectable block in
    # FEMM with its own mesh size, not just a differently-numbered label on
    # the same block.
    add_label((yoke_radius + inner_r1) / 2, config.yoke_height / 2,
              "Air", config.mesh_size_inner_air, group=INNER_REGION_GROUP)
    # Everything from the inner rectangle out to the ABC boundary is a
    # second, separate connected region -- one label, coarse mesh. Halfway
    # between the rectangle's edge and the boundary radius, on the z = 0
    # plane, is always outside the rectangle (r > inner_r1) and always
    # inside the semicircular ABC boundary, regardless of the model's size.
    add_label((inner_r1 + config.boundary_radius) / 2, 0.0,
              "Air", config.mesh_size_outer_air)

    femm.mi_makeABC(7, config.boundary_radius, 0, 0, 0)
    femm.mi_saveas(str(PATHS.model))
    femm.mi_zoomnatural()


# ---------------------------------------------------------------------------
# Magnetostatic solve
# ---------------------------------------------------------------------------

def actuator_zoom_box(config: SimulationConfig) -> Tuple[float, float, float, float]:
    """Bounding box (r0, z0, r1, z1) framing just the actuator -- not the
    large circular ABC air region around it -- for video frame capture.

    A fixed box (computed once from the at-rest geometry) is used for every
    frame so the camera does not jump around as the armature moves; since
    travel only ever shrinks the armature's z-extent from its at-rest
    maximum, this box safely contains the actuator at every step.
    """
    margin = 0.1
    r0 = 0.0
    r1 = config.yoke_outer_radius * (1 + margin)
    z0 = -margin * config.model_height
    z1 = config.model_height * (1 + margin)
    return r0, z0, r1, z1


def air_gap_zoom_box(config: SimulationConfig) -> Tuple[float, float, float, float]:
    """Bounding box (r0, z0, r1, z1) framing the traveling air gap between
    the armature and the stopper, with enough margin to keep both pole
    faces in view -- for a close-up video of the gap closing.

    Fixed for the whole run (computed from the at-rest, i.e. widest, gap),
    same reasoning as `actuator_zoom_box`: travel only ever shrinks the
    gap, so this box contains it at every step.
    """
    margin_z = config.armature_diameter
    r0 = 0.0
    r1 = config.coil_outer_radius * 1.1
    z0 = config.stopper_height - margin_z
    z1 = config.armature_bottom_z + margin_z
    return r0, z0, r1, z1


FrameView = Tuple[Path, Tuple[float, float, float, float]]


def solve_femm(
    position_mm: float,
    current_a: float,
    previous_position_mm: float,
    frame_views: Optional[List[FrameView]] = None,
    density_max: float = 2.0,
) -> Tuple[float, float]:
    """Solve one magnetostatic state and return force and flux linkage.

    If `frame_views` is given, a flux-density plot of this solved state is
    saved for each (frame_path, zoom_box) pair in it -- zoomed to that box
    -- before any block is selected, so the block selection highlight in
    `mo_groupselectblock` below never leaks into the saved images.

    The color scale runs from 0 to `density_max` tesla. A literal (0, 0)
    range renders as one flat, invisible color rather than autoscaling, so
    a real upper limit is required to actually see the density plot (the
    flux/equipotential lines are a separate overlay FEMM always draws, and
    show up regardless of this range).
    """
    femm.mi_modifycircprop(CIRCUIT_NAME, 1, float(current_a))
    displacement = -(position_mm - previous_position_mm)
    if abs(displacement) > 1e-14:
        femm.mi_selectgroup(ARMATURE_GROUP)
        femm.mi_movetranslate(0, displacement)
        femm.mi_clearselected()

    femm.mi_saveas(str(PATHS.model))
    femm.mi_analyze(0)
    femm.mi_loadsolution()

    for frame_path, zoom_box in frame_views or []:
        femm.mo_zoom(*zoom_box)
        femm.mo_showdensityplot(1, 0, density_max, 0, "bmag")
        femm.mo_savebitmap(str(frame_path))

    femm.mo_groupselectblock(ARMATURE_GROUP)
    # Verified against a live solve: for this axisymmetric model, block
    # integral 19 returns the real (nonzero) axial pull force, and 20 comes
    # back exactly 0. Do not "fix" this to 20 without re-verifying against
    # a live FEMM solve first.
    force_z = float(np.real(femm.mo_blockintegral(19)))
    femm.mo_clearblock()
    circuit_data = femm.mo_getcircuitproperties(CIRCUIT_NAME)
    femm.mo_close()
    return -force_z, float(np.real(circuit_data[2]))


# ---------------------------------------------------------------------------
# Live dashboard: the air-gap flux view and performance curves, updated on
# screen step by step as the transient actually runs.
# ---------------------------------------------------------------------------

class LiveDashboard:
    """One matplotlib window, updated in place while the transient runs: the
    air-gap flux-density frame -- where the armature's actual motion happens
    -- on the left, and four performance-curve panels (position, velocity,
    current, force; built up point by point) on the right.
    """

    def __init__(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(11, 8))
        gs = self.fig.add_gridspec(4, 2, width_ratios=[1, 1.3])

        self.ax_airgap = self.fig.add_subplot(gs[:, 0])
        self.ax_airgap.set_title("Air gap (flux density)", fontsize=10)
        self.ax_airgap.set_xticks([])
        self.ax_airgap.set_yticks([])
        self.im_airgap = None

        ylabels = ["Position (mm)", "Velocity (m/s)", "Current (A)", "Force (N)"]
        self.curve_axes = [self.fig.add_subplot(gs[i, 1]) for i in range(4)]
        self.curve_lines = [ax.plot([], [])[0] for ax in self.curve_axes]
        for ax, ylabel in zip(self.curve_axes, ylabels):
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True)
        self.curve_axes[-1].set_xlabel("Time (ms)", fontsize=10)

        self.fig.suptitle("Running transient simulation...", fontsize=13, fontweight="bold")
        self._flush()

    def update_images(self, airgap_path: Path) -> None:
        airgap_img = cv2.cvtColor(cv2.imread(str(airgap_path)), cv2.COLOR_BGR2RGB)
        if self.im_airgap is None:
            self.im_airgap = self.ax_airgap.imshow(airgap_img)
        else:
            self.im_airgap.set_data(airgap_img)
        self._flush()

    def update_curves(self, x, *series: np.ndarray) -> None:
        """`series`: position_mm, velocity, current, force -- each already
        sliced to the steps completed so far, so the panels grow point by
        point as the transient progresses."""
        for ax, line, values in zip(self.curve_axes, self.curve_lines, series):
            line.set_data(x, values)
            ax.relim()
            ax.autoscale_view()
        self._flush()

    def _flush(self) -> None:
        # Re-run tight_layout on every update: tick-label width changes as
        # data grows (e.g. more digits), and a layout computed once up front
        # doesn't reserve margin for that -- which is what made the y-axis
        # labels hard to read/clipped against the image panel.
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        plt.close(self.fig)
        plt.ioff()


# ---------------------------------------------------------------------------
# Transient (coupled electrical/mechanical time-stepping)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransientResult:
    time: np.ndarray
    position_mm: np.ndarray
    velocity: np.ndarray
    current: np.ndarray
    force: np.ndarray               # N; magnetic pull from the coil (positive, toward the stopper)
    restoring_force: np.ndarray     # N; spring + preload + damping, always opposing `force`
    back_emf: np.ndarray            # V; motional EMF from the plunger's motion
    flux_linkage: np.ndarray


def run_transient(
    config: SimulationConfig = CONFIG, dashboard: Optional[LiveDashboard] = None
) -> TransientResult:
    """Time-step the coupled electrical/mechanical pull-in transient.

    Electrical equation (per step): V = I*R + (dλ/dx)*v + (dλ/dI)*(dI/dt).
    The incremental inductance dλ/dI is found by perturbing the coil current
    by `config.current_perturbation` and re-solving FEMM at the same armature
    position (no geometry change). While the plunger is actually moving
    (velocity != 0), the motional back-EMF term dλ/dx is likewise found by
    perturbing *position* by `config.position_perturbation` at the same
    current and re-solving -- a true partial derivative at fixed current,
    rather than a finite difference across the real trajectory (which would
    conflate the effect of current changing between steps with the effect of
    position changing, and would give no signal at all for the first moving
    step). This is what lets the back-EMF show up as the expected dip/plateau
    in the current trace right as the plunger accelerates.

    Mechanical equation (per step): m*a = F(x, I) - restoring_force, where
    restoring_force = preload + c*v + k*x is the spring preload, damping,
    and spring stiffness terms combined -- all newtons, and all opposing the
    coil's magnetic pull force `F` by construction (subtracted, never added).
    Position is clamped to [0, seated_position_mm] to model the armature
    resting against its initial stop (no pickup yet) or fully seated at the
    stopper (minus `config.min_air_gap`, so FEMM never meshes a literal
    zero-width gap).

    When `config.save_flux_video` is set, two flux-density frames are saved
    for the state actually reached at each step (not for the intermediate
    perturbation solves used to estimate dLambda/dI and dLambda/dx): one
    framing the whole actuator (`PATHS.frames_dir` / `PATHS.video`) and one
    zoomed in on the armature-stopper air gap (`PATHS.airgap_frames_dir` /
    `PATHS.airgap_video`). Both are assembled into their videos once the
    loop finishes.

    If `dashboard` is given (and `config.save_flux_video` is set, since the
    dashboard's image panel needs that same air-gap frame), the air-gap
    frame plus the position/velocity/current/force curves (built up step by
    step) are pushed to it live, so the transient can be watched as it runs.
    """
    config.validate()
    n_steps = int(round(config.simulation_time / config.time_step)) + 1
    time = np.linspace(0.0, config.simulation_time, n_steps)

    position_mm = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    current = np.zeros(n_steps)
    force = np.zeros(n_steps)
    restoring_force = np.zeros(n_steps)
    back_emf = np.zeros(n_steps)
    flux_linkage = np.zeros(n_steps)

    actuator_frame_paths: List[Path] = []
    airgap_frame_paths: List[Path] = []
    if config.save_flux_video:
        PATHS.frames_dir.mkdir(exist_ok=True)
        PATHS.airgap_frames_dir.mkdir(exist_ok=True)
    actuator_box = actuator_zoom_box(config)
    airgap_box = air_gap_zoom_box(config)

    def frame_views_for(step: int) -> List[FrameView]:
        if not config.save_flux_video:
            return []
        actuator_path = PATHS.frames_dir / f"frame_{step:05d}.bmp"
        airgap_path = PATHS.airgap_frames_dir / f"frame_{step:05d}.bmp"
        actuator_frame_paths.append(actuator_path)
        airgap_frame_paths.append(airgap_path)
        return [(actuator_path, actuator_box), (airgap_path, airgap_box)]

    def restoring_force_at(position_mm_value: float, velocity_value: float) -> float:
        """Spring preload + damping + spring stiffness, in newtons -- always
        opposes the coil's magnetic pull force `force`, never adds to it."""
        return (
            config.spring_preload
            + config.damping * velocity_value
            + config.spring_stiffness * (position_mm_value * 1e-3)
        )

    def update_dashboard(step: int) -> None:
        if dashboard is None:
            return
        if config.save_flux_video:
            dashboard.update_images(airgap_frame_paths[-1])
        dashboard.update_curves(
            time[: step + 1] * 1000,
            position_mm[: step + 1],
            velocity[: step + 1],
            current[: step + 1],
            force[: step + 1],
        )

    force[0], flux_linkage[0] = solve_femm(
        position_mm[0], current[0], position_mm[0],
        frame_views_for(0), config.flux_density_plot_max,
    )
    restoring_force[0] = restoring_force_at(position_mm[0], velocity[0])
    update_dashboard(0)
    # Tracks wherever the armature actually sits in the live FEMM model,
    # which can briefly differ from position_mm[i-1] right after a position-
    # perturbation solve (see below) -- every solve_femm() call must be told
    # the true current position so it translates the geometry correctly.
    last_geometry_position_mm = position_mm[0]

    for i in range(1, n_steps):
        dt = time[i] - time[i - 1]

        _, flux_current_perturbed = solve_femm(
            position_mm[i - 1],
            current[i - 1] + config.current_perturbation,
            last_geometry_position_mm,
        )
        last_geometry_position_mm = position_mm[i - 1]
        incremental_inductance = (
            (flux_current_perturbed - flux_linkage[i - 1]) / config.current_perturbation
        )
        if incremental_inductance <= 0:
            raise RuntimeError(f"Non-positive incremental inductance at step {i}")

        if velocity[i - 1] != 0.0:
            delta_mm = config.position_perturbation
            perturbed_position_mm = position_mm[i - 1] + delta_mm
            if perturbed_position_mm > config.seated_position_mm:
                delta_mm = -delta_mm
                perturbed_position_mm = position_mm[i - 1] + delta_mm
            _, flux_position_perturbed = solve_femm(
                perturbed_position_mm, current[i - 1], last_geometry_position_mm,
            )
            last_geometry_position_mm = perturbed_position_mm
            dlambda_dx = (
                (flux_position_perturbed - flux_linkage[i - 1]) / (delta_mm * 1e-3)
            )
        else:
            dlambda_dx = 0.0
        back_emf[i] = dlambda_dx * velocity[i - 1]

        d_current_dt = (
            config.input_voltage - current[i - 1] * config.coil_resistance - back_emf[i]
        ) / incremental_inductance
        current[i] = current[i - 1] + d_current_dt * dt

        net_force = force[i - 1] - restoring_force[i - 1]
        acceleration = net_force / config.moving_mass
        velocity[i] = velocity[i - 1] + acceleration * dt
        position_mm[i] = position_mm[i - 1] + velocity[i] * dt * 1e3

        if position_mm[i] <= 0.0:
            position_mm[i] = 0.0
            velocity[i] = 0.0
        elif position_mm[i] >= config.seated_position_mm:
            position_mm[i] = config.seated_position_mm
            velocity[i] = 0.0

        force[i], flux_linkage[i] = solve_femm(
            position_mm[i], current[i], last_geometry_position_mm,
            frame_views_for(i), config.flux_density_plot_max,
        )
        last_geometry_position_mm = position_mm[i]
        restoring_force[i] = restoring_force_at(position_mm[i], velocity[i])
        update_dashboard(i)

    if config.save_flux_video:
        build_flux_video(actuator_frame_paths, PATHS.video, config.flux_video_fps)
        build_flux_video(airgap_frame_paths, PATHS.airgap_video, config.flux_video_fps)

    return TransientResult(
        time, position_mm, velocity, current, force, restoring_force, back_emf, flux_linkage
    )


def build_flux_video(frame_paths: List[Path], video_path: Path, fps: float) -> None:
    """Stitch saved flux-density frames (in order) into a single video file."""
    if not frame_paths:
        return

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read saved frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read saved frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def _compose_side_by_side_video(
    fig,
    markers: List,
    marker_x_values: np.ndarray,
    plunger_frame_paths: List[Path],
    video_path: Path,
    fps: float,
) -> None:
    """Shared compositing loop: redraw `fig` with each of `markers` moved to
    `marker_x_values[i]`, and write it side-by-side with `plunger_frame_paths[i]`
    to `video_path`. `fig` is closed once the video is written.
    """
    canvas = fig.canvas
    canvas.draw()
    plot_width, plot_height = canvas.get_width_height()

    first_plunger_frame = cv2.imread(str(plunger_frame_paths[0]))
    if first_plunger_frame is None:
        raise RuntimeError(f"Could not read saved frame: {plunger_frame_paths[0]}")
    target_height = first_plunger_frame.shape[0]
    plot_target_width = int(plot_width * target_height / plot_height)

    combined_width = first_plunger_frame.shape[1] + plot_target_width
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (combined_width, target_height))
    try:
        for i, plunger_path in enumerate(plunger_frame_paths):
            for marker in markers:
                marker.set_xdata([marker_x_values[i], marker_x_values[i]])
            canvas.draw()
            plot_rgba = np.asarray(canvas.buffer_rgba())
            plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)
            plot_bgr = cv2.resize(plot_bgr, (plot_target_width, target_height))

            plunger_frame = cv2.imread(str(plunger_path))
            if plunger_frame is None:
                raise RuntimeError(f"Could not read saved frame: {plunger_path}")
            if plunger_frame.shape[0] != target_height:
                new_width = int(plunger_frame.shape[1] * target_height / plunger_frame.shape[0])
                plunger_frame = cv2.resize(plunger_frame, (new_width, target_height))

            combined = cv2.hconcat([plunger_frame, plot_bgr])
            if combined.shape[1] != combined_width:
                combined = cv2.resize(combined, (combined_width, target_height))
            writer.write(combined)
    finally:
        writer.release()
        plt.close(fig)


def _write_plot_only_video(
    fig,
    markers: List,
    marker_x_values: np.ndarray,
    n_frames: int,
    video_path: Path,
    fps: float,
) -> None:
    """Write `fig`'s frames -- with each of `markers` swept across
    `marker_x_values` -- directly to `video_path`, with no side-by-side
    plunger frame. `fig` is closed once the video is written.
    """
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    try:
        for i in range(n_frames):
            for marker in markers:
                marker.set_xdata([marker_x_values[i], marker_x_values[i]])
            canvas.draw()
            plot_rgba = np.asarray(canvas.buffer_rgba())
            plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)
            writer.write(plot_bgr)
    finally:
        writer.release()
        plt.close(fig)


def _build_transient_comparison_figure(result: TransientResult):
    """Build the 4-panel position/velocity/current/force figure with a
    movable time marker on each panel. Shared by `build_comparison_video`
    (plunger frame alongside) and `build_comparison_plot_video` (plot only).
    Returns (fig, markers, time_ms).
    """
    time_ms = result.time * 1000

    fig, axes = plt.subplots(4, 1, figsize=(6, 9))

    def add_series(ax, y_values, ylabel, femm_label: str = "FEMM") -> None:
        ax.plot(time_ms, y_values, label=femm_label)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    add_series(axes[0], result.position_mm, "Position (mm)")
    add_series(axes[1], result.velocity, "Velocity (m/s)")
    add_series(axes[2], result.current, "Current (A)")
    add_series(axes[3], result.force, "Force (N)", femm_label="Coil force")
    axes[3].plot(time_ms, -result.restoring_force, label="Restoring force (neg)")
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_xlabel("Time (ms)")

    markers = [ax.axvline(time_ms[0], color="red", linewidth=1) for ax in axes]
    for ax in axes:
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig, markers, time_ms


def build_comparison_video(
    result: TransientResult,
    plunger_frame_paths: List[Path],
    video_path: Path,
    fps: float,
) -> None:
    """Build one video with the plunger's movement (the saved air-gap
    flux-density frames) beside an animated position/velocity/current/force
    plot, with a vertical marker sweeping across all four panels in sync
    with the frame.
    """
    if not plunger_frame_paths:
        return

    fig, markers, time_ms = _build_transient_comparison_figure(result)
    _compose_side_by_side_video(fig, markers, time_ms, plunger_frame_paths, video_path, fps)


def build_comparison_plot_video(
    result: TransientResult,
    video_path: Path,
    fps: float,
) -> None:
    """Animate just the position/velocity/current/force plot -- no FEMM
    plunger/flux frame alongside -- into its own video, with a vertical
    marker sweeping across all four panels over time.
    """
    fig, markers, time_ms = _build_transient_comparison_figure(result)
    _write_plot_only_video(fig, markers, time_ms, len(time_ms), video_path, fps)


def plot_results(result: TransientResult) -> None:
    """Display position, velocity, current, and force against time."""
    time_ms = result.time * 1000

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

    axes[0].plot(time_ms, result.position_mm)
    axes[0].set_ylabel("Position (mm)")

    axes[1].plot(time_ms, result.velocity)
    axes[1].set_ylabel("Velocity (m/s)")

    axes[2].plot(time_ms, result.current)
    axes[2].set_ylabel("Current (A)")

    # Restoring force is plotted negated: it always opposes the coil's pull
    # by construction (subtracted in the equation of motion), so this shows
    # the two curves on opposite sides of zero rather than both positive.
    axes[3].plot(time_ms, result.force, label="Coil force")
    axes[3].plot(time_ms, -result.restoring_force, label="Restoring force (neg)")
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_ylabel("Force (N)")
    axes[3].set_xlabel("Time (ms)")

    for ax in axes:
        ax.grid(True)
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def save_results_csv(result: TransientResult, path: Optional[Path] = None) -> None:
    # A Path default would be captured once at function-definition time and
    # go stale after start_new_run() points PATHS elsewhere, so resolve the
    # current PATHS.results_csv at call time instead.
    path = path if path is not None else PATHS.results_csv
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_s", "position_mm", "velocity_m_s", "current_A",
            "force_N", "restoring_force_N", "back_emf_V", "flux_linkage_Wb",
        ])
        writer.writerows(
            zip(
                result.time,
                result.position_mm,
                result.velocity,
                result.current,
                result.force,
                result.restoring_force,
                result.back_emf,
                result.flux_linkage,
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Point every output at a fresh results/<timestamp>/ folder, build the
    model, let the user inspect it, then run the transient -- with a live
    dashboard (air-gap flux view and performance curves) shown throughout."""
    start_new_run()
    print(f"Run outputs will be saved under: {PATHS.root}")

    build_model()
    print(f"FEMM model saved to: {PATHS.model}")
    print(f"Static current reference: {CONFIG.peak_current:.3f} A")
    input("Inspect the geometry in FEMM, then press Enter to run the transient...")

    dashboard = LiveDashboard()
    result = run_transient(dashboard=dashboard)
    dashboard.close()

    save_results_csv(result)
    print(f"Transient results saved to: {PATHS.results_csv}")

    if CONFIG.save_flux_video:
        print(f"Flux-density video saved to: {PATHS.video}")
        print(f"Air-gap close-up video saved to: {PATHS.airgap_video}")
        airgap_frame_paths = [
            PATHS.airgap_frames_dir / f"frame_{i:05d}.bmp" for i in range(len(result.time))
        ]
        build_comparison_video(
            result, airgap_frame_paths, PATHS.comparison_video, CONFIG.flux_video_fps
        )
        print(f"Plunger-movement + plot video saved to: {PATHS.comparison_video}")

    # No FEMM frames needed for this one -- just the animated results plot.
    build_comparison_plot_video(
        result, PATHS.comparison_plot_only_video, CONFIG.flux_video_fps
    )
    print(f"Plot-only results video saved to: {PATHS.comparison_plot_only_video}")

    seated = result.position_mm >= CONFIG.seated_position_mm
    if seated.any():
        seated_time_ms = result.time[np.argmax(seated)] * 1000
        print(f"Armature seated at t = {seated_time_ms:.3f} ms")
    else:
        print("Armature did not reach full stroke within simulation_time")

    plot_results(result)


if __name__ == "__main__":
    main()
