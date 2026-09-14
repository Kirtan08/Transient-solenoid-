"""User-facing controls for the solenoid transient analysis.

Everything you would want to tweak between runs -- geometry, materials,
electrical drive, mechanical load, and simulation timing -- lives in
`SimulationConfig` below. Nothing else in the project should define a
tunable parameter; other modules only *derive* values (radii, boundary
size, etc.) from what is set here.

To try a different design, edit the field values on `CONFIG` at the
bottom of this file (or construct your own `SimulationConfig(...)` and
pass it into `build_model()` / `run_transient()` instead of the default).
"""

from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    # --- Geometry (mm) ---------------------------------------------------
    armature_length: float = 142.0
    clearance: float = 2.0              # radial gap between armature and coil bore
    coil_thickness: float = 44.0        # radial thickness of the coil winding
    yoke_height: float = 274.0          # overall axial height of the steel yoke can
    armature_diameter: float = 40.0
    yoke_flange_thickness: float = 12.0  # thickness of the top/bottom yoke flanges
    stopper_height: float = 132.0       # axial position of the stopper face
    stroke: float = 1.1                 # armature travel distance, gap to seated
    outer_diameter: float = 140.0       # outer diameter of the yoke can
    min_air_gap: float = 0.01           # mm; residual gap enforced at full seat so
                                         # FEMM never meshes a literal zero-width gap

    # --- Electrical --------------------------------------------------------
    turns: int = 3300
    input_voltage: float = 200.0
    coil_resistance: float = 200.0

    # --- Mechanical (spring/load) -------------------------------------------
    # All forces/loads here are in SI units (newtons), and always oppose the
    # coil's magnetic pull force -- see the restoring-force note in
    # simulation.py's run_transient().
    spring_stiffness: float = 1500.0    # N/m
    spring_preload: float = 5         # N
    damping: float = 0               # N*s/m
    moving_mass: float = 6           # kg

    # --- Simulation / solver settings ---------------------------------------
    time_step: float = 1e-3
    simulation_time: float = 0.040
    current_perturbation: float = 1e-3   # A; used to estimate dLambda/dI numerically
    position_perturbation: float = 0.01  # mm; used to estimate dLambda/dx numerically
                                          # (the back-EMF term) while the plunger moves

    # --- Mesh sizes per region (mm) -------------------------------------------
    # One size per part rather than a single global value, so the near-field
    # (which includes the working air gap and drives solve accuracy) can stay
    # fine while everything far away is coarsened for solve speed.
    mesh_size_armature: float = 5.0
    mesh_size_stopper: float = 5.0
    mesh_size_coil: float = 10.0
    mesh_size_yoke: float = 5.0          # outer can wall + both flanges
    mesh_size_inner_air: float = 5.0     # dense: gap, clearance annulus, near-field air
    mesh_size_outer_air: float = 25.0    # coarse: far-field air out to the ABC boundary

    # --- Flux-line video output ----------------------------------------------
    save_flux_video: bool = True        # save a flux-density frame at each solved step
    flux_video_fps: float = 10.0        # playback frame rate of the assembled video
    flux_density_plot_max: float = 2.0  # tesla; upper color-scale limit for |B| frames (lower fixed at 0)

    def validate(self) -> None:
        positive = {
            "turns": self.turns,
            "coil_resistance": self.coil_resistance,
            "spring_stiffness": self.spring_stiffness,
            "moving_mass": self.moving_mass,
            "time_step": self.time_step,
            "simulation_time": self.simulation_time,
            "current_perturbation": self.current_perturbation,
            "position_perturbation": self.position_perturbation,
            "stroke": self.stroke,
            "armature_length": self.armature_length,
            "min_air_gap": self.min_air_gap,
            "mesh_size_armature": self.mesh_size_armature,
            "mesh_size_stopper": self.mesh_size_stopper,
            "mesh_size_coil": self.mesh_size_coil,
            "mesh_size_yoke": self.mesh_size_yoke,
            "mesh_size_inner_air": self.mesh_size_inner_air,
            "mesh_size_outer_air": self.mesh_size_outer_air,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These parameters must be positive: {', '.join(invalid)}")
        if self.stopper_height <= self.yoke_flange_thickness:
            raise ValueError("stopper_height must be greater than yoke_flange_thickness")
        if self.min_air_gap >= self.stroke:
            raise ValueError("min_air_gap must be less than stroke")

    # --- Derived quantities (do not set these directly) ---------------------
    @property
    def peak_current(self) -> float:
        return self.input_voltage / self.coil_resistance

    @property
    def coil_inner_radius(self) -> float:
        return self.armature_diameter / 2 + self.clearance

    @property
    def coil_outer_radius(self) -> float:
        return self.coil_inner_radius + self.coil_thickness

    @property
    def yoke_outer_radius(self) -> float:
        return self.outer_diameter / 2

    @property
    def armature_bottom_z(self) -> float:
        """z of the armature's bottom face at zero travel, set by stroke."""
        return self.stopper_height + self.stroke

    @property
    def armature_top_z(self) -> float:
        return self.armature_bottom_z + self.armature_length

    @property
    def seated_position_mm(self) -> float:
        """Travel value used for 'fully seated', slightly less than `stroke`
        so the armature never touches the stopper with zero remaining air
        gap -- a literal zero-width gap is a degenerate FEMM mesh region and
        produces bad (near-zero or NaN-adjacent) force/inductance results,
        which is why the armature was observed dropping back off the
        stopper shortly after appearing to seat."""
        return self.stroke - self.min_air_gap

    @property
    def model_height(self) -> float:
        """Tallest z-extent of the drawn geometry (the armature may protrude
        past yoke_height as a rod), used to size the outer ABC boundary."""
        return max(self.yoke_height, self.armature_top_z)

    @property
    def boundary_radius(self) -> float:
        return 2 * np.hypot(self.yoke_outer_radius, self.model_height)


# The configuration used by default across the project. Edit the field
# values above, or build an alternate SimulationConfig(...) and pass it
# explicitly to build_model()/run_transient() to try a variant without
# touching this default.
CONFIG: Final = SimulationConfig()
