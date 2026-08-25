"""
Time integration for the Newtonian Binary program.

The integrator uses an Euler predictor followed by an iterated
trapezoidal corrector. A working timestep is reduced when the
acceleration changes too much during the predictor or when the
corrector fails to converge.
"""

from dataclasses import dataclass
from math import atan2, hypot, isfinite, pi
from numbers import Real
from typing import List

import physics_binary as phys
from physics_binary import BinaryState, accelerations, energies


@dataclass
class BinaryResult:
    times: List[float]
    xA: List[float]
    yA: List[float]
    vA: List[float]
    uA: List[float]
    xB: List[float]
    yB: List[float]
    vB: List[float]
    uB: List[float]
    U: List[float]
    K: List[float]
    E: List[float]
    completed_orbit: bool
    accepted_steps: int
    model_version: str = phys.MODEL_VERSION
    build_id: str = phys.BUILD_ID


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _finite_real(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def _halve_timestep(dt_work: float, dt_min: float) -> float:
    """Halve a trial timestep, but refuse numerically meaningless collapse."""
    new_dt = 0.5 * dt_work
    if new_dt < dt_min or new_dt == 0.0 or new_dt == dt_work:
        raise RuntimeError(
            "The required timestep has fallen below the numerical safety limit. "
            "This usually indicates a near-collision or extremely rapid motion; "
            "the calculation cannot continue reliably."
        )
    return new_dt


def _vector_relative_change(old_x, old_y, new_x, new_y) -> float:
    """Dimensionless relative change of a 2D vector."""
    change = hypot(new_x - old_x, new_y - old_y)
    scale = max(hypot(old_x, old_y), hypot(new_x, new_y))
    if scale == 0.0:
        return 0.0
    return change / scale


def _signed_angle_increment(x0, y0, x1, y1) -> float:
    """Signed angle from vector (x0,y0) to (x1,y1), in [-pi, pi]."""
    return atan2(x0 * y1 - y0 * x1, x0 * x1 + y0 * y1)


def integrate_binary(
    MA: float,
    MB: float,
    xInitA: float,
    yInitA: float,
    vInitA: float,
    uInitA: float,
    xInitB: float,
    yInitB: float,
    vInitB: float,
    uInitB: float,
    dt: float,
    max_steps: int,
    eps1: float,
    eps2: float,
    stop_after_one_orbit: bool = True,
) -> BinaryResult:
    """
    Integrate the binary orbit.

    max_steps is a safety ceiling on accepted integration steps.
    If stop_after_one_orbit is True, a bound orbit normally stops after
    the relative position vector has accumulated one full revolution.
    """
    for name, value in [
        ("MA", MA), ("MB", MB),
        ("xInitA", xInitA), ("yInitA", yInitA),
        ("vInitA", vInitA), ("uInitA", uInitA),
        ("xInitB", xInitB), ("yInitB", yInitB),
        ("vInitB", vInitB), ("uInitB", uInitB),
        ("dt", dt), ("eps1", eps1), ("eps2", eps2),
    ]:
        _finite_real(name, value)

    if MA <= 0.0 or MB <= 0.0:
        raise ValueError("MA and MB must both be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    _positive_int("max_steps", max_steps)
    if not (0.0 < eps1 < 1.0):
        raise ValueError("eps1 must satisfy 0 < eps1 < 1.")
    if not (0.0 < eps2 < 1.0):
        raise ValueError("eps2 must satisfy 0 < eps2 < 1.")
    if not isinstance(stop_after_one_orbit, bool):
        raise ValueError("stop_after_one_orbit must be True or False.")

    # Validates the initial separation.
    accelerations(MA, MB, xInitA, yInitA, xInitB, yInitB)

    state = BinaryState(
        t=0.0,
        xA=xInitA, yA=yInitA, vA=vInitA, uA=uInitA,
        xB=xInitB, yB=yInitB, vB=vInitB, uB=uInitB,
    )

    times: List[float] = []
    xA_list: List[float] = []
    yA_list: List[float] = []
    vA_list: List[float] = []
    uA_list: List[float] = []
    xB_list: List[float] = []
    yB_list: List[float] = []
    vB_list: List[float] = []
    uB_list: List[float] = []
    U_list: List[float] = []
    K_list: List[float] = []
    E_list: List[float] = []

    def record(s: BinaryState) -> None:
        U, K, E = energies(
            MA, MB,
            s.xA, s.yA, s.vA, s.uA,
            s.xB, s.yB, s.vB, s.uB,
        )
        times.append(s.t)
        xA_list.append(s.xA)
        yA_list.append(s.yA)
        vA_list.append(s.vA)
        uA_list.append(s.uA)
        xB_list.append(s.xB)
        yB_list.append(s.yB)
        vB_list.append(s.vB)
        uB_list.append(s.uB)
        U_list.append(U)
        K_list.append(K)
        E_list.append(E)

    record(state)

    dt_work = dt
    # Prevent repeated adaptive halving from collapsing to a zero-time step.
    dt_min = dt * 1.0e-12
    max_corrector_iterations = 10
    max_retries_per_step = 60
    accepted_steps = 0
    completed_orbit = False
    accumulated_angle = 0.0

    rel_x_old = state.xA - state.xB
    rel_y_old = state.yA - state.yB

    while accepted_steps < max_steps:
        axA, ayA, axB, ayB = accelerations(
            MA, MB, state.xA, state.yA, state.xB, state.yB
        )

        accepted = False

        for _retry in range(max_retries_per_step):
            xA_pred = state.xA + state.vA * dt_work
            yA_pred = state.yA + state.uA * dt_work
            vA_pred = state.vA + axA * dt_work
            uA_pred = state.uA + ayA * dt_work

            xB_pred = state.xB + state.vB * dt_work
            yB_pred = state.yB + state.uB * dt_work
            vB_pred = state.vB + axB * dt_work
            uB_pred = state.uB + ayB * dt_work

            axA_pred, ayA_pred, axB_pred, ayB_pred = accelerations(
                MA, MB, xA_pred, yA_pred, xB_pred, yB_pred
            )

            acc_change = _vector_relative_change(
                axA, ayA, axA_pred, ayA_pred
            )
            if acc_change > eps1:
                dt_work = _halve_timestep(dt_work, dt_min)
                continue

            vA_corr, uA_corr = vA_pred, uA_pred
            vB_corr, uB_corr = vB_pred, uB_pred
            axA_c, ayA_c = axA_pred, ayA_pred
            axB_c, ayB_c = axB_pred, ayB_pred

            converged = False

            for _ in range(max_corrector_iterations):
                vA_new = state.vA + 0.5 * (axA + axA_c) * dt_work
                uA_new = state.uA + 0.5 * (ayA + ayA_c) * dt_work
                vB_new = state.vB + 0.5 * (axB + axB_c) * dt_work
                uB_new = state.uB + 0.5 * (ayB + ayB_c) * dt_work

                change_A = _vector_relative_change(
                    vA_corr, uA_corr, vA_new, uA_new
                )
                change_B = _vector_relative_change(
                    vB_corr, uB_corr, vB_new, uB_new
                )

                vA_corr, uA_corr = vA_new, uA_new
                vB_corr, uB_corr = vB_new, uB_new

                xA_new = state.xA + 0.5 * (state.vA + vA_corr) * dt_work
                yA_new = state.yA + 0.5 * (state.uA + uA_corr) * dt_work
                xB_new = state.xB + 0.5 * (state.vB + vB_corr) * dt_work
                yB_new = state.yB + 0.5 * (state.uB + uB_corr) * dt_work

                if max(change_A, change_B) < eps2:
                    converged = True
                    break

                axA_c, ayA_c, axB_c, ayB_c = accelerations(
                    MA, MB, xA_new, yA_new, xB_new, yB_new
                )

            if not converged:
                dt_work = _halve_timestep(dt_work, dt_min)
                continue

            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "The integrator could not find a converged timestep after "
                f"{max_retries_per_step} retries. Try less extreme initial "
                "conditions or a smaller dt."
            )

        new_time = state.t + dt_work
        if new_time == state.t:
            raise RuntimeError(
                "The timestep is too small to advance time numerically; "
                "the calculation cannot continue reliably."
            )

        state = BinaryState(
            t=new_time,
            xA=xA_new, yA=yA_new, vA=vA_corr, uA=uA_corr,
            xB=xB_new, yB=yB_new, vB=vB_corr, uB=uB_corr,
        )
        accepted_steps += 1
        record(state)

        rel_x_new = state.xA - state.xB
        rel_y_new = state.yA - state.yB
        accumulated_angle += _signed_angle_increment(
            rel_x_old, rel_y_old, rel_x_new, rel_y_new
        )
        rel_x_old, rel_y_old = rel_x_new, rel_y_new

        if stop_after_one_orbit and abs(accumulated_angle) >= 2.0 * pi:
            completed_orbit = True
            break

        dt_work = min(dt_work * 1.1, dt)

    return BinaryResult(
        times=times,
        xA=xA_list, yA=yA_list, vA=vA_list, uA=uA_list,
        xB=xB_list, yB=yB_list, vB=vB_list, uB=uB_list,
        U=U_list, K=K_list, E=E_list,
        completed_orbit=completed_orbit,
        accepted_steps=accepted_steps,
        model_version=phys.MODEL_VERSION,
        build_id=phys.BUILD_ID,
    )
