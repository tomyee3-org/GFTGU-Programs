"""
Binary orbit driver module.

This module performs the time integration of the
two-body problem using a simple predictor–corrector
scheme and adaptive time-step logic.
"""

from dataclasses import dataclass
from typing import List

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
) -> BinaryResult:
    """
    Integrate the binary orbit using a
    predictor–corrector with adaptive time-step.

    Parameters mirror Binary.java:
        MA, MB      : masses
        xInitA,B    : initial positions
        vInitA,B    : initial velocities (x-components)
        uInitA,B    : initial velocities (y-components)
        dt          : base time-step
        max_steps   : maximum number of steps
        eps1        : time-step accuracy threshold
        eps2        : predictor–corrector accuracy threshold
    """

    # Initial state
    state = BinaryState(
        t=0.0,
        xA=xInitA,
        yA=yInitA,
        vA=vInitA,
        uA=uInitA,
        xB=xInitB,
        yB=yInitB,
        vB=vInitB,
        uB=uInitB,
    )

    dt_work = dt

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

    def frac_change(old, new):
        if old == 0.0:
            return abs(new)
        return abs((new - old) / old)

    for step in range(max_steps):
        # Record current (accepted) state and energies. This happens
        # exactly once per accepted step -- retries below never reach
        # this point again, so rejected attempts are never logged and
        # never consume the max_steps budget.
        U, K, E = energies(
            MA, MB,
            state.xA, state.yA, state.vA, state.uA,
            state.xB, state.yB, state.vB, state.uB,
        )

        times.append(state.t)
        xA_list.append(state.xA)
        yA_list.append(state.yA)
        vA_list.append(state.vA)
        uA_list.append(state.uA)
        xB_list.append(state.xB)
        yB_list.append(state.yB)
        vB_list.append(state.vB)
        uB_list.append(state.uB)
        U_list.append(U)
        K_list.append(K)
        E_list.append(E)

        # Accelerations at the current, accepted state
        axA, ayA, axB, ayB = accelerations(
            MA, MB,
            state.xA, state.yA,
            state.xB, state.yB,
        )

        # Definite (if provisional) assignment for the values the
        # accepted step will commit below. They are always overwritten
        # inside the loop before the loop's `break`, but assigning them
        # here up front -- rather than only inside the while/for nesting
        # below -- gives static analysis an unconditional assignment to
        # find, instead of one reachable only through a continue/break
        # path across two nested loops.
        xA_new, yA_new = state.xA, state.yA
        xB_new, yB_new = state.xB, state.yB
        vA_corr, uA_corr = state.vA, state.uA
        vB_corr, uB_corr = state.vB, state.uB

        # --- Adaptive time-step / predictor-corrector for the NEXT step ---
        # A rejected attempt here only shrinks dt_work and loops back to
        # retry the SAME step; it does not touch the output lists and
        # does not advance `step`, mirroring the `dt1 /= 2; j--;` retry
        # in the original Binary.java.
        while True:
            # Predictor step: simple Euler
            xA_pred = state.xA + state.vA * dt_work
            yA_pred = state.yA + state.uA * dt_work
            vA_pred = state.vA + axA * dt_work
            uA_pred = state.uA + ayA * dt_work

            xB_pred = state.xB + state.vB * dt_work
            yB_pred = state.yB + state.uB * dt_work
            vB_pred = state.vB + axB * dt_work
            uB_pred = state.uB + ayB * dt_work

            # Accelerations at predicted state
            axA_pred, ayA_pred, axB_pred, ayB_pred = accelerations(
                MA, MB,
                xA_pred, yA_pred,
                xB_pred, yB_pred,
            )

            # Time-step accuracy check (eps1), done against the cheap
            # predictor -- BEFORE paying for the corrector iterations --
            acc_change = abs(axA_pred - axA) + abs(ayA_pred - ayA)
            acc_ref = abs(axA) + abs(ayA)
            if acc_ref > 0.0 and acc_change > eps1 * acc_ref:
                dt_work *= 0.5
                continue  # retry the SAME step with a smaller dt_work

            # Corrector: average accelerations and velocities, iterating
            # until fractional change < eps2. Accelerations are
            # re-evaluated at the refined position each pass so the
            # corrector actually corrects, instead of repeating the same
            # predictor-based estimate.
            vA_corr, uA_corr = vA_pred, uA_pred
            vB_corr, uB_corr = vB_pred, uB_pred
            axA_c, ayA_c, axB_c, ayB_c = axA_pred, ayA_pred, axB_pred, ayB_pred
            xA_new, yA_new = xA_pred, yA_pred
            xB_new, yB_new = xB_pred, yB_pred

            for _ in range(10):  # modest iteration cap
                vA_new = state.vA + 0.5 * (axA + axA_c) * dt_work
                uA_new = state.uA + 0.5 * (ayA + ayA_c) * dt_work
                vB_new = state.vB + 0.5 * (axB + axB_c) * dt_work
                uB_new = state.uB + 0.5 * (ayB + ayB_c) * dt_work

                max_change = max(
                    frac_change(vA_corr, vA_new),
                    frac_change(uA_corr, uA_new),
                    frac_change(vB_corr, vB_new),
                    frac_change(uB_corr, uB_new),
                )

                vA_corr, uA_corr, vB_corr, uB_corr = vA_new, uA_new, vB_new, uB_new

                # Position update using the corrected (averaged) velocities
                xA_new = state.xA + 0.5 * (state.vA + vA_corr) * dt_work
                yA_new = state.yA + 0.5 * (state.uA + uA_corr) * dt_work
                xB_new = state.xB + 0.5 * (state.vB + vB_corr) * dt_work
                yB_new = state.yB + 0.5 * (state.uB + uB_corr) * dt_work

                if max_change < eps2:
                    break

                # Re-evaluate accelerations at the refined position for
                # the next corrector pass.
                axA_c, ayA_c, axB_c, ayB_c = accelerations(
                    MA, MB, xA_new, yA_new, xB_new, yB_new,
                )

            break  # step accepted; leave the retry loop

        # Commit the accepted step
        state = BinaryState(
            t=state.t + dt_work,
            xA=xA_new,
            yA=yA_new,
            vA=vA_corr,
            uA=uA_corr,
            xB=xB_new,
            yB=yB_new,
            vB=vB_corr,
            uB=uB_corr,
        )

        # Let the step size grow back towards the user's requested dt
        # now that we know the last step was well-behaved.
        dt_work = min(dt_work * 1.1, dt)

    return BinaryResult(
        times=times,
        xA=xA_list,
        yA=yA_list,
        vA=vA_list,
        uA=uA_list,
        xB=xB_list,
        yB=yB_list,
        vB=vB_list,
        uB=uB_list,
        U=U_list,
        K=K_list,
        E=E_list,
    )