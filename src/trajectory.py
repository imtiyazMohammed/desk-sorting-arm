"""
Trapezoidal velocity trajectory generator with multi-joint synchronization.

Why we need this
----------------
Sending step-changes to servos causes:
  1. Mechanical shock (gear stripping, especially DS3218 shoulder joint).
  2. Overshoot and oscillation from PID controllers in the servo horn.
  3. Voltage sag as multiple servos draw stall current simultaneously.

The solution is to interpolate each joint from its current angle to its target
along a smooth velocity profile.

Trapezoidal profile
-------------------
A trajectory of duration T from angle q0 to q1 with max velocity v_max and
max acceleration a_max has three phases:

    phase 1: accelerate from 0 to v_peak over t_acc = v_peak / a_max
    phase 2: cruise at v_peak for t_cruise
    phase 3: decelerate from v_peak to 0 over t_acc

If the move is short enough that v_max is never reached, the profile
degenerates to a triangle (no cruise phase). The peak velocity in that case
is v_peak = sqrt(|q1 - q0| * a_max).

Synchronization
---------------
For multi-joint moves we want ALL joints to start and finish at the same
instant. We compute each joint's minimum-time profile independently, pick the
maximum, then time-scale the faster joints to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .geometry import ArmGeometry, DEFAULT_ARM


# =========================================================================
# Single-joint profile
# =========================================================================


@dataclass(frozen=True)
class TrapezoidalProfile:
    """
    Parameters for a single-joint trapezoidal or triangular velocity profile.

    All times in seconds; positions in radians; velocities in rad/s;
    accelerations in rad/s^2. `direction` is +1 or -1 encoding the sign of
    the movement.
    """

    q_start: float
    q_end: float
    v_peak: float          # actual peak velocity used (<=v_max)
    a_used: float          # acceleration used (rad/s^2)
    t_acc: float           # duration of accel phase (and decel phase)
    t_cruise: float        # duration of cruise phase (>=0)
    total_time: float
    direction: int

    def sample(self, t: float) -> float:
        """
        Return joint angle at time t (seconds) since profile start.
        Values outside [0, total_time] are clamped to endpoints.
        """
        if t <= 0.0:
            return self.q_start
        if t >= self.total_time:
            return self.q_end
        # Zero-distance profile: joint isn't moving, stay put throughout.
        if self.v_peak == 0.0 or (self.t_acc == 0.0 and self.t_cruise == 0.0):
            return self.q_start

        d = 0.0
        if t < self.t_acc:
            # Accelerating: q(t) = 0.5 * a * t^2
            d = 0.5 * self.a_used * t * t
        elif t < self.t_acc + self.t_cruise:
            # Cruising: full accel distance + v_peak * (t - t_acc)
            d_acc = 0.5 * self.a_used * self.t_acc * self.t_acc
            d = d_acc + self.v_peak * (t - self.t_acc)
        else:
            # Decelerating
            d_acc = 0.5 * self.a_used * self.t_acc * self.t_acc
            d_cruise = self.v_peak * self.t_cruise
            dt = t - self.t_acc - self.t_cruise
            d = d_acc + d_cruise + self.v_peak * dt - 0.5 * self.a_used * dt * dt

        return self.q_start + self.direction * d

    def sample_velocity(self, t: float) -> float:
        """Signed instantaneous velocity in rad/s."""
        if t <= 0.0 or t >= self.total_time:
            return 0.0
        if t < self.t_acc:
            v = self.a_used * t
        elif t < self.t_acc + self.t_cruise:
            v = self.v_peak
        else:
            dt = t - self.t_acc - self.t_cruise
            v = self.v_peak - self.a_used * dt
        return self.direction * v


def _min_time_profile(
    q_start: float,
    q_end: float,
    v_max: float,
    a_max: float,
) -> TrapezoidalProfile:
    """Compute the minimum-time trapezoidal profile respecting v_max, a_max."""
    if v_max <= 0.0 or a_max <= 0.0:
        raise ValueError("v_max and a_max must be positive")

    delta = q_end - q_start
    direction = 1 if delta >= 0 else -1
    dist = abs(delta)

    if dist < 1e-12:
        return TrapezoidalProfile(
            q_start=q_start, q_end=q_end,
            v_peak=0.0, a_used=a_max,
            t_acc=0.0, t_cruise=0.0, total_time=0.0,
            direction=direction,
        )

    # Distance covered while accelerating from 0 to v_max
    d_to_vmax = v_max * v_max / (2.0 * a_max)

    if 2.0 * d_to_vmax >= dist:
        # Triangular profile: never reach v_max
        v_peak = float(np.sqrt(dist * a_max))
        t_acc = v_peak / a_max
        t_cruise = 0.0
    else:
        # Trapezoidal profile
        v_peak = v_max
        t_acc = v_max / a_max
        d_cruise = dist - 2.0 * d_to_vmax
        t_cruise = d_cruise / v_max

    total_time = 2.0 * t_acc + t_cruise
    return TrapezoidalProfile(
        q_start=q_start, q_end=q_end,
        v_peak=v_peak, a_used=a_max,
        t_acc=t_acc, t_cruise=t_cruise, total_time=total_time,
        direction=direction,
    )


def _rescale_profile(
    q_start: float,
    q_end: float,
    target_time: float,
    a_max: float,
) -> TrapezoidalProfile:
    """
    Build a trapezoidal profile of a specific total duration (>= min-time).

    Given a fixed T we solve for the (v_peak, t_acc, t_cruise) that:
      * Cover distance = |q_end - q_start|
      * Sum to duration T
      * Respect a_max

    Symmetric profile with equal accel/decel time. Two unknowns
    (t_acc, v_peak) with two equations:
        (a) 2 * t_acc + t_cruise = T
        (b) t_acc * v_peak + t_cruise * v_peak = dist       [area under curve]
    plus the accel constraint v_peak = a_used * t_acc, where we solve for
    the required a_used <= a_max.

    Combining (a) and v_peak = a_used * t_acc:
        area = t_acc * v_peak + (T - 2*t_acc) * v_peak
             = v_peak * (T - t_acc)
             = a_used * t_acc * (T - t_acc)  = dist
    We pick t_acc = T/3 (common heuristic giving equal thirds), then:
        a_used = dist / (t_acc * (T - t_acc))
    If a_used > a_max we fall back to the minimum-time profile.
    """
    delta = q_end - q_start
    direction = 1 if delta >= 0 else -1
    dist = abs(delta)

    if dist < 1e-12 or target_time <= 0.0:
        return TrapezoidalProfile(
            q_start=q_start, q_end=q_end,
            v_peak=0.0, a_used=a_max,
            t_acc=0.0, t_cruise=0.0, total_time=max(target_time, 0.0),
            direction=direction,
        )

    t_acc = target_time / 3.0
    a_used = dist / (t_acc * (target_time - t_acc))
    if a_used > a_max:
        # Cannot slow down to target_time without exceeding a_max — should not
        # happen if target_time was chosen as max(min_times), but guard anyway.
        return _min_time_profile(q_start, q_end, float("inf"), a_max)

    v_peak = a_used * t_acc
    t_cruise = target_time - 2.0 * t_acc
    return TrapezoidalProfile(
        q_start=q_start, q_end=q_end,
        v_peak=v_peak, a_used=a_used,
        t_acc=t_acc, t_cruise=t_cruise, total_time=target_time,
        direction=direction,
    )


# =========================================================================
# Multi-joint synchronized trajectory
# =========================================================================


@dataclass(frozen=True)
class SynchronizedTrajectory:
    """A set of per-joint trapezoidal profiles that all finish at total_time."""

    profiles: List[TrapezoidalProfile]
    total_time: float

    def sample(self, t: float) -> np.ndarray:
        """Return joint-angle vector (radians) at time t (seconds)."""
        return np.array([p.sample(t) for p in self.profiles])

    def sample_velocity(self, t: float) -> np.ndarray:
        return np.array([p.sample_velocity(t) for p in self.profiles])

    def discretize(self, dt: float) -> np.ndarray:
        """
        Sample the trajectory at fixed timesteps.
        Returns array of shape (num_samples, num_joints).
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        n = int(np.ceil(self.total_time / dt)) + 1
        samples = np.zeros((n, len(self.profiles)))
        for i in range(n):
            samples[i] = self.sample(i * dt)
        return samples


def plan_synchronized_motion(
    q_start_rad: Sequence[float],
    q_end_rad: Sequence[float],
    arm: ArmGeometry = DEFAULT_ARM,
    v_scale: float = 1.0,
    a_scale: float = 1.0,
) -> SynchronizedTrajectory:
    """
    Build a synchronized trapezoidal trajectory between two joint configurations.

    All joints start at t=0, all joints finish at t=total_time. Joints with
    shorter individual travel are slowed down (not delayed) so motion is
    smooth throughout.

    Parameters
    ----------
    v_scale, a_scale : float in (0, 1]
        Optional derating for safety. 1.0 = full servo dynamic limits.
    """
    q0 = np.asarray(q_start_rad, dtype=float)
    q1 = np.asarray(q_end_rad, dtype=float)
    if q0.shape != q1.shape or q0.shape != (arm.num_dof,):
        raise ValueError(
            f"q_start and q_end must have shape ({arm.num_dof},); "
            f"got {q0.shape} and {q1.shape}"
        )
    if not (0.0 < v_scale <= 1.0 and 0.0 < a_scale <= 1.0):
        raise ValueError("v_scale and a_scale must lie in (0, 1]")

    v_max = arm.servo_max_speed_rad_s * v_scale
    a_max = arm.servo_max_accel_rad_s2 * a_scale

    # Step 1: compute each joint's minimum-time profile independently
    individual = [
        _min_time_profile(q0[i], q1[i], v_max, a_max)
        for i in range(arm.num_dof)
    ]

    # Step 2: find the slowest joint
    total_time = max(p.total_time for p in individual)

    if total_time <= 0.0:
        # No motion needed
        return SynchronizedTrajectory(
            profiles=individual, total_time=0.0
        )

    # Step 3: rescale every joint to complete in total_time
    synchronized = [
        _rescale_profile(q0[i], q1[i], total_time, a_max)
        for i in range(arm.num_dof)
    ]
    return SynchronizedTrajectory(profiles=synchronized, total_time=total_time)


if __name__ == "__main__":
    q0 = np.zeros(5)
    q1 = np.array([np.pi / 2, -np.pi / 3, np.pi / 2, np.pi / 4, 0.0])

    traj = plan_synchronized_motion(q0, q1)
    print(f"Total trajectory time: {traj.total_time:.3f} s")
    for i, p in enumerate(traj.profiles):
        print(
            f"  Joint {i}: delta={np.degrees(p.q_end - p.q_start):7.2f}°  "
            f"v_peak={p.v_peak:.3f} rad/s  "
            f"a_used={p.a_used:.3f} rad/s²  "
            f"t_acc={p.t_acc:.3f}s"
        )

    # Verify endpoints
    print(f"\nStart sample: {np.degrees(traj.sample(0)).round(2)}")
    print(f"End sample:   {np.degrees(traj.sample(traj.total_time)).round(2)}")
    print(f"Target:       {np.degrees(q1).round(2)}")
