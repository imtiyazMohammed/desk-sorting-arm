"""
Test suite for the Phase A kinematics simulation.

Test areas
----------
1. Geometry invariants
2. Forward kinematics — closed-form known positions
3. IK / FK round-trip consistency
4. Reachability envelope correctness
5. Joint-limit enforcement
6. Trajectory endpoint / synchronization guarantees
"""

from __future__ import annotations

import numpy as np
import pytest

from src.arm_chain import build_chain, extract_active_angles, full_joint_vector
from src.forward_kinematics import ForwardKinematics
from src.geometry import DEFAULT_ARM, ArmGeometry, JointLimit
from src.inverse_kinematics import IKStatus, InverseKinematics
from src.trajectory import (
    plan_synchronized_motion,
    _min_time_profile,
)


# =========================================================================
# 1. Geometry invariants
# =========================================================================


class TestGeometry:
    def test_default_geometry_is_5dof(self):
        assert DEFAULT_ARM.num_dof == 5

    def test_total_reach_sums_links(self):
        assert DEFAULT_ARM.total_reach_mm == pytest.approx(950.0)

    def test_safe_reach_is_85pct(self):
        assert DEFAULT_ARM.safe_reach_mm == pytest.approx(950.0 * 0.85)

    def test_worst_case_desk_corner_matches_geometry(self):
        # base at (600, 0) on 1200x600 desk => corner at (0, 600) or (1200, 600)
        # distance = sqrt(600^2 + 600^2) = 848.528 mm
        assert DEFAULT_ARM.worst_case_desk_reach_mm() == pytest.approx(
            np.sqrt(600.0**2 + 600.0**2), abs=1e-6
        )

    def test_joint_limit_ordering(self):
        with pytest.raises(ValueError):
            JointLimit(min_rad=1.0, max_rad=0.5, name="bad")

    def test_joint_limit_clamp(self):
        lim = JointLimit(-1.0, 1.0, "test")
        assert lim.clamp(2.0) == 1.0
        assert lim.clamp(-2.0) == -1.0
        assert lim.clamp(0.5) == 0.5


# =========================================================================
# 2. Forward kinematics
# =========================================================================


class TestForwardKinematics:
    @pytest.fixture
    def fk(self) -> ForwardKinematics:
        return ForwardKinematics(DEFAULT_ARM)

    def test_zero_pose_is_horizontal_extension(self, fk):
        pose = fk.compute(np.zeros(5))
        # Arm horizontal along +X: tip at (L1+L2+L3, 0, base_height)
        assert pose.position_mm[0] == pytest.approx(950.0, abs=1e-6)
        assert pose.position_mm[1] == pytest.approx(0.0, abs=1e-9)
        assert pose.position_mm[2] == pytest.approx(100.0, abs=1e-6)

    def test_shoulder_up_gives_vertical_arm(self, fk):
        pose = fk.compute([0.0, -np.pi / 2, 0.0, 0.0, 0.0])
        assert pose.position_mm[0] == pytest.approx(0.0, abs=1e-9)
        assert pose.position_mm[1] == pytest.approx(0.0, abs=1e-9)
        assert pose.position_mm[2] == pytest.approx(100.0 + 950.0, abs=1e-6)

    def test_base_yaw_rotates_to_y_axis(self, fk):
        pose = fk.compute([np.pi / 2, 0.0, 0.0, 0.0, 0.0])
        assert pose.position_mm[0] == pytest.approx(0.0, abs=1e-9)
        assert pose.position_mm[1] == pytest.approx(950.0, abs=1e-6)
        assert pose.position_mm[2] == pytest.approx(100.0, abs=1e-6)

    def test_negative_base_yaw_rotates_to_negative_y(self, fk):
        pose = fk.compute([-np.pi / 2, 0.0, 0.0, 0.0, 0.0])
        assert pose.position_mm[1] == pytest.approx(-950.0, abs=1e-6)

    def test_folded_pose_matches_geometry(self, fk):
        # shoulder -45, elbow -45 makes upper arm at 45° up,
        # forearm+wrist_to_tip pointing straight up.
        # elbow at (L1*cos45, 0, base_height + L1*sin45)
        # tcp   at elbow + (0, 0, L2+L3)
        pose = fk.compute([0.0, -np.pi / 4, -np.pi / 4, 0.0, 0.0])
        cos45 = np.cos(np.pi / 4)
        expected_x = DEFAULT_ARM.l1_upper_arm_mm * cos45
        expected_z = (
            DEFAULT_ARM.base_height_mm
            + DEFAULT_ARM.l1_upper_arm_mm * cos45
            + DEFAULT_ARM.l2_forearm_mm
            + DEFAULT_ARM.l3_wrist_to_tip_mm
        )
        assert pose.position_mm[0] == pytest.approx(expected_x, abs=1e-6)
        assert pose.position_mm[1] == pytest.approx(0.0, abs=1e-9)
        assert pose.position_mm[2] == pytest.approx(expected_z, abs=1e-6)

    def test_wrong_dof_count_raises(self, fk):
        with pytest.raises(ValueError):
            fk.compute(np.zeros(4))

    def test_out_of_limit_angle_raises(self, fk):
        with pytest.raises(ValueError):
            fk.compute([np.radians(200.0), 0.0, 0.0, 0.0, 0.0])

    def test_joint_positions_shape(self, fk):
        positions = fk.joint_positions(np.zeros(5))
        assert positions.shape == (5, 3)


# =========================================================================
# 3. IK / FK round-trip consistency
# =========================================================================


class TestIKFKConsistency:
    """
    The strongest test: for any joint configuration q, running FK gives a
    TCP position P; feeding P back through IK should yield joint angles q'
    whose FK also produces P. Note q' may differ from q (redundant / multiple
    solutions), but the reproduced TCP position must match.
    """

    @pytest.fixture
    def fk(self) -> ForwardKinematics:
        return ForwardKinematics(DEFAULT_ARM)

    @pytest.fixture
    def ik(self) -> InverseKinematics:
        return InverseKinematics(DEFAULT_ARM, position_tolerance_mm=1.0)

    @pytest.mark.parametrize(
        "q_deg",
        [
            [0.0, -30.0, 60.0, 0.0, 0.0],
            [45.0, -45.0, 90.0, 0.0, 0.0],
            [-30.0, -60.0, 90.0, 30.0, 0.0],
            [90.0, -20.0, 40.0, -20.0, 0.0],
        ],
    )
    def test_ik_fk_round_trip(self, fk, ik, q_deg):
        q_rad = np.radians(q_deg)
        target_pose = fk.compute(q_rad)
        target_position = target_pose.position_mm

        sol = ik.solve(target_position)
        assert sol.status == IKStatus.SUCCESS, sol.message
        assert sol.position_error_mm < 1.0

        # Re-verify with our independent FK implementation
        reproduced = fk.compute(sol.joint_angles_rad)
        error = np.linalg.norm(reproduced.position_mm - target_position)
        assert error < 1.0


# =========================================================================
# 4. Reachability envelope
# =========================================================================


class TestReachability:
    @pytest.fixture
    def ik(self) -> InverseKinematics:
        return InverseKinematics(DEFAULT_ARM)

    def test_far_point_unreachable(self, ik):
        # 2000mm from shoulder is way beyond 950mm total reach
        assert not ik.is_reachable([2000.0, 0.0, 100.0])
        sol = ik.solve([2000.0, 0.0, 100.0])
        assert sol.status == IKStatus.UNREACHABLE

    def test_far_below_desk_unreachable(self, ik):
        # A point 500mm below the desk surface, out of realistic sphere
        assert not ik.is_reachable([0.0, 0.0, -1500.0])

    def test_mid_workspace_reachable(self, ik):
        assert ik.is_reachable([400.0, 200.0, 200.0])

    def test_boundary_point_at_max_reach(self, ik):
        # Exactly at sphere boundary: shoulder is at (0,0,100), so target
        # (950, 0, 100) is 950mm out = total_reach. Should be on the edge
        # of the pre-flight envelope (inclusive).
        assert ik.is_reachable([950.0, 0.0, 100.0])


# =========================================================================
# 5. Joint-limit enforcement in IK
# =========================================================================


class TestJointLimits:
    def test_ik_respects_bounds(self):
        ik = InverseKinematics(DEFAULT_ARM)
        # Try many random reachable targets; solutions must be in-bounds.
        rng = np.random.default_rng(42)
        successes = 0
        for _ in range(50):
            target = rng.uniform([100, -400, 50], [700, 400, 400])
            sol = ik.solve(target)
            if sol.status != IKStatus.SUCCESS:
                continue
            successes += 1
            for angle, limit in zip(
                sol.joint_angles_rad, DEFAULT_ARM.joint_limits
            ):
                assert limit.contains(angle, tolerance_rad=1e-4), (
                    f"Joint {limit.name} = {np.degrees(angle):.2f}° "
                    f"outside [{limit.min_deg:.2f}, {limit.max_deg:.2f}]"
                )
        # Sanity: at least some solves should have succeeded
        assert successes > 10, f"Only {successes}/50 random targets succeeded"


# =========================================================================
# 6. Trajectory tests
# =========================================================================


class TestTrajectory:
    def test_zero_motion_returns_zero_time(self):
        q = np.array([0.1, -0.2, 0.3, 0.0, 0.0])
        traj = plan_synchronized_motion(q, q)
        assert traj.total_time == 0.0

    def test_endpoints_are_exact(self):
        q0 = np.zeros(5)
        q1 = np.array([np.radians(60), np.radians(-70), np.radians(90),
                       np.radians(30), 0.0])
        traj = plan_synchronized_motion(q0, q1)
        assert np.allclose(traj.sample(0.0), q0, atol=1e-6)
        assert np.allclose(traj.sample(traj.total_time), q1, atol=1e-6)

    def test_all_joints_finish_at_same_time(self):
        q0 = np.zeros(5)
        q1 = np.array([np.radians(60), np.radians(-70), np.radians(90),
                       np.radians(30), 0.0])
        traj = plan_synchronized_motion(q0, q1)
        for p in traj.profiles:
            assert p.total_time == pytest.approx(traj.total_time, abs=1e-9)

    def test_zero_distance_joint_stays_still(self):
        q0 = np.zeros(5)
        q1 = np.array([np.radians(60), np.radians(-70), np.radians(90),
                       np.radians(30), 0.0])   # joint 4 unchanged
        traj = plan_synchronized_motion(q0, q1)
        mid = traj.total_time / 2
        assert traj.sample(mid)[4] == pytest.approx(0.0, abs=1e-9)

    def test_velocity_zero_at_endpoints(self):
        q0 = np.zeros(5)
        q1 = np.array([np.radians(90), 0.0, 0.0, 0.0, 0.0])
        traj = plan_synchronized_motion(q0, q1)
        v0 = traj.sample_velocity(0.0)
        vf = traj.sample_velocity(traj.total_time)
        assert np.allclose(v0, 0.0, atol=1e-6)
        assert np.allclose(vf, 0.0, atol=1e-6)

    def test_min_time_profile_matches_kinematics(self):
        # A 90° move at v_max=6 rad/s, a_max=30 rad/s^2 should be trapezoidal:
        # t_acc = 6/30 = 0.2s, d_acc = 0.5*30*0.04 = 0.6 rad each end
        # remaining distance = pi/2 - 1.2 rad = 0.371 rad, cruise = 0.371/6 = 0.0618s
        # total = 0.4 + 0.0618 = 0.4618 s
        p = _min_time_profile(0.0, np.pi / 2, v_max=6.0, a_max=30.0)
        assert p.t_acc == pytest.approx(0.2, abs=1e-6)
        assert p.t_cruise == pytest.approx(
            (np.pi / 2 - 1.2) / 6.0, abs=1e-6
        )
        assert p.total_time == pytest.approx(
            0.4 + (np.pi / 2 - 1.2) / 6.0, abs=1e-6
        )
        # End position exactly correct
        assert p.sample(p.total_time) == pytest.approx(np.pi / 2, abs=1e-9)


# =========================================================================
# 7. Chain construction sanity (ikpy matches our FK)
# =========================================================================


class TestArmChain:
    """
    Independently verify that ikpy's Chain, constructed by build_chain,
    computes the same TCP position as our hand-derived FK for a few
    representative joint configurations.
    """

    @pytest.mark.parametrize(
        "q_deg",
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -45.0, 90.0, 0.0, 0.0],
            [30.0, -60.0, 45.0, 15.0, 0.0],
        ],
    )
    def test_ikpy_fk_matches_manual_fk(self, q_deg):
        q_rad = np.radians(q_deg)
        manual_fk = ForwardKinematics(DEFAULT_ARM)
        chain = build_chain(DEFAULT_ARM)

        manual_pose = manual_fk.compute(q_rad)

        ikpy_full = full_joint_vector(q_rad)
        ikpy_frame = chain.forward_kinematics(ikpy_full)
        ikpy_position_mm = ikpy_frame[:3, 3] * 1000.0  # m -> mm

        error = np.linalg.norm(manual_pose.position_mm - ikpy_position_mm)
        assert error < 1e-3, (
            f"Manual FK vs ikpy FK disagree: "
            f"manual={manual_pose.position_mm}, ikpy={ikpy_position_mm}"
        )
