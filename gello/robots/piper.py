"""AgileX Piper 6-DOF robot arm interface for GELLO teleoperation."""

import time
from typing import Dict

import numpy as np
from piper_sdk import C_PiperInterface
from scipy.spatial.transform import Rotation


class PiperRobot:
    """AgileX Piper 6-DOF robot arm with optional gripper."""

    def __init__(self, can_port: str = "can0", gripper: bool = True):
        """
        Initialize connection to Piper robot.

        Args:
            can_port: CAN interface name (e.g., "can0")
            gripper: Whether the robot has a gripper attached
        """
        self._gripper = gripper
        self._piper = C_PiperInterface(can_name=can_port)
        self._piper.ConnectPort()

        # Wait for connection
        timeout = 5.0
        start = time.time()
        while not self._piper.isOk() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if not self._piper.isOk():
            raise ConnectionError(f"Failed to connect to Piper on {can_port}")

        # Enable the arm (7 = all joints including gripper)
        self._piper.EnableArm(7)
        if self._gripper:
            self._piper.GripperCtrl(0, 1000, 0x01, 0)  # Initialize gripper

        # Conversion factors
        # Piper SDK uses degrees * 1000 for joint angles
        self._deg_to_rad = 0.001 * (np.pi / 180.0)  # degrees*1000 -> radians
        self._rad_to_deg1000 = 1000.0 * (180.0 / np.pi)  # radians -> degrees*1000

    def num_dofs(self) -> int:
        """Return number of degrees of freedom (6 arm + 1 gripper if present)."""
        return 7 if self._gripper else 6

    def get_joint_state(self) -> np.ndarray:
        """
        Get current joint positions in radians.

        Returns:
            Array of joint positions [j1, j2, j3, j4, j5, j6, (gripper)]
        """
        joint_data = self._piper.GetArmJointMsgs().joint_state

        positions = np.array([
            joint_data.joint_1 * self._deg_to_rad,
            joint_data.joint_2 * self._deg_to_rad,
            joint_data.joint_3 * self._deg_to_rad,
            joint_data.joint_4 * self._deg_to_rad,
            joint_data.joint_5 * self._deg_to_rad,
            joint_data.joint_6 * self._deg_to_rad,
        ])

        if self._gripper:
            # Gripper position is in micrometers, convert to meters
            gripper_pos = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle
            gripper_meters = gripper_pos / 1_000_000.0
            positions = np.append(positions, gripper_meters)

        return positions

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """
        Command robot to move to target joint positions.

        Args:
            joint_state: Target positions in radians [j1, j2, j3, j4, j5, j6, (gripper)]
        """
        # Convert radians to degrees * 1000
        j1 = int(joint_state[0] * self._rad_to_deg1000)
        j2 = int(joint_state[1] * self._rad_to_deg1000)
        j3 = int(joint_state[2] * self._rad_to_deg1000)
        j4 = int(joint_state[3] * self._rad_to_deg1000)
        j5 = int(joint_state[4] * self._rad_to_deg1000)
        j6 = int(joint_state[5] * self._rad_to_deg1000)

        self._piper.JointCtrl(j1, j2, j3, j4, j5, j6)

        if self._gripper and len(joint_state) >= 7:
            # Convert meters to micrometers
            gripper_um = int(abs(joint_state[6]) * 1_000_000)
            self._piper.GripperCtrl(gripper_um, 1000, 0x01, 0)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get full observation dictionary.

        Returns:
            Dictionary with keys:
                - joint_positions: Current joint positions (radians)
                - joint_velocities: Current joint velocities (rad/s)
                - ee_pos_quat: End-effector pose [x, y, z, qx, qy, qz, qw]
                - gripper_position: Normalized gripper position [0-1]
        """
        # Joint positions
        joint_positions = self.get_joint_state()

        # Joint velocities (from motor speed data)
        speed_data = self._piper.GetArmHighSpdInfoMsgs()
        # Motor speeds are in RPM * 1000, convert to rad/s
        rpm_to_rads = (2 * np.pi) / 60.0 / 1000.0
        joint_velocities = np.array([
            speed_data.motor_1.motor_speed * rpm_to_rads,
            speed_data.motor_2.motor_speed * rpm_to_rads,
            speed_data.motor_3.motor_speed * rpm_to_rads,
            speed_data.motor_4.motor_speed * rpm_to_rads,
            speed_data.motor_5.motor_speed * rpm_to_rads,
            speed_data.motor_6.motor_speed * rpm_to_rads,
        ])
        if self._gripper:
            joint_velocities = np.append(joint_velocities, 0.0)

        # End-effector pose
        ee_data = self._piper.GetArmEndPoseMsgs().end_pose
        # Position: micrometers -> meters
        ee_pos = np.array([
            ee_data.X_axis / 1_000_000.0,
            ee_data.Y_axis / 1_000_000.0,
            ee_data.Z_axis / 1_000_000.0,
        ])
        # Orientation: degrees*1000 -> radians -> quaternion
        roll = ee_data.RX_axis * self._deg_to_rad
        pitch = ee_data.RY_axis * self._deg_to_rad
        yaw = ee_data.RZ_axis * self._deg_to_rad
        quat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # [x, y, z, w]

        ee_pos_quat = np.concatenate([ee_pos, quat])

        # Gripper position (normalized 0-1, max 80mm)
        if self._gripper:
            gripper_pos = joint_positions[-1] / 0.08
        else:
            gripper_pos = 0.0

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([gripper_pos]),
        }

    def disable(self) -> None:
        """Disable the arm. Call this on shutdown."""
        self._piper.DisableArm(7)
        if self._gripper:
            self._piper.GripperCtrl(0, 1000, 0x02, 0)  # Release gripper
