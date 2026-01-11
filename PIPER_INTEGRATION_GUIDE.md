# Integrating AgileX Piper Robot with GELLO

This guide explains how to use GELLO teleoperation with the AgileX Piper 6-DOF robot arm.

## Prerequisites

- GELLO leader arm built with Dynamixel servos (0.5 scale of Piper)
- Servo IDs assigned via Dynamixel Wizard
- URDF of the Piper arm
- USB-to-CAN adapter connected to the Piper arm

## Robot Specifications

| Property | Value |
|----------|-------|
| Manufacturer | AgileX Robotics |
| Model | Piper |
| DOF | 6 joints + 1 gripper (7 total) |
| Communication | CAN bus via USB-to-CAN adapter |
| SDK | `piper_sdk` (`C_PiperInterface` class) |
| Control rate | Up to 200 Hz |

### Joint Limits

| Joint | Type | Min | Max | Velocity | Axis |
|-------|------|-----|-----|----------|------|
| joint1 | Revolute | -2.618 rad | 2.618 rad | 5 rad/s | Z |
| joint2 | Revolute | 0 rad | 3.14 rad | 5 rad/s | Z |
| joint3 | Revolute | -2.967 rad | 0 rad | 5 rad/s | Z |
| joint4 | Revolute | -1.745 rad | 1.745 rad | 5 rad/s | Z |
| joint5 | Revolute | -1.22 rad | 1.22 rad | 5 rad/s | Z |
| joint6 | Revolute | -2.0944 rad | 2.0944 rad | 3 rad/s | Z |
| joint7 (gripper) | Prismatic | 0 m | 0.035 m | 1 m/s | Z |

---

## Step 1: Install Dependencies

```bash
pip3 install piper_sdk python-can scipy
```

## Step 2: Set Up CAN Interface

The Piper communicates over CAN bus. You must activate the CAN interface before each session.

### Find your CAN port

```bash
bash ~/piper_ros/src/piper_ros/piper/scripts/find_all_can_port.sh
```

### Activate CAN interface

```bash
sudo ip link set can0 up type can bitrate 1000000
```

Or use the provided script:

```bash
bash ~/piper_ros/src/piper_ros/piper/scripts/can_activate.sh can0 1000000
```

---

## Step 3: Create the Robot Class

Create the file `gello/robots/piper.py`:

```python
import numpy as np
from typing import Dict
from piper_sdk import C_PiperInterface
import time

class PiperRobot:
    """AgileX Piper 6-DOF robot arm with gripper for GELLO teleoperation."""

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
        self._piper.GripperCtrl(0, 1000, 0x01, 0)  # Initialize gripper

        # Conversion factors
        # Piper SDK uses degrees * 1000 for joint angles
        self._deg_to_rad = 0.001 * (np.pi / 180.0)  # degrees*1000 -> radians
        self._rad_to_deg1000 = 1000.0 * (180.0 / np.pi)  # radians -> degrees*1000

    def num_dofs(self) -> int:
        """Return number of degrees of freedom (6 arm + 1 gripper)."""
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
        from scipy.spatial.transform import Rotation

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
        self._piper.GripperCtrl(0, 1000, 0x02, 0)  # Release gripper
```

---

## Step 4: Create YAML Configuration

Create the file `configs/piper.yaml`:

```yaml
robot:
  _target_: gello.robots.piper.PiperRobot
  can_port: "can0"
  gripper: true

agent:
  _target_: gello.agents.gello_agent.GelloAgent
  port: "/dev/serial/by-id/<your-U2D2-device>"
  dynamixel_config:
    _target_: gello.agents.gello_agent.DynamixelRobotConfig
    joint_ids: [1, 2, 3, 4, 5, 6]  # Your GELLO servo IDs
    joint_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # From calibration
    joint_signs: [1, 1, 1, 1, 1, 1]  # From calibration
    gripper_config: [7, 195, 152]  # [servo_id, open_deg, close_deg]

hz: 50
```

Replace `<your-U2D2-device>` with your actual U2D2 serial device. Find it with:

```bash
ls /dev/serial/by-id/
```

---

## Step 5: Calibrate Your GELLO

### 5.1 Find a Known Pose

Put both your GELLO leader arm and the Piper robot in a matching known configuration. A good choice is a "zero" or "home" pose where joint angles are well-defined.

### 5.2 Run Calibration Script

```bash
python scripts/gello_get_offset.py \
    --port /dev/serial/by-id/<your-U2D2-device> \
    --start-joints 0 0 0 0 0 0 \
    --joint-signs 1 1 1 1 1 1 \
    --gripper true
```

Arguments:
- `--start-joints`: The expected joint angles (in radians) when GELLO is in the known pose
- `--joint-signs`: Direction multiplier for each joint (1 or -1)
- `--gripper`: Set to `true` if you have a gripper servo

### 5.3 Adjust Joint Signs

If a joint moves in the wrong direction during testing:
1. Change that joint's sign from `1` to `-1` (or vice versa)
2. Re-run the calibration script

### 5.4 Update Configuration

Copy the output offsets into your `configs/piper.yaml`:

```yaml
joint_offsets: [3.142, 4.712, 1.571, ...]  # Example values
joint_signs: [1, -1, 1, 1, -1, 1]  # Example values
gripper_config: [7, 195, 152]  # [id, open_degrees, close_degrees]
```

---

## Step 6: Run GELLO Teleoperation

### 6.1 Activate CAN Interface

```bash
sudo ip link set can0 up type can bitrate 1000000
```

### 6.2 Launch GELLO

```bash
python experiments/launch_yaml.py --left-config-path configs/piper.yaml
```

The Piper arm should now follow the movements of your GELLO leader arm.

---

## Troubleshooting

### CAN Connection Issues

- Ensure CAN interface is active: `ip link show can0`
- Check CAN traffic: `candump can0`
- Verify USB-CAN adapter is connected: `lsusb`

### Permission Denied on Serial Port

Add your user to the `dialout` group:

```bash
sudo usermod -aG dialout $USER
```

Then log out and back in.

### Joint Moves Wrong Direction

Flip the sign for that joint in `joint_signs` and re-run calibration.

### Robot Doesn't Move

1. Check that the arm is enabled (LED indicators on robot)
2. Verify CAN communication is working
3. Check for error codes in the Piper SDK output

### GELLO Offsets Are Wrong

Re-run calibration with GELLO and Piper in an exact matching pose. Ensure both are powered on and stable before running the script.

---

## Unit Conversion Reference

The Piper SDK uses non-standard units:

| Value | SDK Unit | Standard Unit | Conversion |
|-------|----------|---------------|------------|
| Joint angle | degrees × 1000 | radians | `rad = (val / 1000) * (π / 180)` |
| Position | micrometers | meters | `m = val / 1,000,000` |
| Gripper | micrometers | meters | `m = val / 1,000,000` |
| Motor speed | RPM × 1000 | rad/s | `rad/s = (val / 1000) * (2π / 60)` |

---

## Additional Resources

- [GELLO Project Website](https://wuphilipp.github.io/gello_site/)
- [GELLO Hardware Repository](https://github.com/wuphilipp/gello_mechanical)
- [Piper ROS Repository](~/piper_ros)
- [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
