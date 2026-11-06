import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size, get_action_dim


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.action_dim = get_action_dim(self.env_cfg_type)
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)
        self.latest_obs = None
        self.latest_obs_batch = None

        assert len(self.robot_action_dim_info["arm_dim"]) == len(
            self.robot_action_dim_info["ee_dim"]
        ), "Arm and EE action dimensions must match"
        if self.action_type not in {"joint", "ee"}:
            raise ValueError("action_type must be 'joint' or 'ee'")

        print(f"[Model] Model successfully initialized with action type: {self.action_type}")

    def update_obs(self, obs):
        self.latest_obs = obs
        print("[Model] Received observation")

    def update_obs_batch(self, obs_list):
        self.latest_obs_batch = list(obs_list)
        print(f"[Model] Received observation batch of size: {len(obs_list)}")

    def get_action(self):
        if self.latest_obs is None:
            raise RuntimeError("update_obs must be called before get_action")
        action_list = self._actions_for_observation(self.latest_obs)
        print("[Model] Generated action")
        return action_list

    def get_action_batch(self, env_idx_list=None):
        if self.latest_obs_batch is None:
            raise RuntimeError(
                "update_obs_batch must be called before get_action_batch"
            )
        expected_size = (
            len(env_idx_list) if env_idx_list is not None else self.batch_size
        )
        if len(self.latest_obs_batch) != expected_size:
            raise ValueError(
                "Observation batch size does not match the requested environment batch."
            )
        action_batch = [
            self._actions_for_observation(obs) for obs in self.latest_obs_batch
        ]
        print(f"[Model] Generated action batch of size: {expected_size}")
        return action_batch

    def reset(self):
        self.latest_obs = None
        self.latest_obs_batch = None
        print("[Model] Model successfully reset")

    def _actions_for_observation(self, obs):
        state = obs.get("state")
        if not isinstance(state, dict):
            raise ValueError("Observation must contain a state dictionary.")

        arm_keys, ee_keys = self._action_keys()
        action = {}
        for index, (arm_key, ee_key) in enumerate(zip(arm_keys, ee_keys)):
            arm_dim = (
                self.robot_action_dim_info["arm_dim"][index]
                if self.action_type == "joint"
                else 7
            )
            action[arm_key] = self._state_vector(state, arm_key, arm_dim)
            action[ee_key] = self._state_vector(
                state,
                ee_key,
                self.robot_action_dim_info["ee_dim"][index],
            )
        return [
            {key: value.copy() for key, value in action.items()}
            for _ in range(2)
        ]

    def _action_keys(self):
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms == 1:
            arm_keys = [
                "arm_joint_state" if self.action_type == "joint" else "ee_pose"
            ]
            return arm_keys, ["ee_joint_state"]
        if num_arms == 2:
            prefix = ("left", "right")
            arm_suffix = (
                "arm_joint_state" if self.action_type == "joint" else "ee_pose"
            )
            return (
                [f"{side}_{arm_suffix}" for side in prefix],
                [f"{side}_ee_joint_state" for side in prefix],
            )
        raise NotImplementedError(f"Unsupported number of arms: {num_arms}")

    @staticmethod
    def _state_vector(state, key, expected_dim):
        if key not in state:
            raise KeyError(f"Observation state is missing {key!r}.")
        value = np.asarray(state[key], dtype=np.float32)
        if value.shape != (expected_dim,):
            raise ValueError(
                f"Observation state {key!r} must have shape ({expected_dim},), "
                f"got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Observation state {key!r} must be finite.")
        return value.copy()
