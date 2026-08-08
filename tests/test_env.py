import numpy as np
import mujoco
from gymnasium.utils.env_checker import check_env
from pathlib import Path

import jaka_zu35_mujoco_rl
from jaka_zu35_mujoco_rl import PandaPickEnv
from jaka_zu35_mujoco_rl.envs import SO101PickEnv, make_env


def test_model_loads():
    model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
    xml_path = model_dir / "panda_pick_scene.xml"
    assert xml_path.exists()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    # Panda arm: 7 joints + 8 gripper joints + 7 cube freejoint qpos = 22 nq.
    assert model.nq == 22
    # 7 arm + 8 gripper + 6 cube freejoint velocity dofs = 21 nv.
    assert model.nv == 21
    # 7 arm motor actuators + 1 gripper tendon actuator = 8 nu.
    assert model.nu == 8


def test_env_reset_and_step():
    env = PandaPickEnv(seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (32,)
    assert obs.dtype == np.float64
    assert isinstance(info, dict)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (32,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance_cube_to_tray" in info
    assert "distance_ee_to_cube" in info
    assert "steps" in info
    env.close()


def test_gymnasium_api():
    env = PandaPickEnv(seed=0)
    check_env(env, skip_render_check=True)
    env.close()


def test_backward_reach_alias_still_works():
    """Old imports (PandaReachEnv, JakaReachEnv) should still instantiate the pick env."""
    from jaka_zu35_mujoco_rl import JakaReachEnv, PandaReachEnv

    env_reach = PandaReachEnv(seed=0)
    env_jaka = JakaReachEnv(seed=0)
    assert env_reach.action_space.shape == (8,)
    assert env_jaka.action_space.shape == (8,)
    env_reach.close()
    env_jaka.close()


def test_so101_model_loads():
    model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
    xml_path = model_dir / "so101_pick_scene.xml"
    assert xml_path.exists()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nq == 13
    assert model.nv == 12
    assert model.nu == 6


def test_so101_env_reset_and_step():
    env = SO101PickEnv(seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (28,)
    assert obs.dtype == np.float64
    assert isinstance(info, dict)
    assert env.action_space.shape == (6,)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (28,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance_cube_to_tray" in info
    assert "distance_ee_to_cube" in info
    assert "steps" in info
    env.close()


def test_so101_gymnasium_api():
    env = SO101PickEnv(seed=0)
    check_env(env, skip_render_check=True)
    env.close()


def test_so101_overhead_and_wrist_cameras():
    env = SO101PickEnv(seed=0)
    env.reset(seed=1)
    overhead = env.get_observation_dict(camera="overhead")
    wrist = env.get_observation_dict(camera="wrist_cam")
    assert overhead["image"].shape == (480, 640, 3)
    assert wrist["image"].shape == (480, 640, 3)
    env.close()


def test_so101_default_camera_is_wrist():
    env = SO101PickEnv(seed=0)
    env.reset(seed=1)
    default_image = env.get_observation_dict()["image"]
    wrist_image = env.get_observation_dict(camera="wrist_cam")["image"]
    assert np.array_equal(default_image, wrist_image)
    env.close()


def test_so101_base_has_collision_geom():
    env = make_env("so101", render_mode="rgb_array")
    model = env.model
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_geom_ids = np.where(model.geom_bodyid == base_body_id)[0]
    assert len(base_geom_ids) > 0
    collisions = [
        gid
        for gid in base_geom_ids
        if model.geom_contype[gid] > 0 and model.geom_conaffinity[gid] > 0
    ]
    assert len(collisions) > 0, "base body has no collision geometry"
    env.close()


def test_so101_cube_stays_on_table():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    initial_pos = env._get_cube_position().copy()
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    for _ in range(100):
        env.step(action)
    final_pos = env._get_cube_position().copy()
    assert final_pos[2] >= env._table_height + env._cube_half_size - 1e-4
    assert abs(final_pos[2] - initial_pos[2]) < 1e-3
    assert np.linalg.norm(final_pos[:2] - initial_pos[:2]) < 1e-3
    env.close()


def test_so101_arm_visual_meshes_collide_with_table():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    # Drive the arm downward into the table surface.
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[1] = -1.0
    action[2] = -1.0
    action[3] = -1.0
    action[5] = 1.0
    for _ in range(200):
        env.step(action)

    model = env.model
    data = env.data
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    arm_bodies = {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"}
    contacts = False
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == table_id or c.geom2 == table_id:
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
            if b1 in arm_bodies or b2 in arm_bodies:
                contacts = True
                break
    assert contacts, "expected contact between table and arm visual mesh"
    env.close()


def test_make_env_factory():
    env_panda = make_env(robot="panda", seed=0)
    env_so101 = make_env(robot="so101", seed=0)
    assert env_panda.action_space.shape == (8,)
    assert env_so101.action_space.shape == (6,)
    env_panda.close()
    env_so101.close()

    try:
        make_env(robot="unknown")
        raise AssertionError("Expected ValueError for unknown robot")
    except ValueError:
        pass


def test_so101_cube_and_tray_rest_on_table():
    env = make_env("so101", render_mode="rgb_array")
    model = env.model

    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    table_height = float(
        model.body_pos[table_body_id, 2]
        + model.geom_pos[table_geom_id, 2]
        + model.geom_size[table_geom_id, 2]
    )

    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    cube_half_size = float(model.geom_size[cube_geom_id, 2])
    cube_z = float(model.body_pos[cube_body_id, 2])
    assert cube_z >= table_height + cube_half_size - 1e-6

    tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    tray_floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tray_floor")
    tray_floor_bottom = float(
        model.body_pos[tray_body_id, 2]
        + model.geom_pos[tray_floor_id, 2]
        - model.geom_size[tray_floor_id, 2]
    )
    assert tray_floor_bottom >= table_height - 1e-6

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_z = float(model.body_pos[base_body_id, 2])
    assert base_z >= table_height - 1e-6
    env.close()


def test_cube_is_small():
    for xml_name in ["panda_pick_scene.xml", "so101_pick_scene.xml"]:
        xml_path = Path(jaka_zu35_mujoco_rl.__file__).parent / "models" / xml_name
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        assert model.geom_size[cube_geom_id, 0] <= 0.02


def test_reward_components_include_lifted_and_push():
    env = PandaPickEnv(seed=0)
    env.reset(seed=1)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert "cube_lifted" in info
    assert "push" in info["reward_components"]
    assert "open_gripper" in info["reward_components"]
    env.close()
