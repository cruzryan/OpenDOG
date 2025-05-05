import mujoco
import mujoco.viewer
import time
import os

# Load the scene XML which includes the robot XML
xml_path = '../our_robot/walking_scene.xml' # Make sure this path is correct

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
  start = time.time()
  while viewer.is_running(): # Run for 30 seconds
    step_start = time.time()

    # --- CRITICAL: Do NOT add any code here that modifies data.qpos, data.qvel, or data.xfrc_applied ---

    # Step the simulation
    mujoco.mj_step(model, data)

    # Sync viewer
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)