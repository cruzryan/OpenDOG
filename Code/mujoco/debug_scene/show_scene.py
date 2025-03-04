import mujoco
import matplotlib.pyplot as plt
from mujoco import viewer

model = mujoco.MjModel.from_xml_path('../app/our_robot/walking_scene.xml')
data = mujoco.MjData(model)
viewer.launch(model, data)
