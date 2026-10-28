# The pickle object import did not work

# import pickle
# with open(f"freemocap/{video_version}/saved_data/info/freemocap_data_handler.pkl", "rb") as f:
#     handler = pickle.load(f)
# print(handler)

# Getting data from the numpy files instead
import json
import numpy as np

video_version = "video1"

with open(f"freemocap/{video_version}/saved_data/info/trajectory_names.json") as f:
    trajectory_names = json.load(f)

nose_index = trajectory_names["body"].index("nose")
body_data = np.load(f"freemocap/{video_version}/saved_data/npy/body_frame_name_xyz.npy")  # shape: (frame, trajectory, xyz)
nose_xyz = body_data[:, nose_index, :]  # all frames, nose landmark, x/y/z

print(f"Nose XYZ shape: {nose_xyz.shape}")
print(f"Nose XYZ data (first 5 frames):\n{nose_xyz[:5]}")
