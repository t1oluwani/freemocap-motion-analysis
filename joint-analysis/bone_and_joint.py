import pandas as pd

DATA_TYPE = "bone_and_joint_data.csv"
RECORDING_NUMBER = "recording_17_33_15_gmt-7"
VIDEO_FOLDER_PATH = "freemocap/video1/saved_data"
BONE_AND_JOINT_CSV_FILE = f"{VIDEO_FOLDER_PATH}/{RECORDING_NUMBER}_{DATA_TYPE}"

# The headers in the bone and joint data files follow a pattern of <prefix>_<joint_name>.
# I have defined the prefixes as the HEADER_PREFIXES list.
HEADER_PREFIXES = [
    # Lower body
    "pelvis.R", "thigh.R", "shin.R", "foot.R", "heel.02.R",
    "pelvis.L", "thigh.L", "shin.L", "foot.L", "heel.02.L",

    # Spine
    "spine", "spine.001", "neck",

    # Right arm + hand
    "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
    "thumb.carpal.R", "thumb.01.R", "thumb.02.R", "thumb.03.R",
    "palm.01.R", "f_index.01.R", "f_index.02.R", "f_index.03.R",
    "palm.02.R", "f_middle.01.R", "f_middle.02.R", "f_middle.03.R",
    "palm.03.R", "f_ring.01.R", "f_ring.02.R", "f_ring.03.R",
    "palm.04.R", "f_pinky.01.R", "f_pinky.02.R", "f_pinky.03.R",

    # Left arm + hand
    "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
    "thumb.carpal.L", "thumb.01.L", "thumb.02.L", "thumb.03.L",
    "palm.01.L", "f_index.01.L", "f_index.02.L", "f_index.03.L",
    "palm.02.L", "f_middle.01.L", "f_middle.02.L", "f_middle.03.L",
    "palm.03.L", "f_ring.01.L", "f_ring.02.L", "f_ring.03.L",
    "palm.04.L", "f_pinky.01.L", "f_pinky.02.L", "f_pinky.03.L",
]

# The headers in the bone and joint data files follow a pattern of <bone_name>_<suffix>. 
# I have defined the suffixes as the HEADER_SUFFIXES list. 
HEADER_SUFFIXES = [
    [
        # Starting point
        "_head_center_world_x", "_head_center_world_y", "_head_center_world_z",

        # Ending point
        "_tail_center_world_x", "_tail_center_world_y", "_tail_center_world_z",

        # Rotation (quaternion)
        "_rotation_quaternion_x", "_rotation_quaternion_y", "_rotation_quaternion_z", "_rotation_quaternion_w",

        # Rotation (euler)
        "_rotation_euler_x", "_rotation_euler_y", "_rotation_euler_z", "_rotation_euler_order"

        # Transformation matrix (4x4)
    ] + [f"_matrix_{i}_{j}" for i in range(4) for j in range(4)]
]

# The relevant prefixes are the ones that correspond to the joints we are interested in analyzing.
RELEVANT_PREFIXES = [
    "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
    "thigh.L", "thigh.R", "shin.L", "shin.R"
]

# The JOINT_TO_BONE dictionary maps joint names to their corresponding bone names.
# This mapping should theoretically be used to convert joint data to bone data for analysis.
# These mappings were decided arbitrarily, but should line up with correct anatomically relevant locations.
JOINT_TO_BONE = {
    "left_shoulder":  "upper_arm.L",
    "right_shoulder": "upper_arm.R",
    "left_elbow":     "forearm.L",
    "right_elbow":    "forearm.R",
    "left_hip":       "thigh.L",
    "right_hip":      "thigh.R",
    "left_knee":      "shin.L",
    "right_knee":     "shin.R",
}
    
def build_relevant_columns_from_csv(csv_file_path: str) -> list[str]:
    """
    Build a list of relevant column names by reading the headers from a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file containing bone and joint data.

    Returns:
        list[str]: A list of relevant column names.
    """

    # Read the CSV file to get the headers
    df = pd.read_csv(csv_file_path, nrows=0)  # Read only the header row
    headers = df.columns.tolist()

    # Filter headers based on relevant prefixes and suffixes
    relevant_columns = []
    for header in headers:
        for prefix in RELEVANT_PREFIXES:
            if header.startswith(prefix):
                relevant_columns.append(header)
                break  # No need to check other prefixes once a match is found

    return relevant_columns

def get_joint_to_bone_mapping() -> dict[str, str]:
    """
    Get the mapping of joint names to their corresponding bone names.

    Returns:
        dict[str, str]: A dictionary mapping joint names to bone names.
    """
    return JOINT_TO_BONE

def get_relevant_prefixes() -> list[str]:
    """
    Get the list of relevant prefixes for bone and joint data.

    Returns:
        list[str]: A list of relevant prefixes.
    """
    return RELEVANT_PREFIXES

def get_header_prefixes() -> list[str]:
    """
    Get the list of header prefixes for bone and joint data.

    Returns:
        list[str]: A list of header prefixes.
    """
    return HEADER_PREFIXES



#call it
if __name__ == "__main__":
    relevant_columns = build_relevant_columns_from_csv(BONE_AND_JOINT_CSV_FILE)
    print(relevant_columns)