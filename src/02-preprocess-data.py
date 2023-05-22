import os
import csv
import pandas as pd

root_folder = "data/motion_capture"

# Iterate over the subject folders
for subject in range(1, 12):
    subject_folder = os.path.join(root_folder, f"subject {subject}")

    # Check if the subject folder exists
    if not os.path.exists(subject_folder):
        continue

    # Iterate over the STS folders
    for sts_folder in ["STS", "STSAsym", "STSFast", "STSWeakLegs"]:
        sts_folder_path = os.path.join(subject_folder, sts_folder)

        # Check if the STS folder exists
        if not os.path.exists(sts_folder_path):
            continue

        # Create a CSV file for the merged data
        csv_file_path = os.path.join(sts_folder_path, f"{sts_folder}_merged.csv")

        # Collect data from each file
        data = []
        extension_types = (".sto", ".mot", ".trc")

        # Iterate over the files in the STS folder
        for file_name in os.listdir(sts_folder_path):
            file_path = os.path.join(sts_folder_path, file_name)

            # Check if the file is of the desired extension
            if file_name.endswith(extension_types):
                with open(file_path) as data_file:
                    # Skip the header lines
                    if file_name.endswith((".mot")):
                        skip_lines = 10
                        header_len = 0
                    elif file_name.endswith((".sto")):
                        skip_lines = 6
                        header_len = 0
                    elif file_name.endswith((".trc")):
                        skip_lines = 4
                        header_len = 2
                    for _ in range(skip_lines):
                        next(data_file)

                    # Read the lines of the file
                    lines = data_file.readlines()
                    if header_len > 0:
                        lines[header_len - 1] = "Frame#\tTime\t" + "".join(
                            lines[0:header_len]
                        )
                        lines = lines[header_len - 1 :]

                    # Append the lines to the data list
                    # split each line
                    lines = [line.strip().split() for line in lines]
                    data.append(lines)

        # Transpose the data to concatenate horizontally
        df = pd.concat(
            [
                pd.DataFrame(data[i][1:], columns=data[i][0])
                for i in range(len(extension_types))
            ],
            axis=1,
        )
        df.to_csv(csv_file_path, index=False)
