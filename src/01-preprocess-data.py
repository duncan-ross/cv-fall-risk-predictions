import os
import pandas as pd
import shutil


def get_resp_bucket(scaled_response):
    if scaled_response == 0:
        return 0
    elif scaled_response <= 2:
        return 1
    else:
        return 2


if __name__ == "__main__":
    pose_data = pd.read_csv("data/raw/dataClean_text.csv")
    # Extract valid IDs
    valid_ids = set(pose_data.subjectid.to_list())
    video_files = os.listdir("data/raw/videos/")
    video_id = set([f.split(".mp4")[0] for f in os.listdir("data/raw/videos/")])
    kept_ids = list(valid_ids.intersection(video_id))

    with open("data/processed/kept-video-ids.txt", "w") as f:
        f.writelines([id + "\n" for id in kept_ids])

    # Extract valid videos
    out_dir = "data/processed/valid-videos/"
    # If path already present, clear/delete the current output filepath first
    if os.path.exists(out_dir):
        [os.remove(os.path.join(out_dir, f)) for f in os.listdir(out_dir)]
    # If directory not present, create it
    else:
        os.makedirs(out_dir)
    valid_video_files = [
        path
        for path in os.listdir("data/raw/videos/")
        if path.split(".mp4")[0] in kept_ids
    ]
    for file_path in valid_video_files:
        shutil.copy2(
            os.path.join("data/raw/videos/", file_path),
            os.path.join(out_dir, file_path),
        )

    # Filter and process survey data accordingly
    col_names = pd.read_csv("data/raw/dataSurvey.csv", nrows=1).columns
    survey = pd.read_csv("data/raw/dataSurvey.csv", header=1, names=col_names)
    valid_survey = survey[[id in kept_ids for id in survey.subjectid]]
    # Make response variable
    # Remove copy warning
    pd.set_option("mode.chained_assignment", None)
    valid_survey["y_fall_risk"] = valid_survey.falling_1.apply(get_resp_bucket)
    # Reinstate copy warning
    pd.set_option("mode.chained_assignment", "warn")
    valid_survey.to_csv("data/processed/survey-data-processed.csv", index=False)
    print(f"{len(kept_ids)} unique observations in processed data")
    print(f"All processed data in data/processed/")
