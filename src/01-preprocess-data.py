import argparse
import numpy as np
import os
import pandas as pd
import shutil

argp = argparse.ArgumentParser()
argp.add_argument("--data_dir", type=str, default="data/")
argp.add_argument("--seed", type=int, default=231)

def get_resp_bucket(scaled_response):
    if scaled_response == 0:
        return 0
    elif scaled_response <= 2:
        return 1
    else:
        return 2

def train_val_test_split(idx, pct_train, pct_test, seed):
    np.random.seed(seed)
    n = len(idx)
    n_train = round(n * pct_train)
    n_test = round(n * pct_test)
    train_idx = list(np.random.choice(idx, size=n_train, replace=False))
    non_train_idx = [ind for ind in idx if ind not in train_idx]
    assert len(non_train_idx) == n - n_train
    test_idx = list(np.random.choice(non_train_idx, size=n_test, replace=False))
    val_idx = [ind for ind in non_train_idx if ind not in test_idx]
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    assert len(set(train_idx).intersection(set(val_idx), set(test_idx))) == 0
    return {"train": train_idx, "val": val_idx, "test": test_idx}

if __name__ == "__main__":
    args = argp.parse_args()
    data_dir = args.data_dir
    seed = args.seed
    pose_data = pd.read_csv(os.path.join(data_dir, "raw/dataClean_text.csv"))
    # Extract valid IDs
    valid_ids = pose_data.subjectid.to_list()
    video_files = os.listdir(os.path.join(data_dir, "raw/videos/"))
    video_ids = set([f.split(".mp4")[0] for f in video_files])
    # Keep as list to preserve order
    kept_ids = [idx for idx in valid_ids
                if idx in set(valid_ids).intersection(video_ids)]
    print(f"{len(kept_ids)} unique observations in processed data")

    # Filter and process survey data accordingly
    col_names = pd.read_csv(os.path.join(data_dir, "raw/dataSurvey.csv"),
                            nrows=1).columns
    survey = pd.read_csv(os.path.join(data_dir, "raw/dataSurvey.csv"),
                         header=1, names=col_names)
    # Make response variable
    # Remove copy warning
    pd.set_option("mode.chained_assignment", None)
    survey["y_fall_risk"] = survey.falling_1.apply(get_resp_bucket)
    survey["y_fall_risk_binary"] = survey.y_fall_risk.apply(lambda x: int(x > 0))
    # Reinstate copy warning
    pd.set_option("mode.chained_assignment", "warn")

    id_splits = train_val_test_split(
        kept_ids, pct_train=0.7, pct_test=0.15, seed=seed
    )

    for split, split_idx in id_splits.items():
        id_txt = os.path.join(data_dir, f"processed/{split}-video-ids.txt")
        with open(id_txt, "w") as f:
            f.writelines([id + "\n" for id in split_idx])
        # Extract valid videos
        out_dir = os.path.join(data_dir, f"processed/{split}-videos/")
        # If path present, clear/delete the current output filepath first
        if os.path.exists(out_dir):
            [os.remove(os.path.join(out_dir, f)) for f in os.listdir(out_dir)]
        # If directory not present, create it
        else:
            os.makedirs(out_dir)
        sub_video_files = [
            path
            for path in os.listdir(os.path.join(data_dir, "raw/videos/"))
            if path.split(".mp4")[0] in split_idx
        ]
        for file_path in sub_video_files:
            shutil.copy2(os.path.join(data_dir, "raw/videos/", file_path),
                         os.path.join(out_dir, file_path))
        sub_survey = survey[[id in split_idx for id in survey.subjectid]]
        sub_survey.to_csv(
            os.path.join(data_dir, f"processed/{split}-survey-data.csv"), index=False
        )
        print(f"{split.title()} has {len(split_idx)} observations")
    print(f"All processed data in {os.path.join(data_dir, 'processed/')}")
