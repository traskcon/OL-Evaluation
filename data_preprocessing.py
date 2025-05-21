import numpy as np
import pandas as pd

def trim_initial_frames(tracking_df):
    # Remove all pre-snap frames
    postsnap_df = tracking_df[tracking_df["frameType"] != "BEFORE_SNAP"]
    return postsnap_df

def set_play_type(df_plays):
    # Label all frames as pass or run play
    df_plays["playType"] = np.where(pd.isna(df_plays["passResult"]), "Run","Pass")
    return df_plays