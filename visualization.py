import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatial import get_areas_from_points, get_positions_from_dataframe, plot_voronoi_cells, plot_players

tracking_file = "tracking_week_1.csv"
plays_file = "plays.csv"
game_id = 2022090800
play_id = 1385

# Handle Data I/O
df_tracking = pd.read_csv(tracking_file)
df_plays = pd.read_csv(plays_file)

df_full_tracking = df_tracking.merge(df_plays, on=["gameId", "playId"])

df_focused = df_full_tracking[
    (df_full_tracking["playId"] == play_id) & (df_full_tracking["gameId"] == game_id)
]

snapshot = df_focused[df_focused["frameType"] == "SNAP"]
point_dict = get_positions_from_dataframe(snapshot)

#plot results
plt.figure()
xx, yy = np.meshgrid(np.linspace(20,45,26), np.linspace(10,45,36))
plot_voronoi_cells(point_dict, xlim=[20,45], ylim=[10,45])
plt.show()