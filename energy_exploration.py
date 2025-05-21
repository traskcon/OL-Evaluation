import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from data_preprocessing import trim_initial_frames, set_play_type
from tqdm import tqdm
pd.options.mode.chained_assignment = None

def kinetic_energy(weight, speed):
    # Return player kinetic energy in Joules
    mass_kg = lbs_to_kg(weight)
    velocity_ms = yds_to_m(speed)
    return 0.5*mass_kg*(velocity_ms**2)

def yds_to_m(yards):
    # convert yards to meters
    return yards * 0.9144

def lbs_to_kg(pounds):
    # convert pounds to kg
    return pounds * 0.453592

def update_dict(force_dict, game_id, play_id, o_linemen, f_absorbed):
    # Store force data in dictionary after each play
    force_dict["gameId"].extend(5*[game_id])
    force_dict["playId"].extend(5*[play_id])
    force_dict["nflId"].extend(o_linemen)
    force_dict["forceAbsorbed"].extend(f_absorbed)
    return force_dict

tracking_file = "tracking_week_1.csv"
plays_file = "plays.csv"
players_file = "players.csv"
force_dict = {"gameId":[], "playId":[], "nflId":[], "forceAbsorbed":[]}

# Handle Data I/O
df_tracking = pd.read_csv(tracking_file)
df_tracking = trim_initial_frames(df_tracking)
df_plays = pd.read_csv(plays_file)
df_plays = set_play_type(df_plays)
df_pass_plays = df_plays[df_plays["playType"] == "Pass"]
df_players = pd.read_csv(players_file)

df_full_tracking = df_tracking.merge(df_pass_plays, on=["gameId", "playId"])
df_player_tracking = df_full_tracking.merge(df_players, on=["nflId","displayName"])

game_ids = pd.unique(df_player_tracking["gameId"])
for game_id in tqdm(game_ids):
    df_game = df_player_tracking[df_player_tracking["gameId"] == game_id]
    play_ids = pd.unique(df_game["playId"])
    for play_id in tqdm(play_ids):
        try:
            df_focused = df_game[df_game["playId"] == play_id]
            df_focused["frameId"] = df_focused["frameId"] - min(df_focused["frameId"]) #Standardize frame so 0 is Snap
            event_dict = pd.Series(df_focused.event.values, index=df_focused.frameId).to_dict()
            # Extract O-line and defensive player IDs
            o_linemen = pd.unique(df_focused[df_focused["position"].isin(["C","G","T"])]["nflId"])
            defense = pd.unique(df_focused[df_focused["club"] == df_focused["defensiveTeam"]]["nflId"])
            # Filter tracking data by position, calculate KE for each frame
            df_qb = df_focused[df_focused["position"] == "QB"]
            df_filtered = df_focused[(df_focused["position"].isin(["C","G","T"])) | (df_focused["club"] == df_focused["defensiveTeam"])]
            df_filtered["kineticEnergy"] = kinetic_energy(df_filtered["weight"], df_filtered["s"])
            ke_matrix = np.zeros((5,11))
            # Pivot data so frameId is the new index
            df_frames = df_filtered.pivot(index="frameId", columns="nflId", values=["x","y","kineticEnergy"])
            initial_def_x = df_frames["x"][defense].iloc[0].to_numpy()
            initial_def_y = df_frames["y"][defense].iloc[0].to_numpy()
            # Go frame-by-frame, calculating blocks, KE_max
            for frame in df_frames.index:
                if (event_dict[frame] == "tackle") or (event_dict[frame] == "pass_forward"):
                    # If play is over (check data for alternative end conditions later)
                    # Record QB, defensive positions
                    qb_pos = df_qb[df_qb["frameId"] == frame][["x","y"]].to_numpy()
                    final_def_x = df_frames["x"][defense].iloc[frame].to_numpy()
                    final_def_y = df_frames["y"][defense].iloc[frame].to_numpy()
                    ke_end = filtered * df_frames["kineticEnergy"][defense].iloc[frame].to_numpy()
                    break
                else:
                    x = df_frames["x"][[*o_linemen,*defense]].iloc[frame].to_numpy()
                    y = df_frames["y"][[*o_linemen,*defense]].iloc[frame].to_numpy()
                    # calculate pairwise distances
                    p_distances = squareform(pdist(np.array([x,y]).T))
                    # Trim down to only OL (rows) and Defense (columns)
                    p_distances = p_distances[:5,5:]
                    # Distance filter to Matchup matrix (block filter currently set at 1.25 yards)
                    filtered = np.where(p_distances < 1.25, 1, 0)
                    # Update KE_max matrix
                    ke_matrix = np.maximum(filtered * df_frames["kineticEnergy"][defense].iloc[frame].to_numpy(), ke_matrix)
            # calculate vectors
            v_ideal = qb_pos - np.array([initial_def_x, initial_def_y]).T
            v_actual = np.array([final_def_x, final_def_y]).T - np.array([initial_def_x, initial_def_y]).T
            d = np.sum(v_actual*v_ideal, axis=1) / np.linalg.norm(v_ideal, axis=1)
            # Update KE_max to F_absorbed
            f_absorbed = np.maximum((ke_matrix - ke_end) / d, np.zeros_like(ke_matrix))
            double_teams = np.maximum(np.sum(np.where(f_absorbed > 0, 1, 0), axis=0), np.ones_like(ke_matrix))
            f_absorbed /= double_teams
            # Sum and record F_absorbed for each O-lineman
            f_absorbed = np.sum(f_absorbed, axis=1)
            update_dict(force_dict, game_id, play_id, o_linemen, f_absorbed)
        except:
            print("GameId: {}, PlayId: {}".format(game_id, play_id))

df_force = pd.DataFrame.from_dict(force_dict)
df_force.to_csv("Week_1_Force_Data.csv")

'''df_post_snap = df_focused[df_focused["frameType"] != "BEFORE_SNAP"]
pass_rusher = df_post_snap[df_post_snap["displayName"].isin(["Aaron Donald","Leonard Floyd","Greg Gaines","A'Shawn Robinson"])]
pass_rusher = pass_rusher.merge(df_players, on=["nflId", "displayName"])

pass_rusher["kineticEnergy"] = kinetic_energy(pass_rusher["weight"], pass_rusher["s"])

df_plot = pass_rusher.pivot(index="frameId", columns="displayName", values="kineticEnergy")
df_plot.plot()
plt.ylabel("Kinetic Energy (Joules)")
plt.vlines(170,-10,1200, color="red", linestyles="dashed", label="Ball is Thrown")
plt.legend()
plt.show()'''
