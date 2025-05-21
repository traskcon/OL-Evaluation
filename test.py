import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from energy_exploration import kinetic_energy

df_players = pd.read_csv("players.csv")
df_tracking = pd.read_csv("sample_play.csv")

energy_dict = {"gameId":[],"playId":[],"nflId":[],"forceAbsorbed":[]}

df_tracking = df_tracking.merge(df_players, on=["nflId","displayName"])
df_tracking["frameId"] = df_tracking["frameId"] - min(df_tracking["frameId"]) #Standardize frames so 0 is the snap
event_dict = pd.Series(df_tracking.event.values, index=df_tracking.frameId).to_dict()

o_linemen = pd.unique(df_tracking[df_tracking["position"].isin(["C","G","T"])]["displayName"])
defense = pd.unique(df_tracking[df_tracking["club"] == df_tracking["defensiveTeam"]]["displayName"])

df_qb = df_tracking[df_tracking["position"] == "QB"]
df_filtered = df_tracking[(df_tracking["position"].isin(["C","G","T"])) | (df_tracking["club"] == df_tracking["defensiveTeam"])]
df_filtered["kineticEnergy"] = kinetic_energy(df_filtered["weight"], df_filtered["s"])
ke_matrix = np.zeros((5,11))

df_frames = df_filtered.pivot(index="frameId", columns="displayName", values=["x","y","kineticEnergy"])
initial_def_x = df_frames["x"][defense].iloc[0].to_numpy()
initial_def_y = df_frames["y"][defense].iloc[0].to_numpy()

for frame in df_frames.index:
    if (event_dict[frame] == "tackle") or (event_dict[frame] == "pass_forward"):
        # If play is over (can check data for other end conditions later)
        #Record QB, defensive positions
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
        # distance filter to matchup matrix
        filtered = np.where(p_distances < 1.25, 1, 0)
        # Update KE_max matrix
        ke_matrix = np.maximum(filtered * df_frames["kineticEnergy"][defense].iloc[frame].to_numpy(), ke_matrix)

#calculate vectors
v_ideal = qb_pos - np.array([initial_def_x, initial_def_y]).T
v_actual = np.array([final_def_x,final_def_y]).T - np.array([initial_def_x,initial_def_y]).T
d = np.sum(v_actual*v_ideal, axis=1) / np.linalg.norm(v_ideal, axis=1)
# Update KE_max to F_absorbed
f_absorbed = np.maximum((ke_matrix - ke_end) / d, np.zeros_like(ke_matrix))
double_teams = np.maximum(np.sum(np.where(f_absorbed > 0, 1, 0), axis=0), np.ones_like(ke_matrix))
f_absorbed /= double_teams
# Sum and record F_absorbed for each OL
f_absorbed = np.sum(f_absorbed, axis=1)
energy_dict["gameId"].extend(5*[22])
energy_dict["nflId"].extend(o_linemen)
energy_dict["forceAbsorbed"].extend(f_absorbed)
print(energy_dict)