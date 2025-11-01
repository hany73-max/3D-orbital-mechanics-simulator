import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="3D Orbital Mechanics Simulator", layout="wide")
st.title("🌍 Orbital Mechanics — Interactive 3D simulation")

if "camera_reset_token" not in st.session_state:
    st.session_state["camera_reset_token"] = "init_token"

st.sidebar.header("Orbital Elements & UI")
a = st.sidebar.number_input("Semi-major axis a (km)", min_value=1000.0, max_value=1e6, value=12000.0, step=100.0, format="%.1f")
e = st.sidebar.slider("Eccentricity e", 0.0, 0.98, 0.4, 0.001)
inc_deg = st.sidebar.slider("Inclination i (°)", 0.0, 180.0, 40.0, 0.1)
raan_deg = st.sidebar.slider("RAAN Ω (°)", 0.0, 360.0, 20.0, 0.1)
argp_deg = st.sidebar.slider("Argument of periapsis ω (°)", 0.0, 360.0, 30.0, 0.1)

n_frames = st.sidebar.slider("Animation frames (resolution)", 24, 1440, 240, 24)
base_frame_ms = st.sidebar.slider("Base frame duration (ms)", 10, 500, 40, 5)
speed_multiplier = st.sidebar.slider("Speed multiplier", 0.1, 10.0, 1.0, 0.1)
frame_duration_ms = float(base_frame_ms) / float(speed_multiplier)

show_orbit_path = st.sidebar.checkbox("Show full orbit path", value=True)
show_axes = st.sidebar.checkbox("Show PQW axes", value=True)
show_vectors = st.sidebar.checkbox("Show r & v vectors", value=True)

st.sidebar.markdown("**Camera controls**")
if st.sidebar.button("Reset camera"):
    st.session_state["camera_reset_token"] = str(time.time())

mu = 398600.4418 
Re = 6371.0 

inc = np.radians(inc_deg)
RAAN = np.radians(raan_deg)
argp = np.radians(argp_deg)

def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

R_pqw_to_eci = Rz(RAAN) @ Rx(inc) @ Rz(argp)

p = a * (1 - e ** 2)
nu0 = 0.0
r_pf0 = p / (1 + e * np.cos(nu0))
r_pqw0 = np.array([r_pf0 * np.cos(nu0), r_pf0 * np.sin(nu0), 0.0])
v_pqw0 = np.sqrt(mu / p) * np.array([-np.sin(nu0), e + np.cos(nu0), 0.0])

r0 = (R_pqw_to_eci @ r_pqw0).ravel()
v0 = (R_pqw_to_eci @ v_pqw0).ravel()

def stumpff_C(z):
    if z > 1e-8:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return 0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40320.0

def stumpff_S(z):
    if z > 1e-8:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
    else:
        return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0 - z**3 / 362880.0

def universal_chi_solver(r0_vec, v0_vec, dt, mu):
    r0_mag = np.linalg.norm(r0_vec)
    vr0 = np.dot(r0_vec, v0_vec) / r0_mag
    v0_sq = np.dot(v0_vec, v0_vec)
    alpha = 2.0 / r0_mag - v0_sq / mu 

    if dt == 0.0:
        return 0.0

    if alpha > 0:
        chi = np.sqrt(mu) * dt * alpha
    elif alpha < 0: 
        chi = np.sign(dt) * np.sqrt(-1.0 / alpha) * np.log((-2.0 * mu * alpha * dt) / (vr0 + np.sign(dt) * np.sqrt(-mu / alpha) * (1 - r0_mag * alpha)))
    else:
        h = np.cross(r0_vec, v0_vec)
        p_par = np.dot(h, h) / mu
        s = 0.5 * (np.pi / 2 - np.arctan(3.0 * np.sqrt(mu / (p_par**3)) * dt))
        w = np.arctan(np.tan(s)**(1/3))
        chi = np.sqrt(p_par) * 2.0 * w / np.tan(2.0 * w)

    for _ in range(200):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)

        F = (r0_mag * vr0 / np.sqrt(mu)) * chi**2 * C + (1.0 - alpha * r0_mag) * chi**3 * S + r0_mag * chi - np.sqrt(mu) * dt

        dF = (r0_mag * vr0 / np.sqrt(mu)) * chi * (1.0 - alpha * chi**2 * S) + (1.0 - alpha * r0_mag) * chi**2 * C + r0_mag

        if abs(dF) < 1e-14:
            break

        delta = F / dF
        chi -= delta
        if abs(delta) < 1e-9:
            break

    return chi

def lagrange_fg_from_r0v0(r0_vec, v0_vec, dt, mu):
    r0_mag = np.linalg.norm(r0_vec)
    v0_mag = np.linalg.norm(v0_vec)
    alpha = 2.0 / r0_mag - v0_mag**2 / mu

    if dt == 0.0:
        return r0_vec.copy(), v0_vec.copy()

    chi = universal_chi_solver(r0_vec, v0_vec, dt, mu)
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)

    f = 1.0 - (chi**2 / r0_mag) * C
    g = dt - (1.0 / np.sqrt(mu)) * chi**3 * S

    r_vec = f * r0_vec + g * v0_vec
    r_mag = np.linalg.norm(r_vec)

    fdot = (np.sqrt(mu) / (r_mag * r0_mag)) * (alpha * chi**3 * S - chi)
    gdot = 1.0 - (chi**2 / r_mag) * C

    v_vec = fdot * r0_vec + gdot * v0_vec

    return r_vec, v_vec

T = 2 * np.pi * np.sqrt(a ** 3 / mu) 
frame_times = np.linspace(0, T, n_frames, endpoint=False)

r_eci = np.zeros((n_frames, 3))
v_eci = np.zeros((n_frames, 3))

for k, dt in enumerate(frame_times):
    rk, vk = lagrange_fg_from_r0v0(r0, v0, dt, mu)
    r_eci[k, :] = rk
    v_eci[k, :] = vk

axis_len = np.max(np.linalg.norm(r_eci, axis=1)) * 0.8
p_end_eci = (R_pqw_to_eci @ np.array([axis_len, 0.0, 0.0])).ravel()
q_end_eci = (R_pqw_to_eci @ np.array([0.0, axis_len, 0.0])).ravel()
w_end_eci = (R_pqw_to_eci @ np.array([0.0, 0.0, axis_len])).ravel()

max_extent = max(np.max(np.linalg.norm(r_eci, axis=1)), Re) * 1.3
axis_range = [-max_extent, max_extent]
vscale = np.max(np.linalg.norm(r_eci, axis=1)) / 8.0 if np.max(np.linalg.norm(r_eci, axis=1)) > 0 else 1.0

phi, theta = np.mgrid[0:np.pi:40j, 0:2 * np.pi:80j]
x = Re * np.sin(phi) * np.cos(theta)
y = Re * np.sin(phi) * np.sin(theta)
z = Re * np.cos(phi)
earth_surface = go.Surface(x=x, y=y, z=z,
                           colorscale=[[0, "rgb(10,30,120)"], [1, "rgb(50,120,255)"]],
                           showscale=False, opacity=0.95, name="Earth", hoverinfo="skip")

orbit_path = go.Scatter3d(x=r_eci[:, 0], y=r_eci[:, 1], z=r_eci[:, 2], mode="lines",
                          line=dict(color="white", width=2), name="Orbit Path", visible=show_orbit_path, hoverinfo="skip")
p_axis = go.Scatter3d(x=[0, p_end_eci[0]], y=[0, p_end_eci[1]], z=[0, p_end_eci[2]], mode="lines",
                      line=dict(color="cyan", width=4), name="P-axis", visible=show_axes, hoverinfo="skip")
q_axis = go.Scatter3d(x=[0, q_end_eci[0]], y=[0, q_end_eci[1]], z=[0, q_end_eci[2]], mode="lines",
                      line=dict(color="magenta", width=4), name="Q-axis", visible=show_axes, hoverinfo="skip")
w_axis = go.Scatter3d(x=[0, w_end_eci[0]], y=[0, w_end_eci[1]], z=[0, w_end_eci[2]], mode="lines",
                      line=dict(color="yellow", width=4), name="W-axis", visible=show_axes, hoverinfo="skip")

sat_trace = go.Scatter3d(x=[r_eci[0, 0]], y=[r_eci[0, 1]], z=[r_eci[0, 2]],
                         mode="markers", marker=dict(size=6, color="red"), name="Satellite")
r_vector = go.Scatter3d(x=[0, r_eci[0, 0]], y=[0, r_eci[0, 1]], z=[0, r_eci[0, 2]],
                        mode="lines", line=dict(color="orange", width=4), name="r vector", visible=show_vectors)
v_vector = go.Scatter3d(x=[r_eci[0, 0], r_eci[0, 0] + v_eci[0, 0] * vscale],
                        y=[r_eci[0, 1], r_eci[0, 1] + v_eci[0, 1] * vscale],
                        z=[r_eci[0, 2], r_eci[0, 2] + v_eci[0, 2] * vscale],
                        mode="lines", line=dict(color="yellow", width=4), name="v vector", visible=show_vectors)

static_traces = [earth_surface, orbit_path, p_axis, q_axis, w_axis]
moving_traces = [sat_trace, r_vector, v_vector]
all_traces = static_traces + moving_traces
sat_index = len(static_traces) + 0
r_index = len(static_traces) + 1
v_index = len(static_traces) + 2

frames = []
nu_array = np.zeros(n_frames)

p_unit_eci = (R_pqw_to_eci @ np.array([1.0, 0.0, 0.0])).ravel()
q_unit_eci = (R_pqw_to_eci @ np.array([0.0, 1.0, 0.0])).ravel()
p_unit_eci /= np.linalg.norm(p_unit_eci)
q_unit_eci /= np.linalg.norm(q_unit_eci)

for k in range(n_frames):
    sat_dat = dict(type="scatter3d", x=[r_eci[k, 0]], y=[r_eci[k, 1]], z=[r_eci[k, 2]])
    r_dat = dict(type="scatter3d", x=[0, r_eci[k, 0]], y=[0, r_eci[k, 1]], z=[0, r_eci[k, 2]])
    v_dat = dict(type="scatter3d",
                 x=[r_eci[k, 0], r_eci[k, 0] + v_eci[k, 0] * vscale],
                 y=[r_eci[k, 1], r_eci[k, 1] + v_eci[k, 1] * vscale],
                 z=[r_eci[k, 2], r_eci[k, 2] + v_eci[k, 2] * vscale])

    rvec = r_eci[k]
    x_p = np.dot(rvec, p_unit_eci)
    x_q = np.dot(rvec, q_unit_eci)
    nu_rad = np.arctan2(x_q, x_p)
    nu_deg = (np.degrees(nu_rad) + 360.0) % 360.0
    nu_array[k] = nu_deg

    sim_time_s = frame_times[k]

    ann = [
        dict(x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
             text=f"ν ≈ {nu_deg:.2f}°", font=dict(size=14), align="left"),
        dict(x=0.02, y=0.94, xref="paper", yref="paper", showarrow=False,
             text=f"RAAN = {raan_deg:.2f}°", font=dict(size=14), align="left"),
        dict(x=0.02, y=0.90, xref="paper", yref="paper", showarrow=False,
             text=f"t = {sim_time_s:.1f} s", font=dict(size=14), align="left"),
    ]

    frame_layout = {"annotations": ann}

    frame = go.Frame(data=[sat_dat, r_dat, v_dat],
                     name=str(k),
                     traces=[sat_index, r_index, v_index],
                     layout=frame_layout)
    frames.append(frame)

steps = []
for k in range(n_frames):
    step = {
        "args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                            "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(k),
        "method": "animate"
    }
    steps.append(step)

initial_annotations = [
    dict(x=0.01, y=0.93, xref="paper", yref="paper", showarrow=False, text=f"ν ≈ {nu_array[0]:.2f}°", font=dict(size=15)),
    dict(x=0.01, y=0.89, xref="paper", yref="paper", showarrow=False, text=f"RAAN = {raan_deg:.2f}°", font=dict(size=15)),
    dict(x=0.01, y=0.85, xref="paper", yref="paper", showarrow=False, text=f"t = {frame_times[0]:.1f} s", font=dict(size=15))
]

default_camera = dict(eye=dict(x=1.5, y=1.5, z=0.9))

layout = go.Layout(
    height=800,
    scene=dict(
        xaxis=dict(range=axis_range, visible=False),
        yaxis=dict(range=axis_range, visible=False),
        zaxis=dict(range=axis_range, visible=False),
        bgcolor="black",
        aspectmode="data",
        camera=default_camera
    ),
    template="plotly_dark",
    title=f"Elliptical Orbit (Lagrange f,g) — a={a:.0f} km, e={e:.3f}, i={inc_deg:.1f}°",
    margin=dict(l=0, r=0, t=60, b=0),
    showlegend=True,
    legend=dict(itemsizing="constant", orientation="h", y=1, x=0),
    updatemenus=[dict(
        type="buttons",
        direction="left",
        x=.08,
        y=0,
        showactive=False,
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]
            )
        ]
    )],
    sliders=[{
        "pad": {"b": 10, "t": 50},
        "currentvalue": {"prefix": "Frame: "},
        "steps": steps
    }],
    annotations=initial_annotations,
    uirevision=str(st.session_state.get("camera_reset_token", "init_token"))
)

fig = go.Figure(data=all_traces, frames=frames, layout=layout)

config = {"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToAdd": ["hoverClosest3d", "orbitRotation"], "responsive": True}
st.plotly_chart(fig, use_container_width=True, height=720, config=config)

st.markdown("---")
st.header("📘 Notes & Controls")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RAAN (deg)", f"{raan_deg:.3f}")
with col2:
    st.metric("True anomaly ν (deg) [frame 0]", f"{nu_array[0]:.3f}")
with col3:
    st.metric("Orbital period (s)", f"{T:.1f}")
