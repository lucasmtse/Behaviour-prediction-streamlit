# to run: python -m streamlit run \\TENIBRE\bs\users\Lucas\Tracking\app.py
# NOTE: modify DEST_FOLDER as needed
# app.py
import streamlit as st
import tempfile
from pathlib import Path
import base64, json
import pandas as pd
import deeplabcut

# ==== NEW imports for features & model ====
import os, re
import numpy as np
from itertools import combinations
import xgboost as xgb  

DEST_FOLDER = Path(r"\\TENIBRE\bs\projects\ethofearless\behavior\video for Deeplabcut tests\Tracking\DLC\Streamlit_out")
st.set_page_config(page_title="Video + Behavior + DLC", layout="centered")
st.title("🎬 Video + Behavior Overlay (single player)")


# def load_and_flatten(h5_path: Path) -> pd.DataFrame:
#     df = pd.read_hdf(h5_path)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = ["_".join([str(x) for x in t if str(x) != ""]) for t in df.columns.to_list()]
#     return df

def build_behavior_map_from_csv(csv_df: pd.DataFrame, fps_if_no_time: int = 30) -> dict:
    """
    From a behavior CSV (with 'time' or 'frame', 'flight', 'stret/streght/stret posture'), 
    build a mapping second -> label (str).
    """
    df = csv_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect column of behavior "stret/stret posture"
    stret_candidates = [
        c for c in df.columns if ("posture" in c and any(k in c for k in ["stret", "streght", "stret"]))
    ]
    stret_col = stret_candidates[0] if stret_candidates else None

    # time -> seconds
    if "time" not in df.columns:
        if "frame" in df.columns:
            df["time"] = df["frame"].astype(float) / float(fps_if_no_time)
        else:
            raise ValueError("CSV must have 'time' (seconds) or 'frame'.")

    def row_label(row):
        flight = int(row.get("flight", 0)) if "flight" in df.columns else 0
        stret = int(row.get(stret_col, 0)) if stret_col else 0
        if flight and stret: return "flight + stret posture"
        if flight: return "flight"
        if stret: return "stret posture"
        if not flight and not stret: return 'no behavior'
        return ""

    df["label"] = df.apply(row_label, axis=1)
    df["sec"] = df["time"].astype(float).round().astype(int)

    # Pour chaque seconde, choisir le label non-vide le plus fréquent
    sec_map = {}
    for sec, s in df.groupby("sec")["label"]:
        if (s != "").any():
            label = s[s != ""].value_counts().idxmax()
        else:
            label = ""
        sec_map[int(sec)] = label
    return sec_map

def video_to_data_url(video_path: Path) -> str:
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"

def show_player_with_behavior(video_src: str,
                              sec_map_gt: dict | None,
                              sec_map_pred: dict | None,
                              no_behavior_text: str = "no behavior"):
    """
    Single video player with TWO cards:
      - Ground truth (always neutral gray)
      - Prediction (green if matches GT, red if different)
    """
    sec_map_gt_json = json.dumps({int(k): v for k, v in (sec_map_gt or {}).items()})
    sec_map_pr_json = json.dumps({int(k): v for k, v in (sec_map_pred or {}).items()})

    html = f"""
    <div style="max-width:1000px;margin:0 auto;font-family:ui-sans-serif,system-ui;">
      <style>
        .card {{
          padding: 10px 14px;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          min-width: 220px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06);
          transition: background .15s ease, color .15s ease, border-color .15s ease;
        }}
        .label {{ font-size: 12px; opacity: 0.7; margin-bottom: 2px; }}
        .value {{ font-weight: 700; font-size: 16px; }}

        .ok {{
          background:#10B981;  /* green */
          color:#ffffff;
          border-color:#10B981;
        }}
        .bad {{
          background:#EF4444;  /* red */
          color:#ffffff;
          border-color:#EF4444;
        }}
        .neutral {{
          background:#F3F4F6;  /* gray */
          color:#111827;
          border-color:#e5e7eb;
        }}

        .timechip {{
          padding: 8px 12px; border:1px solid #e5e7eb; border-radius: 12px; font-family:monospace;
        }}
      </style>

      <video id="vid" src="{video_src}" style="width:100%;border-radius:12px;" preload="metadata" controls></video>

      <div style="display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap;">
        <div class="card neutral" id="cardGT">
          <div class="label">Ground truth</div>
          <div id="behaviorGT" class="value">—</div>
        </div>
        <div class="card neutral" id="cardPR">
          <div class="label">Prediction</div>
          <div id="behaviorPR" class="value">—</div>
        </div>
        <div class="timechip"><span id="time">00:00 / 00:00</span></div>
      </div>
    </div>

    <script>
      const v = document.getElementById('vid');
      const timeBox = document.getElementById('time');

      const cardPR = document.getElementById('cardPR');
      const boxGT  = document.getElementById('behaviorGT');
      const boxPR  = document.getElementById('behaviorPR');

      const secMapGT = {sec_map_gt_json};
      const secMapPR = {sec_map_pr_json};

      const hasGTMap = Object.keys(secMapGT).length > 0;
      const hasPRMap = Object.keys(secMapPR).length > 0;

      function fmt(t){{
        if(!isFinite(t)) return "00:00";
        const m=Math.floor(t/60), s=Math.floor(t%60);
        return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
      }}

      function norm(x){{
        const s = (x||'').toLowerCase().trim();
        const hasFlight = s.includes('flight');
        const hasstret = s.includes('stret') || s.includes('streght') || s.includes('stret');
        if (hasFlight && hasstret) return 'flight + stret posture';
        if (hasFlight) return 'flight';
        if (hasstret) return 'stret posture';
        return '';
      }}

      function setState(el, state) {{
        if (!el) return;
        el.classList.remove('ok','bad','neutral');
        el.classList.add(state);
      }}

      function updateUI(){{
        const cur=v.currentTime||0, dur=v.duration||0;
        timeBox.textContent = fmt(cur)+' / '+fmt(dur);
        const sec = Math.floor(cur);

        const rawGT = hasGTMap ? (secMapGT.hasOwnProperty(sec) ? (secMapGT[sec]||'') : '') : '';
        const rawPR = hasPRMap ? (secMapPR.hasOwnProperty(sec) ? (secMapPR[sec]||'') : '') : '';

        const a = norm(rawGT);
        const b = norm(rawPR);

        boxGT.textContent = a || "{no_behavior_text}";
        boxPR.textContent = b || "{no_behavior_text}";

        if (!hasPRMap) {{
          setState(cardPR, 'neutral');
        }} else {{
          if (a === b) {{
            setState(cardPR, 'ok');
          }} else {{
            setState(cardPR, 'bad');
          }}
        }}
      }}

      v.addEventListener('timeupdate',updateUI);
      v.addEventListener('loadedmetadata',updateUI);
      v.addEventListener('ratechange',updateUI);
      v.addEventListener('seeked',updateUI);
      v.addEventListener('play',()=>updateUI());
      v.addEventListener('pause',updateUI);
      setInterval(updateUI,200);
    </script>
    """
    st.components.v1.html(html, height=600, scrolling=False)

# ==== NEW: helpers for feature extraction =====================================
def read_tracking_csv_flexible(file_or_path):
    """
    Read a multi-index (3/4 levels) or already flattened DLC CSV and return a DataFrame
    with flattened columns. Cleans the large SuperAnimal prefix if present.
    """
    df = None
    for header_levels in ([0,1,2,3], [0,1,2], None):
        try:
            if header_levels is None:
                df = pd.read_csv(file_or_path)
                df.columns = [str(c).strip() for c in df.columns]
            else:
                df = pd.read_csv(file_or_path, header=header_levels)
                df.columns = ['_'.join([str(x) for x in tup]).strip() for tup in df.columns.values]
            break
        except Exception:
            df = None
    if df is None:
        raise ValueError("Unable to read CSV (multi-index or flat).")

    if 'scorer_individuals_bodyparts_coords' in df.columns:
        df = df.drop(columns=['scorer_individuals_bodyparts_coords'])

    # Delete prefix "superanimal_..._animal1_" if present
    pat = re.compile(r"^superanimal_[^_]+_fasterrcnn_[^_]+_fpn_[^_]+_[^_]+_animal\d+_")
    df.columns = [re.sub(pat, "", c) for c in df.columns]
    return df

def fill_short_gaps(arr: np.ndarray, max_gap: int = 15) -> np.ndarray:
    """Interpolate short gaps (NaN) in 1D array, up to max_gap length."""
    a = arr.astype(float).copy()
    n = a.size
    isn = ~np.isfinite(a)
    if not isn.any():
        return a
    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        start = i
        while i < n and isn[i]:
            i += 1
        end = i - 1
        L = end - start + 1
        if (L <= max_gap and start > 0 and end < n - 1
            and np.isfinite(a[start - 1]) and np.isfinite(a[end + 1])):
            left_val = a[start - 1]
            right_val = a[end + 1]
            a[start:end + 1] = np.linspace(left_val, right_val, L + 2)[1:-1]
    return a

def last_prev_valid_index(valid_mask: np.ndarray) -> np.ndarray:
    """Index of last previous valid (True) in valid_mask, or NaN if none."""
    idx = np.arange(valid_mask.size)
    return pd.Series(np.where(valid_mask, idx, np.nan)).ffill().shift(1).to_numpy()

def curated_bones(bps):
    S = set(bps)
    cand = [
        ("left_ear", "right_ear"),
        ("nose", "neck"),
        ("neck", "mid_back"),
        ("mid_back", "tail_base"),
        ("left_shoulder", "left_midside"),
        ("left_midside", "left_hip"),
        ("right_shoulder", "right_midside"),
        ("right_midside", "right_hip"),
        ("nose", "head_midpoint"),
        ("mid_back", "mouse_center"),
    ]
    tail = [bp for bp in ["tail_base","tail1","tail2","tail3","tail4","tail5","tail_end"] if bp in S]
    cand += list(zip(tail[:-1], tail[1:]))
    mb = [bp for bp in ["mid_back","mid_backend","mid_backend2","mid_backend3"] if bp in S]
    cand += list(zip(mb[:-1], mb[1:]))
    return [(a,b) for (a,b) in cand if a in S and b in S]

def extract_features_from_tracking_df(df_in: pd.DataFrame, FPS: float, MAX_GAP: int,
                                      bones_mode="auto",
                                      compute_angle_unsigned=True,
                                      compute_angle_signed=True):
    """
    Pipeline extract:
      - pairing *_x/_y
      - gap filling
      - distances toutes paires
      - vitesses par bodypart
      - bones (auto/all) + speed midpoint + angle deltas
    Retourne feat_df (DataFrame)
    """
    df = df_in.copy() 

    # *_x / *_y
    cols = [str(c) for c in df.columns]
    x_cols = [c for c in cols if c.endswith("_x")]
    prefixes = [c[:-2] for c in x_cols if (c[:-2] + "_y") in cols]
    if not prefixes:
        raise ValueError("No *_x / *_y columns found. Provide a DLC-like tracking CSV (flattened).")

    # Clean bodyparts names (remove common prefix)
    tokenized = [p.split("_") for p in prefixes]
    common_prefix_len = 0
    min_len = min(len(t) for t in tokenized)
    for i in range(min_len):
        tok = tokenized[0][i]
        if all(t[i] == tok for t in tokenized):
            common_prefix_len += 1
        else:
            break

    bodyparts, prefix_map, seen = [], {}, set()
    for t in tokenized:
        bp = "_".join(t[common_prefix_len:])
        base = "_".join(t)
        if bp and bp not in seen:
            seen.add(bp)
            bodyparts.append(bp)
            prefix_map[bp] = base

    B = len(bodyparts)
    T = len(df)

    # Gap fill for coords (-1 -> NaN)
    for bp, base in prefix_map.items():
        for coord in (f"{base}_x", f"{base}_y"):
            s = pd.to_numeric(df[coord], errors="coerce")
            s = s.mask(s == -1, np.nan)
            s = pd.Series(fill_short_gaps(s.to_numpy(), max_gap=MAX_GAP))
            df.loc[:, coord] = s.values

    # Tensor coords (T,B,2)
    coords = np.empty((T, B, 2), dtype=np.float32)
    for j, bp in enumerate(bodyparts):
        base = prefix_map[bp]
        coords[:, j, 0] = pd.to_numeric(df[f"{base}_x"], errors="coerce").to_numpy()
        coords[:, j, 1] = pd.to_numeric(df[f"{base}_y"], errors="coerce").to_numpy()

    # Distances (all pairs)
    iu = np.triu_indices(B, k=1)
    diff = coords[:, iu[0], :] - coords[:, iu[1], :]
    D = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)
    dist_cols = [f"dist__{bodyparts[i]}__{bodyparts[j]}" for i, j in zip(iu[0], iu[1])]
    feat_df = pd.DataFrame(D, columns=dist_cols)
    feat_df.insert(0, "frame", np.arange(T, dtype=int))

    # Speed by bodypart
    idx = np.arange(T)
    for bp in bodyparts:
        base = prefix_map[bp]
        x = pd.to_numeric(df[f"{base}_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[f"{base}_y"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        pidx = last_prev_valid_index(valid)
        vel = np.full(T, np.nan, dtype=np.float32)
        rows = valid & ~np.isnan(pidx)
        if np.any(rows):
            prev = pidx[rows].astype(int)
            dt = (idx[rows] - prev).astype(float)
            dx = x[rows] - x[prev]
            dy = y[rows] - y[prev]
            vel[rows] = (np.sqrt(dx*dx + dy*dy) * (float(FPS) / dt)).astype(np.float32)
        vel[valid & np.isnan(pidx)] = 0.0
        feat_df[f"vel__{bp}"] = vel

    # Bones (auto/all)
    if bones_mode == "all":
        bones = list(combinations(bodyparts, 2))
    elif bones_mode == "auto":
        bones = curated_bones(bodyparts)
    else:
        bones = [(a,b) for (a,b) in bones_mode if a in bodyparts and b in bodyparts]

    for (bp0, bp1) in bones:
        i0 = bodyparts.index(bp0)
        i1 = bodyparts.index(bp1)
        p0 = coords[:, i0, :]
        p1 = coords[:, i1, :]
        valid_now = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)

        mid = (p0 + p1) / 2.0
        pidx = last_prev_valid_index(valid_now)

        # speed at midpoint
        speed = np.full(T, np.nan, dtype=np.float32)
        rows = valid_now & ~np.isnan(pidx)
        if np.any(rows):
            prev = pidx[rows].astype(int)
            dt = (idx[rows] - prev).astype(float)
            dp = mid[rows] - mid[prev]
            speed[rows] = (np.linalg.norm(dp, axis=1) * (float(FPS) / dt)).astype(np.float32)
        speed[valid_now & np.isnan(pidx)] = 0.0
        feat_df[f"bone_speed_mid__{bp0}__{bp1}"] = speed

        # angle delta (unsigned / signed)
        v_now = p1 - p0
        norm_now = np.linalg.norm(v_now, axis=1)
        has_prev = ~np.isnan(pidx)
        ok_now = valid_now & has_prev & (norm_now > 0)

        if compute_angle_unsigned:
            ang_unsigned = np.full(T, np.nan, dtype=np.float32)
        if compute_angle_signed:
            ang_signed = np.full(T, np.nan, dtype=np.float32)

        if np.any(ok_now):
            prev = pidx[ok_now].astype(int)
            v_prev = v_now[prev]
            norm_prev = np.linalg.norm(v_prev, axis=1)
            both_ok_mask = norm_prev > 0
            sel_idx = np.where(ok_now)[0][both_ok_mask]
            if sel_idx.size > 0:
                vn = v_now[sel_idx]
                vp = v_prev[both_ok_mask]
                if compute_angle_unsigned:
                    cosang = (vn * vp).sum(axis=1) / (np.linalg.norm(vn, axis=1) * np.linalg.norm(vp, axis=1))
                    ang_unsigned[sel_idx] = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))).astype(np.float32)
                if compute_angle_signed:
                    theta_now  = np.degrees(np.arctan2(vn[:,1], vn[:,0]))
                    theta_prev = np.degrees(np.arctan2(vp[:,1], vp[:,0]))
                    dtheta = theta_now - theta_prev
                    dtheta = (dtheta + 180.0) % 360.0 - 180.0
                    ang_signed[sel_idx] = dtheta.astype(np.float32)

        if compute_angle_unsigned:
            feat_df[f"bone_angle_delta_unsigned__{bp0}__{bp1}_deg"] = ang_unsigned
        if compute_angle_signed:
            feat_df[f"bone_angle_delta_signed__{bp0}__{bp1}_deg"] = ang_signed

    return feat_df


# 1) Upload VIDEO (top)

st.subheader("1) Upload video")
uploaded_video = st.file_uploader("Upload an MP4 video", type=["mp4"], key="video")
if not uploaded_video:
    st.stop()

tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_video.read())
video_path = Path(tfile.name)
st.caption(f"Saved to: {video_path}")

# 2) Upload CSVs (middle)

st.divider()
st.subheader("2) Upload behavior CSVs")
col_csv1, col_csv2 = st.columns(2)
with col_csv1:
    uploaded_csv_gt = st.file_uploader("Ground-truth CSV (frame/time + flight + stret/streght/stret posture)", type=["csv"], key="behavior_csv_gt")
with col_csv2:
    uploaded_csv_pred = st.file_uploader("Prediction CSV (optional; same schema)", type=["csv"], key="behavior_csv_pred")

fps_if_no_time = st.number_input("FPS if 'time' is missing", min_value=1, value=30)

sec_map_gt = None
sec_map_pred = None
# GT
if uploaded_csv_gt:
    try:
        behavior_df_gt = pd.read_csv(uploaded_csv_gt)
        sec_map_gt = build_behavior_map_from_csv(behavior_df_gt, fps_if_no_time=fps_if_no_time)
        st.success("✅ Ground-truth CSV loaded.")
    except Exception as e:
        st.error(f"GT CSV error: {e}")
# Prediction
if uploaded_csv_pred:
    try:
        behavior_df_pred = pd.read_csv(uploaded_csv_pred)
        sec_map_pred = build_behavior_map_from_csv(behavior_df_pred, fps_if_no_time=fps_if_no_time)
        st.success("✅ Prediction CSV loaded.")
    except Exception as e:
        st.error(f"Prediction CSV error: {e}")

# ---- Render ONE player (with GT and/or Prediction boxes if present)
video_src = video_to_data_url(video_path)
show_player_with_behavior(video_src, sec_map_gt, sec_map_pred, no_behavior_text="no behavior")


# 3) DLC section (bottom)

st.divider()
st.subheader("3) DeepLabCut SuperAnimal (optional)")
with st.expander("Run DLC on the uploaded video"):
    if st.button("Run DeepLabCut SuperAnimal"):
        st.info("Running inference... ⏳")

        dest_folder = DEST_FOLDER
        dest_folder.mkdir(parents=True, exist_ok=True)

        superanimal_name = "superanimal_topviewmouse"
        scale_list = list(range(100, 250, 25))

        deeplabcut.video_inference_superanimal(
            [str(video_path)],
            superanimal_name,
            model_name="resnet_50",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            scale_list=scale_list,
            video_adapt=False,
            dest_folder=str(dest_folder),
            max_individuals=1,
            pseudo_threshold=0.1,
            bbox_threshold=0.9,
            detector_epochs=4,
            pose_epochs=4,
        )
        st.success(f"✅ Inference finished. Results saved in {dest_folder}")

        h5_files = list(dest_folder.glob("*.h5"))
        if not h5_files:
            st.error("❌ No .h5 file found after inference.")
        else:
            for h5 in h5_files:
                csv_path = h5.with_suffix(".csv")
                # df = load_and_flatten(h5)
                df = pd.read_hdf(h5)
                df.to_csv(csv_path, index=True)
                st.success(f"→ Converted {h5.name} to {csv_path.name}")
                st.dataframe(df.head())
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {csv_path.name}",
                        data=f,
                        file_name=csv_path.name,
                        mime="text/csv",
                    )


# 4) Feature extraction (tracking CSV -> engineered features)

st.divider()
st.subheader("4) Extract features from tracking CSV")

fx_csv = st.file_uploader("Tracking CSV (DLC flatten: *_x, *_y, *_likelihood optionnel)", type=["csv"], key="feat_csv")

col_fx1, col_fx2, col_fx3 = st.columns(3)
with col_fx1:
    FX_FPS = st.number_input("FPS", min_value=1, value=30)
with col_fx2:
    MAX_GAP = st.number_input("Max gap fill (frames)", min_value=0, value=15)
with col_fx3:
    bones_mode = st.selectbox("Bones", ["auto", "all"], index=0)

col_fx4, col_fx5 = st.columns(2)
with col_fx4:
    compute_unsigned = st.checkbox("Angle delta unsigned", value=True)
with col_fx5:
    compute_signed   = st.checkbox("Angle delta signed", value=True)

# stocker feat_df dans la session pour la section 5
if "feat_df" not in st.session_state:
    st.session_state.feat_df = None

if st.button("🚀 Extract features", disabled=fx_csv is None):
    if fx_csv is None:
        st.warning("Upload a tracking CSV first.")
    else:
        try:
            with st.spinner("Reading & processing..."):
                df_trk = read_tracking_csv_flexible(fx_csv)
                feat_df = extract_features_from_tracking_df(
                    df_trk,
                    FPS=float(FX_FPS),
                    MAX_GAP=int(MAX_GAP),
                    bones_mode=bones_mode,
                    compute_angle_unsigned=compute_unsigned,
                    compute_angle_signed=compute_signed
                )

            st.success(f"✅ Features computed. Shape: {feat_df.shape}")
            st.dataframe(feat_df.head())
            st.session_state.feat_df = feat_df  # <-- dispo pour la section 5

            tmp_out = Path(tempfile.gettempdir()) / f"dlc_features_gapfill{MAX_GAP}.csv"
            feat_df.to_csv(tmp_out, index=False)
            with open(tmp_out, "rb") as f:
                st.download_button(
                    "⬇️ Download features CSV",
                    data=f,
                    file_name=tmp_out.name,
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Feature extraction failed: {e}")


# 5) Apply trained model (XGBoost) to features -> predicted behaviors CSV

st.divider()
st.subheader("5) 🤖 Apply trained model (XGBoost) to features")

col_m1, col_m2 = st.columns([2,1])
with col_m1:
    model_path = st.text_input("Model path (.json)", value=r"D:\Lucas\Tracking\DLC\Streamlit_out\model\xgb_behavior_model.json")
with col_m2:
    out_fps = st.number_input("FPS for output 'time' column", min_value=1.0, value=22.35, help="time = frame / FPS")

st.caption("Use features from Section 4 (above) or upload an existing features CSV:")

col_src1, col_src2 = st.columns(2)
with col_src1:
    use_session_feat = st.checkbox("Use features from Section 4", value=st.session_state.feat_df is not None)
with col_src2:
    uploaded_feat_csv = st.file_uploader("Or upload features CSV (must include 'frame' + numeric columns)", type=["csv"], key="feat_apply_csv")

if st.button("🚀 Predict behaviors", disabled=(not use_session_feat and uploaded_feat_csv is None)):
    try:
        # 1) Load features
        if use_session_feat and (st.session_state.feat_df is not None):
            feat = st.session_state.feat_df.copy()
        else:
            if uploaded_feat_csv is None:
                st.warning("Provide features (use Section 4 or upload a CSV).")
                st.stop()
            feat = pd.read_csv(uploaded_feat_csv)

        # 2) Select numeric columns (excluding 'frame')
        num_cols = [c for c in feat.columns if c != "frame" and pd.api.types.is_numeric_dtype(feat[c])]
        if not num_cols:
            st.error("No numeric feature columns found (besides 'frame').")
            st.stop()

        # 3) Load XGBoost model from .json
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            st.stop()

        booster = xgb.Booster()
        booster.load_model(model_path)

        # 4) Prediction (DMatrix)
        full_X = feat[num_cols].astype(float).values
        dmat = xgb.DMatrix(full_X)
        full_y_pred = booster.predict(dmat)

        
        import numpy as _np
        if full_y_pred.ndim == 2 and full_y_pred.shape[1] > 1:
            pred_labels = _np.argmax(full_y_pred, axis=1)
        else:
            # binaire/label direct
            pred_labels = _np.rint(full_y_pred).astype(int)  

        feat["predicted_behavior"] = pred_labels.astype(int)

        # 5) If any NaN in features, set predicted_behavior to 0 (no behavior)
        has_nan = feat[num_cols].isnull().any(axis=1)
        feat.loc[has_nan, "predicted_behavior"] = 0

        # 6) Build output DataFrame
        output_df = feat[["frame", "predicted_behavior"]].copy()
        output_df["stret posture"] = (output_df["predicted_behavior"] == 1).astype(int)
        output_df["flight"] = (output_df["predicted_behavior"] == 2).astype(int)
        # time from frame
        output_df["time"] = output_df["frame"] / float(out_fps)
        output_df = output_df[["frame", "time", "stret posture", "flight"]]

        st.success(f"✅ Predictions done. Rows: {len(output_df)}")
        st.dataframe(output_df.head())

        # 7) Save + Download
        out_name = "predicted_behaviors.csv"
        tmp_pred = Path(tempfile.gettempdir()) / out_name
        output_df.to_csv(tmp_pred, index=False)
        with open(tmp_pred, "rb") as f:
            st.download_button(
                "⬇️ Download predicted behaviors CSV",
                data=f,
                file_name=out_name,
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
