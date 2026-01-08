import streamlit as st
import base64
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Hyperspectral Unmixing", layout="wide")

# ---------- Helpers ----------
def find_image(name_base):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = Path(f"{name_base}{ext}")
        if p.exists():
            return p
    return None

def find_dataset_image(dataset, kind):
    dataset = dataset.lower()
    kind = kind.lower()

    if kind == "sign":
        kind_keywords = ["sign", "signature", "endmember"]
    else:
        kind_keywords = ["abund", "abundance"]

    for p in Path(".").iterdir():
        name = p.name.lower()
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue

        # dataset matching
        if dataset == "checkersboard":
            if not ("checker" in name or "board" in name):
                continue
        elif dataset == "synthetic":
            if not ("synthe" in name or "synth" in name):
                continue
        else:
            if dataset not in name:
                continue

        # kind matching
        if any(k in name for k in kind_keywords):
            return p

    return None

def centered_image(path, width=900):
    if path:
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            st.image(path, width=width)

# ---------- Dataset metrics ----------
DATASETS = {
    "urban": {"metrics": {"RMSE": 0.012, "SAD": 1.23, "SID": 0.045}},
    "jasper": {"metrics": {"RMSE": 0.015, "SAD": 1.35, "SID": 0.050}},
    "samson": {"metrics": {"RMSE": 0.010, "SAD": 1.10, "SID": 0.040}},
    "checkersboard": {"metrics": {"RMSE": 0.005, "SAD": 0.90, "SID": 0.030}},
    "synthetic": {"metrics": {"RMSE": 0.004, "SAD": 0.85, "SID": 0.025}},
}

ALIASES = {
    "urban": ["urban"],
    "jasper": ["jasper"],
    "samson": ["samson"],
    "checkersboard": ["checker", "checkers", "board"],
    "synthetic": ["synthe", "synth"],
}

# ---------- Logo ----------
logo = find_image("hsu-logo")
if logo:
    with open(logo, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin:20px 0;">
        <img src="data:image/png;base64,{logo_b64}" width="180">
    </div>
    """, unsafe_allow_html=True)

# ---------- Hero ----------
bg = find_image("hsu-back")
with open(bg, "rb") as f:
    bg64 = base64.b64encode(f.read()).decode()
mime = "image/jpeg" if bg.suffix in [".jpg", ".jpeg"] else "image/png"

st.markdown(f"""
<style>
.hero {{
    height: 80vh;
    background: url("data:{mime};base64,{bg64}") center/cover no-repeat;
    display:flex; align-items:center; justify-content:center;
}}
.hero-content {{ color:white; text-align:center; }}
.hero-content h1 {{ font-size:4rem; }}
.hero-content p {{ font-size:1.5rem; }}
.badge {{ padding:4px 10px; border-radius:12px; background:#2563eb; color:white; font-size:0.75rem; }}
.pub-row {{ padding:10px; border:1px solid #1f2937; border-radius:8px; margin-bottom:8px; }}
</style>
<div class="hero">
    <div class="hero-content">
        <h1>Hyperspectral Unmixing</h1>
        <p>See Beyond RGB</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Content ----------
st.markdown("---")
st.header("What is Hyperspectral Imaging")
st.write("Hyperspectral imaging allows us to see beyond RGB, capturing hundreds of spectral bands at each pixel instead of just red, green, and blue. This rich spectral detail creates a unique signature for every material, enabling precise identification and quantification of what’s present even when it's invisible to the human eye or standard cameras. From agriculture and mineral exploration to urban monitoring and disaster response, hyperspectral imaging turns complex spectral data into meaningful insights about the world around us.")
centered_image(find_image("HSI"))

st.markdown("---")
st.header("Hyperspectral Data")
st.write("A hyperspectral data cube is a 3D block of information where each pixel holds a complete light spectrum instead of just red, green, and blue values. Imagine it as a 3D stack of images, where each layer shows how the scene looks at a specific wavelength. The two spatial dimensions (X and Y) represent the image, while the third dimension (Z) spans hundreds of spectral bands. This structure gives each pixel a rich spectral fingerprint, allowing precise identification of materials and their abundance even when they look identical in normal color images.")
centered_image(find_image("hsu-cube-data"))

st.markdown("---")
st.header("What is Hyperspectral Unmixing")
st.write("Hyperspectral Unmixing is the process of decomposing mixed pixels in hyperspectral images into their underlying material spectra, known as endmembers, and estimating their corresponding abundance fractions. Since each pixel often records the combined reflectance of multiple materials, HSU is formulated as an inverse ill-posed problem. To obtain reliable solutions, algorithms typically apply constraints such as non-negativity and abundance sum-to-one, while modern approaches also leverage spatial information, machine learning, and deep learning to improve accuracy and robustness across real-world datasets.")
centered_image(find_image("hsu-unmix"))

st.markdown("---")
st.header("Applications of HSU")
st.markdown("""
1. Environmental Monitoring  
2. Precision Agriculture  
3. Mineral Mapping  
4. Urban Land Cover & Change  
5. Disaster Assessment  
""")

st.markdown("---")
st.header("Proposed Model")
st.write("In our framework, a 2D-CNN encoder extracts spatial–spectral features from the data, while an attention mechanism highlights the most informative ones, suppressing noise and redundancy. This selective focus helps the network learn more discriminative material signatures and produce cleaner abundance maps. By combining convolutional feature extraction with attention, the model not only improves accuracy but also enhances robustness when dealing with complex real-world hyperspectral scenes.")
centered_image(find_image("Architecture diagram attention"), width=1000)

# ---------- Results ----------
st.markdown("---")
st.header("Results")

uploaded = st.file_uploader("Upload dataset (.mat or .npy)", type=["mat", "npy"])

if uploaded:
    name = uploaded.name.lower()
    matched = None

    for dataset, keys in ALIASES.items():
        if any(k in name for k in keys):
            matched = dataset
            break

    if matched:
        st.subheader("Endmember Signatures")
        centered_image(find_dataset_image(matched, "sign"), width=800)

        st.subheader("Abundance Maps")
        centered_image(find_dataset_image(matched, "abund"), width=800)

        st.subheader("Metrics")
        df = pd.DataFrame.from_dict(DATASETS[matched]["metrics"], orient="index", columns=["Value"])
        st.table(df)
    else:
        st.error("Unknown dataset. Filename must contain urban, jasper, samson, checker, board, or synth.")

# ---------- Presentations ----------
st.markdown("---")
st.header("Presentations & Accepted Papers")

presentations = [
    ("Accepted", "Accepted at IEEE IGARSS 2025 — Brisbane, Australia", "IGARSS acceptance.pdf"),
    ("Presented", "Presented at INSPECT 2025 — IIT Gwalior, India", "inspect-presented.pdf"),
    ("Accepted", "Accepted at IEEE India GRSS 2025 — IIT Bhubaneswar, India", "ingarss-presented.jpeg"),
]

for status, title, pdf in presentations:
    st.markdown(f"""
    <div class="pub-row">
        <span class="badge">{status}</span>
        &nbsp; {title} — <a href="{pdf}" target="_blank">PDF</a>
    </div>
    """, unsafe_allow_html=True)

# ---------- Proof Images ----------
st.markdown("---")
st.header("Proof of Acceptance / Presentation")

row1 = st.columns(2)
with row1[0]:
    st.image(find_image("IG25_InvitationLetter_P1614_A2812_1745314323_page-0001"), caption="IGARSS 2025 Invitation", use_container_width=True)
with row1[1]:
    st.image(find_image("ingarss-presented"), caption="IEEE India GRSS Acceptance", use_container_width=True)

row2 = st.columns(2)
with row2[0]:
    st.image(find_image("inspect-presented_page-0001"), caption="INSPECT 2025 Presentation (Page 1)", use_container_width=True)
with row2[1]:
    st.image(find_image("inspect-presented_page-0002"), caption="INSPECT 2025 Presentation (Page 2)", use_container_width=True)

# ---------- Project Team ----------
st.markdown("---")
st.header("Project Team")

st.markdown("""
<style>
.team-card {
    background: linear-gradient(180deg, #0b1220, #020617);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.team-card img {
    border-radius: 12px;
    width: 140px;
    height: 140px;
    object-fit: cover;
    margin-bottom: 12px;
}
.team-name {
    font-size: 1.2rem;
    font-weight: 600;
    color: white;
}
.team-desc {
    font-size: 0.95rem;
    color: #cbd5f5;
    margin: 8px 0 12px 0;
}
.team-btn {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 8px;
    background: #1d4ed8;
    color: white;
    text-decoration: none;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

cols = st.columns(3)

team = [
    ("rutwik", "Rutwik S Nainoor", "Deep Learning Researcher · Neural Network Architecture, Case Studies & Optimization", "https://www.linkedin.com/in/rutwiksn"),
    ("shreya", "Shreya S Kamalapurkar", "Research Associate · Literature Review, Data Collection & Preprocessing, UI Design", "https://www.linkedin.com/in/shreya-kamalapurkar-a20922317 "),
    ("guide", "Dr. Vijayashekhar S S", "Research Supervisor & Project Guide · Associate Professor & Head, Dept. of AIML", "https://www.linkedin.com/in/dr-vijayashekhar-s-sankannanavar-a25648192 "),
]

for col, (img, name, desc, link) in zip(cols, team):
    img_path = find_image(img)
    img_html = ""
    if img_path:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_html = f'<img src="data:image/png;base64,{img_b64}">'

    col.markdown(f"""
    <div class="team-card">
        {img_html}
        <div class="team-name">{name}</div>
        <div class="team-desc">{desc}</div>
        <a class="team-btn" href="{link}" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
