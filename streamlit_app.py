import streamlit as st
import faiss
import json
import numpy as np
import os
import boto3
import botocore
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
DB_DIR          = "vector_db"
S3_BUCKET       = "mythdatabase"
S3_PREFIX       = "stories"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RESULTS_PER_PAGE = 10
MAX_PAGES = 50

# === AWS S3 CLIENT ===
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets.get("AWS_DEFAULT_REGION", None),
)

@st.cache_resource
def load_index():
    os.makedirs(DB_DIR, exist_ok=True)
    for fname in ["stories.index", "metadata.json"]:
        local_path = os.path.join(DB_DIR, fname)
        if not os.path.exists(local_path):
            try:
                s3.download_file(S3_BUCKET, fname, local_path)
            except botocore.exceptions.ClientError as e:
                st.error(f"Error downloading {fname} from S3: {e}")
                raise

    index = faiss.read_index(os.path.join(DB_DIR, "stories.index"))
    with open(os.path.join(DB_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, model

def fetch_story_from_s3(continent, culture, filename):
    key = f"{S3_PREFIX}/{continent}/{culture}/{filename}"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read().decode("utf-8")
    except botocore.exceptions.ClientError:
        return None

# === UI ===
st.title("🔍 Mythology Story Search")
st.markdown("Search across folklore stories by meaning, species, culture, or emotion.")

index, metadata, model = load_index()

# sidebar filters
all_continents = sorted({m["continent"] for m in metadata if "continent" in m})
all_cultures   = sorted({m["culture"]   for m in metadata if "culture"   in m})
all_species    = sorted({sp for m in metadata if "species_mentions" in m for sp in m["species_mentions"].keys()})

with st.sidebar:
    selected_continent = st.selectbox("🌍 Filter by Continent", ["All"] + all_continents, key="continent_select")
    selected_culture   = st.selectbox("🏛️ Filter by Culture",   ["All"] + all_cultures,   key="culture_select")
    selected_creature  = st.selectbox("🦊 Filter by Creature",  ["All"] + all_species,    key="creature_select")

query = st.text_input("Use the sidebar on the left to filter further by creature, continent, or culture.", key="query_input")

# Session state for pagination
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

def go_next(total_pages):
    if st.session_state.current_page < total_pages:
        st.session_state.current_page += 1

def go_prev():
    if st.session_state.current_page > 1:
        st.session_state.current_page -= 1

if query:
    q_vec = model.encode([query])
    scores, indices = index.search(np.array(q_vec), k=len(metadata))
    scored_items = [(score, idx) for score, idx in zip(scores[0], indices[0])]

    # Apply filters
    filtered_results = []
    for score, idx in scored_items:
        story = metadata[idx]

        if selected_continent != "All" and story.get("continent") != selected_continent:
            continue
        if selected_culture != "All" and story.get("culture") != selected_culture:
            continue
        if selected_creature != "All":
            if "species_mentions" not in story or selected_creature not in story["species_mentions"]:
                continue

        filtered_results.append((score, idx))

    total_results = len(filtered_results)
    total_pages = min(MAX_PAGES, max(1, (total_results - 1) // RESULTS_PER_PAGE + 1))

    start_idx = (st.session_state.current_page - 1) * RESULTS_PER_PAGE
    end_idx = start_idx + RESULTS_PER_PAGE

    st.subheader("🧠 Matching Stories")
    st.markdown(f"**Total Matches:** {total_results} &nbsp;&nbsp; | &nbsp;&nbsp; Page **{st.session_state.current_page}** of **{total_pages}**")

    for i, (score, idx) in enumerate(filtered_results[start_idx:end_idx], start=start_idx + 1):
        story = metadata[idx]

        st.markdown(f"#### {i}. 📜 {story['filename']}")
        st.markdown(
            f"*Culture:* **{story['culture']}**   |   "
            f"*Continent:* **{story['continent']}**"
        )
        st.markdown(f"*Similarity Score:* `{score:.2f}`")

        if "emotion_pred_top3" in story:
            emotions = ", ".join(e["label"] for e in story["emotion_pred_top3"])
            st.markdown(f"**Top Emotions:** {emotions}")

        if "species_mentions" in story and story["species_mentions"]:
            species_list = list(story["species_mentions"].keys())
            st.markdown(f"**Species Mentioned:** {', '.join(species_list)}")

        if st.button("📖 Read story", key=f"read_{idx}_{i}"):
            full_text = fetch_story_from_s3(story["continent"], story["culture"], story["filename"])
            if full_text:
                st.text_area("Full Story", full_text, height=400)
            else:
                st.error("Couldn't load story from S3.")

        st.markdown("---")

    # === PAGINATION CONTROLS AT BOTTOM ===
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_page <= 1):
            go_prev()
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages):
            go_next(total_pages)
