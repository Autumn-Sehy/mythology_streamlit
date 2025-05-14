import streamlit as st
import faiss
import json
import numpy as np
import os
import boto3
import botocore
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
DB_DIR          = "vector_db"            # local cache for index & metadata
S3_BUCKET       = "mythdatabase"         # your S3 bucket name
S3_PREFIX       = "stories"              # prefix where .txt files live
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 10

# === AWS S3 CLIENT (uses Streamlit secrets) ===
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets.get("AWS_DEFAULT_REGION", None),
)

@st.cache_resource
def load_index():
    # ensure local cache folder
    os.makedirs(DB_DIR, exist_ok=True)
    # download vector DB files if missing
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

# sidebar filters --> I want to move these eventually :)
all_continents = sorted({m["continent"] for m in metadata if "continent" in m})
all_cultures   = sorted({m["culture"]   for m in metadata if "culture"   in m})
all_species    = sorted({sp for m in metadata if "species_mentions" in m for sp in m["species_mentions"].keys()})

with st.sidebar:
    selected_continent = st.selectbox("🌍 Filter by Continent", ["All"] + all_continents, key="continent_select")
    selected_culture   = st.selectbox("🏛️ Filter by Culture",   ["All"] + all_cultures,   key="culture_select")
    selected_creature  = st.selectbox("🦊 Filter by Creature",  ["All"] + all_species,    key="creature_select")

query = st.text_input(""Use the sidebar on the left to filter further by creature, continent, or culture."", key="query_input")

if query:
    q_vec  = model.encode([query])
    scores, indices = index.search(np.array(q_vec), k=TOP_K)

    st.subheader("🧠 Top Matching Stories")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        story = metadata[idx]

        # apply filters
        if selected_continent != "All" and story.get("continent") != selected_continent:
            continue
        if selected_culture != "All" and story.get("culture") != selected_culture:
            continue
        if selected_creature != "All":
            if "species_mentions" not in story or selected_creature not in story["species_mentions"]:
                continue

        # display summary
        st.markdown(f"### 📜 {story['filename']}")
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

        # read story button :)
        if st.button("📖 Read story", key=f"read_{idx}_{i}"):
            full_text = fetch_story_from_s3(story["continent"], story["culture"], story["filename"])
            if full_text:
                st.text_area("Full Story", full_text, height=400)
            else:
                st.error("Couldn't load story from S3.")

        st.markdown("---")
