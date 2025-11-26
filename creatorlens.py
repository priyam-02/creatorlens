"""
CreatorLens - Streamlit App
Brand-Creator Matching System

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import torch
import clip
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import faiss
import ssl
import urllib.request
import cv2
import tempfile
import os

# Fix SSL certificate verification for macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Set page config
st.set_page_config(
    page_title="CreatorLens - Creator Matching",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CORE CLASSES & FUNCTIONS
# ============================================================================

@dataclass
class CreatorProfile:
    """Single creator profile."""
    name: str
    video_path: str
    embedding: np.ndarray
    style: Dict
    pacing: Dict

class CreatorDatabase:
    """Database of creator profiles with FAISS search."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.creators = []
        self.index = None
        
    def add_creator(self, profile: Dict, creator_name: str = None):
        """Add a creator profile to database."""
        if creator_name is None:
            creator_name = f"Creator_{len(self.creators)+1}"
        
        creator = CreatorProfile(
            name=creator_name,
            video_path=profile['video_path'],
            embedding=profile['video_embedding'],
            style=profile['visual_style'],
            pacing=profile['pacing']
        )
        
        self.creators.append(creator)
        
    def build_index(self):
        """Build FAISS index for fast similarity search."""
        if len(self.creators) == 0:
            return
        
        embeddings = np.vstack([c.embedding for c in self.creators]).astype('float32')
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[CreatorProfile, float]]:
        """Search for similar creators."""
        if self.index is None:
            self.build_index()
        
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, min(top_k, len(self.creators)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.creators[idx], float(dist)))
        
        return results

class BrandBrief:
    """Represents a brand's creator requirements."""
    
    def __init__(
        self,
        style_preferences: List[str],
        pacing_preference: str = None,
        content_type: str = None,
        clip_model=None,
        device='cuda'
    ):
        self.style_preferences = style_preferences
        self.pacing_preference = pacing_preference
        self.content_type = content_type
        self.clip_model = clip_model
        self.device = device
        
    def to_embedding(self) -> np.ndarray:
        """Convert brand brief to CLIP embedding."""
        text_parts = []
        text_parts.extend(self.style_preferences)
        
        if self.pacing_preference:
            text_parts.append(f"{self.pacing_preference} pacing")
        
        if self.content_type:
            text_parts.append(self.content_type)
        
        combined_text = ", ".join(text_parts)
        
        with torch.no_grad():
            text_tokens = clip.tokenize([combined_text]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features.cpu().numpy()[0]

def match_brand_to_creators(
    brand_brief: BrandBrief,
    database: CreatorDatabase,
    top_k: int = 5
) -> List[Dict]:
    """Match a brand brief to creators using multi-factor scoring."""
    brand_embedding = brand_brief.to_embedding()
    candidates = database.search(brand_embedding, top_k=top_k * 2)
    
    scored_matches = []
    
    for creator, clip_similarity in candidates:
        # Factor 1: CLIP similarity
        clip_score = clip_similarity
        
        # Factor 2: Style match
        style_score = 0.0
        for pref in brand_brief.style_preferences:
            if pref.lower() in creator.style['primary'].lower():
                style_score = 1.0
                break
            elif pref.lower() in creator.style.get('secondary', '').lower():
                style_score = 0.7
        
        # Factor 3: Pacing match
        pacing_score = 0.5  # Neutral default
        if brand_brief.pacing_preference and brand_brief.pacing_preference != "Any":
            if brand_brief.pacing_preference in creator.pacing['category']:
                pacing_score = 1.0
            else:
                brand_pacing = brand_brief.pacing_preference
                creator_pacing = creator.pacing['category']
                if (brand_pacing == "fast" and "fast" in creator_pacing) or \
                   (brand_pacing == "slow" and "slow" in creator_pacing):
                    pacing_score = 1.0
                elif "moderate" in brand_pacing or "moderate" in creator_pacing:
                    pacing_score = 0.5
                else:
                    pacing_score = 0.0
        
        # Weighted final score
        final_score = (
            clip_score * 0.5 +
            style_score * 0.3 +
            pacing_score * 0.2
        )
        
        scored_matches.append({
            'creator': creator,
            'overall_score': final_score,
            'clip_similarity': clip_score,
            'style_match': style_score,
            'pacing_match': pacing_score
        })
    
    scored_matches.sort(key=lambda x: x['overall_score'], reverse=True)
    return scored_matches[:top_k]

def create_synthetic_profile(name: str, style: str, pacing: str) -> Dict:
    """Create a synthetic creator profile for demo."""
    base_embedding = np.random.randn(512).astype('float32')
    base_embedding = base_embedding / np.linalg.norm(base_embedding)

    return {
        'video_path': f"synthetic_{name}.mp4",
        'video_embedding': base_embedding,
        'visual_style': {
            'primary': style,
            'secondary': 'vibrant and energetic',
            'scores': {style: 0.25}
        },
        'pacing': {
            'category': pacing,
            'mean_change': 0.30 if pacing == 'fast/dynamic' else 0.20
        }
    }

def analyze_style_with_clip(frame_embeddings: np.ndarray, clip_model, device) -> List[Dict]:
    """Analyze visual style using CLIP text-image similarity.

    Args:
        frame_embeddings: Array of frame embeddings (N x 512)
        clip_model: Loaded CLIP model
        device: torch device (cpu/cuda)

    Returns:
        List of dicts with 'style' and 'score', sorted by score descending
    """
    style_concepts = [
        "bright and colorful",
        "dark and moody",
        "minimalist and clean",
        "vibrant and energetic",
        "professional and polished",
        "casual and authentic",
        "cinematic and dramatic",
        "playful and fun"
    ]

    # Encode style concepts
    with torch.no_grad():
        text_tokens = clip.tokenize(style_concepts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features.cpu().numpy()

    # Compute similarities between frames and style concepts
    similarities = frame_embeddings @ text_features.T
    mean_scores = similarities.mean(axis=0)

    # Sort by score
    sorted_indices = np.argsort(mean_scores)[::-1]

    results = []
    for idx in sorted_indices:
        results.append({
            'style': style_concepts[idx],
            'score': float(mean_scores[idx])
        })

    return results

def analyze_pacing_from_embeddings(embeddings: np.ndarray) -> Dict:
    """Analyze video pacing from frame embeddings.

    Args:
        embeddings: Array of frame embeddings (N x 512)

    Returns:
        Dict with pacing metrics (mean_change, max_change, category, num_transitions)
    """
    changes = []
    for i in range(1, len(embeddings)):
        similarity = np.dot(embeddings[i-1], embeddings[i])
        changes.append(1 - similarity)

    mean_change = np.mean(changes) if changes else 0
    max_change = np.max(changes) if changes else 0
    num_transitions = int(np.sum(np.array(changes) > 0.3)) if changes else 0

    # Determine pacing category
    if mean_change < 0.15:
        pacing_category = "slow/static"
    elif mean_change < 0.25:
        pacing_category = "moderate"
    else:
        pacing_category = "fast/dynamic"

    return {
        'mean_change': float(mean_change),
        'max_change': float(max_change),
        'category': pacing_category,
        'num_transitions': num_transitions
    }

def extract_video_embedding(video_path: str, clip_model, preprocess, device, num_frames: int = 16) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Extract CLIP embedding from video by sampling frames.

    Args:
        video_path: Path to video file
        clip_model: Loaded CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device (cpu/cuda)
        num_frames: Number of frames to sample evenly from video

    Returns:
        Tuple of (video_embedding, style_results, pacing_metrics)
        - video_embedding: 512-dim averaged embedding
        - style_results: List of dicts with style scores
        - pacing_metrics: Dict with pacing analysis
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    # Calculate frame indices to sample evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frame_embeddings = []

    for frame_idx in frame_indices:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess and encode
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        frame_embeddings.append(image_features.cpu().numpy()[0])

    cap.release()

    if len(frame_embeddings) == 0:
        raise ValueError("Could not extract any frames from video")

    # Convert to array
    frame_embeddings_array = np.vstack(frame_embeddings)

    # Average all frame embeddings for video-level representation
    video_embedding = np.mean(frame_embeddings_array, axis=0).astype('float32')

    # Renormalize
    video_embedding = video_embedding / np.linalg.norm(video_embedding)

    # Analyze style using CLIP
    style_results = analyze_style_with_clip(frame_embeddings_array, clip_model, device)

    # Analyze pacing from embeddings
    pacing_metrics = analyze_pacing_from_embeddings(frame_embeddings_array)

    return video_embedding, style_results, pacing_metrics

def convert_style_results_to_profile_format(style_results: List[Dict], pacing_metrics: Dict) -> Tuple[Dict, Dict]:
    """Convert style/pacing results to CreatorProfile format.

    Args:
        style_results: List of dicts from analyze_style_with_clip
        pacing_metrics: Dict from analyze_pacing_from_embeddings

    Returns:
        Tuple of (visual_style dict, pacing dict) for CreatorProfile
    """
    # Primary style is the top-scoring one
    primary_style = style_results[0]['style']
    secondary_style = style_results[1]['style'] if len(style_results) > 1 else 'auto-detected'

    # Build scores dict from top 3
    scores = {}
    for result in style_results[:3]:
        scores[result['style']] = result['score']

    visual_style = {
        'primary': primary_style,
        'secondary': secondary_style,
        'scores': scores,
        'all_scores': style_results  # Keep full ranking
    }

    pacing = {
        'category': pacing_metrics['category'],
        'mean_change': pacing_metrics['mean_change'],
        'max_change': pacing_metrics['max_change'],
        'num_transitions': pacing_metrics['num_transitions']
    }

    return visual_style, pacing

# ============================================================================
# INITIALIZATION (Cached for Performance)
# ============================================================================

@st.cache_resource
def load_clip_model():
    """Load CLIP model (cached to avoid reloading)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

@st.cache_resource
def initialize_database():
    """Initialize creator database with synthetic profiles."""
    db = CreatorDatabase(embedding_dim=512)
    
    synthetic_creators = [
        ("TechReviewPro", "professional and polished", "moderate"),
        ("CookingQueen", "bright and colorful", "fast/dynamic"),
        ("FitnessGuru", "vibrant and energetic", "fast/dynamic"),
        ("VlogDaily", "casual and authentic", "moderate"),
        ("ArtisticCreator", "cinematic and dramatic", "slow/static"),
        ("MinimalDesign", "minimalist and clean", "moderate"),
        ("ComedyShorts", "playful and fun", "fast/dynamic"),
        ("NatureDoc", "dark and moody", "slow/static"),
        ("FashionIcon", "professional and polished", "moderate"),
        ("GameStreamer", "vibrant and energetic", "fast/dynamic")
    ]
    
    for name, style, pacing in synthetic_creators:
        profile = create_synthetic_profile(name, style, pacing)
        db.add_creator(profile, creator_name=name)
    
    db.build_index()
    return db

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🎯 CreatorLens</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Brand-Creator Matching System</p>', unsafe_allow_html=True)
    
    # Load models and database
    with st.spinner("Loading AI models..."):
        clip_model, preprocess, device = load_clip_model()

    # Use session state for database to persist uploads
    if 'database' not in st.session_state:
        st.session_state.database = initialize_database()

    db = st.session_state.database

    # Sidebar for inputs
    st.sidebar.header("🎨 Brand Requirements")
    
    # Style preferences
    st.sidebar.subheader("Visual Style")
    primary_style = st.sidebar.selectbox(
        "Primary Style",
        [
            "professional and polished",
            "bright and colorful",
            "minimalist and clean",
            "vibrant and energetic",
            "casual and authentic",
            "cinematic and dramatic",
            "playful and fun",
            "dark and moody"
        ],
        help="The visual aesthetic you want for your brand content. Examples: 'bright and colorful' = vibrant, saturated visuals; 'dark and moody' = low-key lighting, dramatic tones"
    )
    
    secondary_style = st.sidebar.selectbox(
        "Secondary Style (Optional)",
        ["None", "minimalist and clean", "vibrant and energetic", "casual and authentic"]
    )
    
    # Pacing preference
    st.sidebar.subheader("Content Pacing")
    pacing = st.sidebar.radio(
        "Preferred Pacing",
        ["Any", "fast", "moderate", "slow"],
        help="Content speed/tempo. Fast = quick cuts, dynamic; Moderate = balanced; Slow = static shots, deliberate pacing"
    )
    
    # Content type
    st.sidebar.subheader("Content Type")
    content_type = st.sidebar.text_input(
        "Type (Optional)",
        placeholder="e.g., tech product, cooking, fitness"
    )
    
    # Number of results
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5
    )

    # Search button
    search_button = st.sidebar.button("🔍 Find Creators", type="primary")

    st.sidebar.divider()

    # Video upload section
    st.sidebar.header("📤 Add Creator")
    st.sidebar.subheader("Upload Video")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a creator video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a video to add a new creator to the database"
    )

    # Upload guidelines
    with st.sidebar.expander("💡 Upload Tips", expanded=False):
        st.markdown("""
        **For best results:**
        - ✅ Use videos that represent the creator's typical style
        - ✅ 10-60 seconds works best
        - ✅ Good lighting and clear visuals improve accuracy
        - ✅ Multiple cuts/scenes help analyze pacing
        - ✅ Supported formats: MP4, MOV, AVI, MKV
        """)

    creator_name = st.sidebar.text_input(
        "Creator Name",
        placeholder="e.g., TechReviewer123",
        help="Optional: Provide a name for this creator"
    )

    upload_button = st.sidebar.button("➕ Add Creator to Database")

    # Process uploaded video
    if upload_button and uploaded_file is not None:
        with st.spinner("Processing video... This may take a moment."):
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Extract embedding and analyze (single pass)
                video_embedding, style_results, pacing_metrics = extract_video_embedding(
                    tmp_path, clip_model, preprocess, device
                )

                # Convert to profile format
                visual_style, pacing_info = convert_style_results_to_profile_format(
                    style_results, pacing_metrics
                )

                # Create profile
                profile = {
                    'video_path': uploaded_file.name,
                    'video_embedding': video_embedding,
                    'visual_style': visual_style,
                    'pacing': pacing_info
                }

                # Add to database
                creator_display_name = creator_name if creator_name else f"Creator_{len(db.creators)+1}"
                db.add_creator(profile, creator_name=creator_display_name)
                db.build_index()

                # Clean up temp file
                os.unlink(tmp_path)

                st.sidebar.success(f"✅ Added {creator_display_name}!")

                # Display top 3 style scores
                st.sidebar.write("**🎨 Top Styles:**")
                for i, result in enumerate(style_results[:3], 1):
                    st.sidebar.write(f"{i}. {result['style']} ({result['score']:.3f})")

                # Display pacing metrics
                st.sidebar.write(f"**⚡ Pacing:** {pacing_info['category']}")
                st.sidebar.write(f"- Mean change: {pacing_info['mean_change']:.3f}")
                st.sidebar.write(f"- Transitions: {pacing_info['num_transitions']}")

            except Exception as e:
                st.sidebar.error(f"❌ Error processing video: {str(e)}")

                # Clean up temp file if it exists
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # Main content area
    if search_button:
        # Prepare brand brief
        styles = [primary_style]
        if secondary_style != "None":
            styles.append(secondary_style)
        
        brand = BrandBrief(
            style_preferences=styles,
            pacing_preference=pacing if pacing != "Any" else None,
            content_type=content_type if content_type else None,
            clip_model=clip_model,
            device=device
        )
        
        # Find matches
        with st.spinner("Matching creators..."):
            matches = match_brand_to_creators(brand, db, top_k=top_k)
        
        # Display results
        st.header("🏆 Top Matching Creators")

        # Scoring methodology explanation
        with st.expander("📐 How We Calculate Match Scores", expanded=False):
            st.markdown("""
            ### Overall Score Formula:
            ```
            Overall Score = (CLIP × 50%) + (Style × 30%) + (Pacing × 20%)
            ```

            #### 🤖 CLIP Similarity (50% weight)
            Our AI analyzes video frames and compares them semantically to your brand description. CLIP (Contrastive Language-Image Pre-training) understands both images and text, allowing it to measure how well a creator's visual content matches what you're looking for.
            - **Higher score** = Creator's content visually aligns with your brand preferences
            - **Range:** 0.0 to 1.0 (cosine similarity of embeddings)

            #### 🎨 Style Match (30% weight)
            Direct comparison of visual aesthetics using 8 predefined categories:
            - bright and colorful, dark and moody, minimalist and clean, vibrant and energetic, professional and polished, casual and authentic, cinematic and dramatic, playful and fun
            - **1.0** = Perfect primary style match
            - **0.7** = Secondary style match
            - **0.0** = No style overlap

            #### ⚡ Pacing Match (20% weight)
            Compares content tempo by analyzing frame-to-frame changes in the video:
            - **fast/dynamic** = Quick cuts, high visual change (>0.25 mean change)
            - **moderate** = Balanced pacing (0.15-0.25 mean change)
            - **slow/static** = Deliberate, static shots (<0.15 mean change)
            - **1.0** = Exact pacing match | **0.5** = Neutral/unspecified | **0.0** = Opposite pacing

            ---
            **Score Interpretation:**
            - 🟢 **0.80 - 1.00**: Excellent match - Highly recommended
            - 🟡 **0.60 - 0.79**: Good match - Strong alignment
            - 🟠 **0.40 - 0.59**: Fair match - Moderate alignment
            - 🔴 **0.00 - 0.39**: Poor match - Low alignment
            """)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Creators Analyzed", len(db.creators))
        with col2:
            st.metric("Top Match Score", f"{matches[0]['overall_score']:.3f}")
        with col3:
            st.metric("Results Returned", len(matches))

        st.divider()
        
        # Display each match
        for i, match in enumerate(matches, 1):
            creator = match['creator']
            
            with st.expander(f"#{i} {creator.name} - Score: {match['overall_score']:.3f}", expanded=(i == 1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 Match Breakdown")
                    
                    # Score breakdown
                    score_data = {
                        'Factor': ['Overall', 'CLIP Similarity', 'Style Match', 'Pacing Match'],
                        'Score': [
                            match['overall_score'],
                            match['clip_similarity'],
                            match['style_match'],
                            match['pacing_match']
                        ],
                        'Weight': ['-', '50%', '30%', '20%'],
                        'Contribution': [
                            '-',
                            f"{match['clip_similarity'] * 0.5:.3f}",
                            f"{match['style_match'] * 0.3:.3f}",
                            f"{match['pacing_match'] * 0.2:.3f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(score_data), hide_index=True, use_container_width=True)
                
                with col2:
                    st.subheader("👤 Creator Profile")
                    st.write(f"**Name:** {creator.name}")
                    st.write(f"**Primary Style:** {creator.style['primary']}")

                    # Show top 3 styles if available
                    if 'all_scores' in creator.style:
                        st.write("**Style Rankings:**")
                        for idx, style_info in enumerate(creator.style['all_scores'][:3], 1):
                            st.write(f"  {idx}. {style_info['style']} ({style_info['score']:.3f})")
                    elif 'scores' in creator.style:
                        st.write("**Style Scores:**")
                        for style, score in list(creator.style['scores'].items())[:3]:
                            st.write(f"  - {style}: {score:.3f}")

                    st.write(f"**Pacing:** {creator.pacing['category']}")

                    # Show additional pacing metrics if available
                    if 'num_transitions' in creator.pacing:
                        st.write(f"  - Transitions: {creator.pacing['num_transitions']}")
                        st.write(f"  - Mean change: {creator.pacing['mean_change']:.3f}")

                    # Score indicator with interpretation
                    if match['overall_score'] >= 0.8:
                        st.success("🟢 Excellent Match (0.80-1.00)")
                        st.caption("Highly recommended - Strong alignment across all factors")
                    elif match['overall_score'] >= 0.6:
                        st.info("🟡 Good Match (0.60-0.79)")
                        st.caption("Strong alignment - Good fit for your brand")
                    elif match['overall_score'] >= 0.4:
                        st.warning("🟠 Fair Match (0.40-0.59)")
                        st.caption("Moderate alignment - Consider specific factors")
                    else:
                        st.error("🔴 Poor Match (0.00-0.39)")
                        st.caption("Low alignment - May not be the best fit")
        
        # Detailed comparison table
        st.divider()
        st.subheader("📋 Comparison Table")
        
        comparison_data = []
        for i, match in enumerate(matches, 1):
            creator = match['creator']
            comparison_data.append({
                'Rank': i,
                'Creator': creator.name,
                'Overall Score': f"{match['overall_score']:.3f}",
                'CLIP': f"{match['clip_similarity']:.3f}",
                'Style': f"{match['style_match']:.3f}",
                'Pacing': f"{match['pacing_match']:.3f}",
                'Creator Style': creator.style['primary'],
                'Creator Pacing': creator.pacing['category']
            })
        
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
        
    else:
        # Initial state - show instructions
        st.info("👈 Configure your brand requirements in the sidebar and click 'Find Creators' to start!")

        # Welcome section with detailed explanation
        st.subheader("🎓 How CreatorLens Works")

        st.markdown("""
        CreatorLens uses **AI-powered video analysis** to match brands with content creators based on visual style and content characteristics.
        Our system analyzes actual video content, not just metadata, to find the best creative partnerships.
        """)

        # Quick explainer on CLIP
        with st.expander("🤖 What is CLIP AI?", expanded=False):
            st.markdown("""
            **CLIP (Contrastive Language-Image Pre-training)** is an AI model from OpenAI that understands both images and text together.

            **Why it matters for matching:**
            - Traditional systems only compare tags or keywords
            - CLIP actually "sees" and understands visual content
            - It can match "vibrant and energetic" (text) to actual video frames showing bright colors and movement

            **How we use it:**
            1. We extract frames from creator videos
            2. CLIP converts each frame into a 512-dimensional "embedding" (numeric representation)
            3. Your brand preferences are also converted to embeddings
            4. We measure similarity using cosine distance (how close the vectors are)
            5. Higher similarity = better visual match

            **Example:** If you request "professional and polished", CLIP will score higher for videos with clean compositions, good lighting, and formal aesthetics - even if those exact words aren't in any metadata.
            """)

        st.divider()

        # Show example flow
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 1️⃣ Define Your Brand
            **What you do:**
            - Choose visual style preferences
              - *Example: "bright and colorful" for energetic brands*
            - Select content pacing
              - *Example: "fast" for social media ads*
            - Optionally describe content type
              - *Example: "tech product reviews"*

            **What happens:**
            - Your preferences are converted into an AI embedding
            - This creates a semantic "fingerprint" of your brand vision
            """)

        with col2:
            st.markdown("""
            ### 2️⃣ AI Analyzes Videos
            **What we do:**
            - Extract 16 frames from each creator video
            - Use CLIP AI to understand visual content
            - Analyze style using 8 aesthetic categories
            - Measure pacing via frame-to-frame changes

            **The matching factors:**
            - 🤖 **CLIP Similarity (50%)** - Semantic visual match
            - 🎨 **Style Match (30%)** - Aesthetic alignment
            - ⚡ **Pacing Match (20%)** - Tempo compatibility
            """)

        with col3:
            st.markdown("""
            ### 3️⃣ Get Ranked Matches
            **What you see:**
            - Top creators ranked by overall score
            - Transparent score breakdowns
            - Detailed creator profiles with:
              - Top 3 style rankings
              - Pacing metrics
              - Transition counts

            **Score ranges:**
            - 🟢 0.80+ = Excellent match
            - 🟡 0.60-0.79 = Good match
            - 🟠 0.40-0.59 = Fair match
            - 🔴 <0.40 = Poor match
            """)
        
        # Example use case
        st.divider()

        with st.expander("💡 Example: How a Fitness Brand Uses CreatorLens", expanded=False):
            st.markdown("""
            ### Scenario
            A fitness supplement brand wants to find creators for an Instagram campaign promoting their new energy drink.

            ### Their Input
            - **Primary Style:** "vibrant and energetic" (matches their brand aesthetic)
            - **Pacing:** "fast" (for short-form social content)
            - **Content Type:** "fitness and workout videos"

            ### What CreatorLens Does
            1. **CLIP Analysis:** Compares their preferences against creator video frames
               - Looks for energetic visuals, bright lighting, dynamic movement
            2. **Style Matching:** Identifies creators with "vibrant and energetic" aesthetics
               - Filters out "dark and moody" or "minimalist" styles
            3. **Pacing Analysis:** Finds creators with quick cuts and high visual change
               - Matches the fast-paced social media style they need

            ### Results
            - **Top Match (0.87):** FitnessGuru - vibrant style, fast pacing, perfect fit
            - **Good Match (0.72):** GameStreamer - energetic but moderate pacing
            - **Fair Match (0.53):** VlogDaily - casual style, different pacing

            ### Outcome
            The brand contacts FitnessGuru for partnership, confident in the visual alignment based on data, not guesswork.
            """)

        # Show database stats
        st.divider()
        st.subheader("📊 Creator Database")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.metric("Total Creators", len(db.creators))
            
            styles_count = {}
            for creator in db.creators:
                style = creator.style['primary']
                styles_count[style] = styles_count.get(style, 0) + 1
            
            st.write("**Styles Distribution:**")
            for style, count in sorted(styles_count.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {style}: {count}")
        
        with stats_col2:
            pacing_count = {}
            for creator in db.creators:
                pacing = creator.pacing['category']
                pacing_count[pacing] = pacing_count.get(pacing, 0) + 1
            
            st.write("**Pacing Distribution:**")
            for pacing, count in sorted(pacing_count.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {pacing}: {count}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>CreatorLens</strong> - Built with Streamlit, CLIP, and FAISS</p>
        <p>Multi-factor AI matching system for creator-brand partnerships</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
