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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Fix SSL certificate verification for macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Set page config
st.set_page_config(
    page_title="CreatorLens - Creator Matching",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Modern Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Poppins:wght@600;700&display=swap');

    /* Root variables for theming - Professional Blue/Teal/Slate */
    :root {
        --primary-color: #0ea5e9;
        --secondary-color: #0284c7;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --gradient-1: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        --gradient-2: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        --gradient-3: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
    }

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main header with gradient text */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        font-family: 'Poppins', sans-serif;
        animation: fadeInDown 0.8s ease-out;
    }

    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadeIn 1s ease-out 0.2s both;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }

    /* Enhanced metrics */
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"] {
        min-height: 120px !important;
        height: 120px !important;
    }

    div[data-testid="stMetricValue"] {
        display: flex;
        align-items: center;
    }

    .stMetric:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* Enhanced buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
        transition: all 0.3s ease;
        font-size: 1rem;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.6);
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
    }

    .stButton>button:active {
        transform: translateY(0px);
    }

    /* Primary button variant */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
    }

    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
    }

    /* Progress bars with gradient */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(2, 132, 199, 0.1) 100%);
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(14, 165, 233, 0.2);
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(2, 132, 199, 0.2) 100%);
        transform: translateX(5px);
    }

    /* Success/Info/Warning/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        animation: slideIn 0.5s ease-out;
    }

    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0ea5e9 0%, #0284c7 100%);
    }

    /* Score badges */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 0.25rem;
    }

    .badge-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
    }

    .badge-good {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }

    .badge-fair {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4);
    }

    .badge-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4);
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.8;
        }
    }

    /* Loading spinner */
    .stSpinner > div {
        border-color: #0ea5e9 transparent #0284c7 transparent !important;
    }

    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0ea5e9, #0284c7, transparent);
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
    content_category: str = None  # Manual content category tag
    hook: Dict = None  # Hook analysis metrics
    cta: Dict = None  # CTA detection metrics
    engagement: Dict = None  # Engagement prediction

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
            pacing=profile['pacing'],
            content_category=profile.get('content_category'),
            hook=profile.get('hook'),
            cta=profile.get('cta'),
            engagement=profile.get('engagement')
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

        # Factor 4: Content category match
        content_score = 0.5  # Neutral default (no category specified)
        if brand_brief.content_type and brand_brief.content_type != "Any":
            if creator.content_category:
                if brand_brief.content_type == creator.content_category:
                    content_score = 1.0  # Exact match
                else:
                    content_score = 0.0  # Mismatch
            # else: creator has no category, keep neutral 0.5

        # Weighted final score (adjusted weights to include content)
        final_score = (
            clip_score * 0.45 +      # CLIP similarity (45%)
            style_score * 0.30 +      # Style match (30%)
            pacing_score * 0.20 +     # Pacing match (20%)
            content_score * 0.05      # Content category match (5%)
        )
        
        scored_matches.append({
            'creator': creator,
            'overall_score': final_score,
            'clip_similarity': clip_score,
            'style_match': style_score,
            'pacing_match': pacing_score,
            'content_match': content_score
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

def analyze_hook(video_path: str, clip_model, preprocess, device) -> Dict:
    """Analyze the first 3-5 seconds of video for hook effectiveness.

    Hook elements detected:
    - Eye contact / face presence
    - Text overlays
    - Bright colors / visual pop
    - Motion / action
    - Unexpected elements

    Args:
        video_path: Path to video file
        clip_model: Loaded CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device

    Returns:
        Dict with hook analysis results
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get frames from first 5 seconds (at 0s, 1s, 2.5s, 4s)
    hook_frames = []
    for frame_num in [0, int(fps * 1), int(fps * 2.5), int(fps * 4)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            hook_frames.append(pil_image)

    cap.release()

    if len(hook_frames) == 0:
        return {
            'hook_score': 0.0,
            'effectiveness': 'Unknown',
            'elements': []
        }

    # Hook concepts to detect
    hook_concepts = [
        "person looking at camera",
        "close-up face",
        "text on screen",
        "bright colorful scene",
        "fast movement",
        "surprising element",
        "eye-catching visual"
    ]

    # Encode hook frames
    embeddings = []
    with torch.no_grad():
        for pil_image in hook_frames:
            image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            features = clip_model.encode_image(image_tensor)
            features = F.normalize(features, dim=-1)
            embeddings.append(features)

    embeddings = torch.cat(embeddings)

    # Encode concepts
    text = clip.tokenize(hook_concepts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

    # Compute similarities
    similarities = (embeddings @ text_features.T).cpu().numpy()
    mean_scores = similarities.mean(axis=0)

    # Score hook effectiveness
    hook_score = float(mean_scores.max())
    hook_elements = [
        {'element': concept, 'score': float(score)}
        for concept, score in zip(hook_concepts, mean_scores)
    ]
    hook_elements.sort(key=lambda x: x['score'], reverse=True)

    # Effectiveness rating
    if hook_score > 0.28:
        effectiveness = "Excellent"
    elif hook_score > 0.24:
        effectiveness = "Good"
    elif hook_score > 0.20:
        effectiveness = "Moderate"
    else:
        effectiveness = "Weak"

    return {
        'hook_score': hook_score,
        'effectiveness': effectiveness,
        'elements': hook_elements[:3]  # Top 3 elements
    }

def detect_cta(video_path: str, clip_model, preprocess, device) -> Dict:
    """Detect call-to-action patterns in video ending.

    CTA types detected:
    - Follow/Subscribe requests
    - Link in bio
    - Product showcase
    - Like/Comment requests

    Args:
        video_path: Path to video file
        clip_model: Loaded CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device

    Returns:
        Dict with CTA detection results
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get frames from last 5 seconds
    cta_frames = []
    for offset in [int(fps * 5), int(fps * 2.5), 5]:
        frame_num = max(0, total_frames - offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            cta_frames.append(pil_image)

    cap.release()

    if len(cta_frames) == 0:
        return {
            'cta_detected': False,
            'cta_score': 0.0,
            'elements': []
        }

    # CTA concepts to detect
    cta_concepts = [
        "text on screen",
        "pointing gesture",
        "subscribe button",
        "product display",
        "person talking to camera",
        "logo or branding",
        "contact information"
    ]

    # Encode frames
    embeddings = []
    with torch.no_grad():
        for pil_image in cta_frames:
            image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            features = clip_model.encode_image(image_tensor)
            features = F.normalize(features, dim=-1)
            embeddings.append(features)

    embeddings = torch.cat(embeddings)

    # Encode CTA concepts
    text = clip.tokenize(cta_concepts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

    # Compute similarities
    similarities = (embeddings @ text_features.T).cpu().numpy()
    mean_scores = similarities.mean(axis=0)

    # Determine CTA presence
    cta_score = float(mean_scores.max())
    cta_detected = cta_score > 0.22

    cta_elements = [
        {'type': concept, 'score': float(score)}
        for concept, score in zip(cta_concepts, mean_scores)
    ]
    cta_elements.sort(key=lambda x: x['score'], reverse=True)

    return {
        'cta_detected': cta_detected,
        'cta_score': cta_score,
        'elements': cta_elements[:3]  # Top 3 elements
    }

def predict_engagement(hook_metrics: Dict, cta_metrics: Dict, pacing_metrics: Dict, style_results: List[Dict], video_duration: float = None) -> Dict:
    """Predict engagement score based on multiple factors.

    Args:
        hook_metrics: Hook analysis results
        cta_metrics: CTA detection results
        pacing_metrics: Pacing analysis results
        style_results: Style analysis results
        video_duration: Video length in seconds (optional)

    Returns:
        Dict with engagement prediction
    """
    # Factor 1: Hook Strength (30%)
    hook_score = hook_metrics['hook_score']
    hook_weight = 0.30

    # Factor 2: CTA Presence (15%)
    cta_score = 1.0 if cta_metrics['cta_detected'] else 0.3
    cta_weight = 0.15

    # Factor 3: Pacing Variety (20%)
    # Good pacing has transitions but not too chaotic
    num_transitions = pacing_metrics.get('num_transitions', 0)
    if 3 <= num_transitions <= 15:
        pacing_variety_score = 1.0
    elif num_transitions > 15:
        pacing_variety_score = 0.7  # Too chaotic
    else:
        pacing_variety_score = 0.5  # Too static
    pacing_weight = 0.20

    # Factor 4: Style Consistency (15%)
    # Primary style should dominate (high score difference)
    if len(style_results) >= 2:
        style_consistency = style_results[0]['score'] - style_results[1]['score']
        style_consistency_score = min(1.0, style_consistency * 3)  # Normalize
    else:
        style_consistency_score = 0.5
    style_weight = 0.15

    # Factor 5: Optimal Length (10%) - if duration provided
    if video_duration:
        pacing_cat = pacing_metrics['category']
        if pacing_cat == 'fast/dynamic' and 15 <= video_duration <= 60:
            length_score = 1.0
        elif pacing_cat == 'moderate' and 30 <= video_duration <= 120:
            length_score = 1.0
        elif pacing_cat == 'slow/static' and 60 <= video_duration <= 180:
            length_score = 1.0
        else:
            length_score = 0.6
    else:
        length_score = 0.7  # Neutral if unknown
    length_weight = 0.10

    # Factor 6: Transition Quality (10%)
    mean_change = pacing_metrics['mean_change']
    if 0.15 <= mean_change <= 0.30:
        transition_score = 1.0  # Smooth, dynamic
    elif mean_change > 0.30:
        transition_score = 0.6  # Too jarring
    else:
        transition_score = 0.7  # Too smooth/static
    transition_weight = 0.10

    # Calculate engagement score
    engagement_score = (
        hook_score * hook_weight +
        cta_score * cta_weight +
        pacing_variety_score * pacing_weight +
        style_consistency_score * style_weight +
        length_score * length_weight +
        transition_score * transition_weight
    )

    # Determine predicted performance
    if engagement_score >= 0.75:
        performance = "Excellent"
    elif engagement_score >= 0.60:
        performance = "Good"
    elif engagement_score >= 0.45:
        performance = "Fair"
    else:
        performance = "Poor"

    # Identify strengths
    strengths = []
    if hook_score > 0.26:
        strengths.append(f"Strong hook ({hook_metrics['effectiveness']})")
    if cta_metrics['cta_detected']:
        strengths.append("Clear CTA present")
    if 3 <= num_transitions <= 15:
        strengths.append("Good pacing variety")
    if style_consistency_score > 0.7:
        strengths.append("Consistent visual style")

    # Identify weaknesses
    weaknesses = []
    if hook_score < 0.20:
        weaknesses.append("Weak hook - needs stronger opening")
    if not cta_metrics['cta_detected']:
        weaknesses.append("No clear CTA detected")
    if num_transitions < 3:
        weaknesses.append("Too static - add more visual variety")
    elif num_transitions > 15:
        weaknesses.append("Too many transitions - may be jarring")

    return {
        'engagement_score': float(engagement_score),
        'predicted_performance': performance,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'factor_scores': {
            'hook': hook_score,
            'cta': cta_score,
            'pacing_variety': pacing_variety_score,
            'style_consistency': style_consistency_score,
            'length_appropriateness': length_score,
            'transition_quality': transition_score
        }
    }

def extract_video_embedding(video_path: str, clip_model, preprocess, device, num_frames: int = 16, analyze_full: bool = True) -> Tuple:
    """Extract CLIP embedding from video by sampling frames.

    Args:
        video_path: Path to video file
        clip_model: Loaded CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device (cpu/cuda)
        num_frames: Number of frames to sample evenly from video
        analyze_full: If True, also analyze hook, CTA, and engagement

    Returns:
        If analyze_full=True: (video_embedding, style_results, pacing_metrics, hook_metrics, cta_metrics, engagement_prediction)
        If analyze_full=False: (video_embedding, style_results, pacing_metrics)
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else None

    cap.release()

    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    # Extract frames for embedding
    cap = cv2.VideoCapture(video_path)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frame_embeddings = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        frame_embeddings.append(image_features.cpu().numpy()[0])

    cap.release()

    if len(frame_embeddings) == 0:
        raise ValueError("Could not extract any frames from video")

    frame_embeddings_array = np.vstack(frame_embeddings)

    # Average all frame embeddings
    video_embedding = np.mean(frame_embeddings_array, axis=0).astype('float32')
    video_embedding = video_embedding / np.linalg.norm(video_embedding)

    # Analyze style using CLIP
    style_results = analyze_style_with_clip(frame_embeddings_array, clip_model, device)

    # Analyze pacing from embeddings
    pacing_metrics = analyze_pacing_from_embeddings(frame_embeddings_array)

    if not analyze_full:
        return video_embedding, style_results, pacing_metrics

    # Full analysis: hook, CTA, engagement
    hook_metrics = analyze_hook(video_path, clip_model, preprocess, device)
    cta_metrics = detect_cta(video_path, clip_model, preprocess, device)
    engagement_prediction = predict_engagement(hook_metrics, cta_metrics, pacing_metrics, style_results, video_duration)

    return video_embedding, style_results, pacing_metrics, hook_metrics, cta_metrics, engagement_prediction

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

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_score_radar_chart(match_data: Dict) -> go.Figure:
    """Create an interactive radar chart for match scores."""
    categories = ['CLIP<br>Similarity', 'Style<br>Match', 'Pacing<br>Match', 'Content<br>Match']
    values = [
        match_data['clip_similarity'],
        match_data['style_match'],
        match_data['pacing_match'],
        match_data['content_match']
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Match Scores',
        line=dict(color='#0ea5e9', width=3),
        fillcolor='rgba(14, 165, 233, 0.3)',
        marker=dict(size=8, color='#0284c7')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
                gridcolor='rgba(14, 165, 233, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(14, 165, 233, 0.2)'
            )
        ),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12, color='#334155')
    )

    return fig

def create_engagement_gauge(engagement_score: float) -> go.Figure:
    """Create a gauge chart for engagement prediction."""
    # Determine color based on score
    if engagement_score >= 0.75:
        color = "#10b981"  # Green
    elif engagement_score >= 0.60:
        color = "#3b82f6"  # Blue
    elif engagement_score >= 0.45:
        color = "#f59e0b"  # Orange
    else:
        color = "#ef4444"  # Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=engagement_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 40, 'family': 'Poppins', 'color': color}},
        title={'text': "Engagement Score", 'font': {'size': 18, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 45], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [45, 60], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [60, 75], 'color': 'rgba(59, 130, 246, 0.1)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': engagement_score * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': '#334155'}
    )

    return fig

def create_database_stats_charts(db) -> tuple:
    """Create pie and bar charts for database statistics."""
    # Content category distribution
    category_count = {}
    style_count = {}

    for creator in db.creators:
        # Count categories
        cat = creator.content_category or "Uncategorized"
        category_count[cat] = category_count.get(cat, 0) + 1

        # Count styles
        style = creator.style['primary']
        style_count[style] = style_count.get(style, 0) + 1

    # Category pie chart
    cat_fig = go.Figure(data=[go.Pie(
        labels=list(category_count.keys()),
        values=list(category_count.values()),
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, family='Inter'),
        hovertemplate='<b>%{label}</b><br>Creators: %{value}<br>%{percent}<extra></extra>'
    )])

    cat_fig.update_layout(
        title=dict(text='Content Categories', font=dict(size=16, family='Poppins', color='#334155')),
        height=350,
        margin=dict(l=40, r=150, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )

    # Style bar chart
    style_fig = go.Figure(data=[go.Bar(
        x=list(style_count.values()),
        y=list(style_count.keys()),
        orientation='h',
        marker=dict(
            color=list(style_count.values()),
            colorscale=[[0, '#0ea5e9'], [1, '#0284c7']],
            line=dict(color='rgba(14, 165, 233, 0.3)', width=1)
        ),
        text=list(style_count.values()),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Creators: %{x}<extra></extra>'
    )])

    style_fig.update_layout(
        title=dict(text='Visual Styles', font=dict(size=16, family='Poppins', color='#334155')),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Number of Creators', showgrid=True, gridcolor='rgba(14, 165, 233, 0.1)'),
        yaxis=dict(title=''),
        showlegend=False
    )

    return cat_fig, style_fig

def create_style_bars(style_results: List[Dict], top_n: int = 3) -> go.Figure:
    """Create horizontal bar chart for style scores."""
    styles = [r['style'] for r in style_results[:top_n]]
    scores = [r['score'] for r in style_results[:top_n]]

    fig = go.Figure(data=[go.Bar(
        y=styles,
        x=scores,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, '#06b6d4'], [0.5, '#0284c7'], [1, '#0ea5e9']],
            line=dict(color='rgba(14, 165, 233, 0.3)', width=1),
            cornerradius=8
        ),
        text=[f'{s:.3f}' for s in scores],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
    )])

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(14, 165, 233, 0.1)', title=''),
        yaxis=dict(title=''),
        showlegend=False,
        font=dict(family='Inter', size=11, color='#334155')
    )

    return fig

@st.cache_resource
def initialize_database():
    """Initialize creator database with synthetic profiles."""
    db = CreatorDatabase(embedding_dim=512)
    
    synthetic_creators = [
        ("TechReviewPro", "professional and polished", "moderate", "Tech/Reviews"),
        ("CookingQueen", "bright and colorful", "fast/dynamic", "Cooking/Food"),
        ("FitnessGuru", "vibrant and energetic", "fast/dynamic", "Fitness/Wellness"),
        ("VlogDaily", "casual and authentic", "moderate", "Vlog/Lifestyle"),
        ("ArtisticCreator", "cinematic and dramatic", "slow/static", "Art/Creative"),
        ("MinimalDesign", "minimalist and clean", "moderate", "Education/Tutorial"),
        ("ComedyShorts", "playful and fun", "fast/dynamic", "Vlog/Lifestyle"),
        ("NatureDoc", "dark and moody", "slow/static", "Education/Tutorial"),
        ("FashionIcon", "professional and polished", "moderate", "Fashion/Beauty"),
        ("GameStreamer", "vibrant and energetic", "fast/dynamic", "Gaming")
    ]

    for name, style, pacing, category in synthetic_creators:
        profile = create_synthetic_profile(name, style, pacing)
        profile['content_category'] = category
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

    # Initialize mode in session state
    if 'mode' not in st.session_state:
        st.session_state.mode = 'welcome'  # Options: 'welcome', 'upload', 'match'

    db = st.session_state.database

    # ========== SIDEBAR: Mode-dependent rendering ==========

    if st.session_state.mode == 'welcome':
        # Welcome mode sidebar - show mode selection buttons
        st.sidebar.title("🎯 CreatorLens")
        st.sidebar.markdown("### Choose an action:")

        if st.sidebar.button("📤 Upload Creator Video", type="primary", use_container_width=True):
            st.session_state.mode = 'upload'
            st.rerun()

        st.sidebar.markdown("Upload a creator's video to analyze style, pacing, hook, CTA, and engagement metrics.")

        st.sidebar.divider()

        if st.sidebar.button("🔍 Match Creators", type="primary", use_container_width=True):
            st.session_state.mode = 'match'
            st.rerun()

        st.sidebar.markdown("Find creators that match your brand requirements using AI-powered analysis.")

    elif st.session_state.mode == 'upload':
        # Upload mode sidebar
        if st.sidebar.button("⬅️ Back to Home"):
            st.session_state.mode = 'welcome'
            st.rerun()

        st.sidebar.header("📤 Upload Creator Video")

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

        content_category = st.sidebar.selectbox(
            "Content Category",
            [
                "Cooking/Food",
                "Tech/Reviews",
                "Fitness/Wellness",
                "Gaming",
                "Fashion/Beauty",
                "Education/Tutorial",
                "Vlog/Lifestyle",
                "Business/Finance",
                "Art/Creative",
                "Other"
            ],
            help="Select the primary content category for this creator. This helps match them with relevant brand campaigns."
        )

        upload_button = st.sidebar.button("➕ Add Creator to Database", type="primary")

    elif st.session_state.mode == 'match':
        # Match mode sidebar
        if st.sidebar.button("⬅️ Back to Home"):
            st.session_state.mode = 'welcome'
            st.rerun()

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

        # Content category
        st.sidebar.subheader("Content Category")
        content_type = st.sidebar.selectbox(
            "Category",
            [
                "Any",
                "Cooking/Food",
                "Tech/Reviews",
                "Fitness/Wellness",
                "Gaming",
                "Fashion/Beauty",
                "Education/Tutorial",
                "Vlog/Lifestyle",
                "Business/Finance",
                "Art/Creative",
                "Other"
            ],
            help="Filter creators by content category. Selecting a specific category will prioritize creators in that niche."
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

    # Initialize variables for other modes
    if st.session_state.mode != 'match':
        primary_style = None
        secondary_style = None
        pacing = None
        content_type = None
        top_k = 5
        search_button = False

    if st.session_state.mode != 'upload':
        upload_button = False
        uploaded_file = None
        creator_name = None
        content_category = None

    # Process uploaded video
    if upload_button and uploaded_file is not None:
        with st.spinner("Processing video... Analyzing hook, CTA, and engagement..."):
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Full analysis including hook, CTA, and engagement
                video_embedding, style_results, pacing_metrics, hook_metrics, cta_metrics, engagement_prediction = extract_video_embedding(
                    tmp_path, clip_model, preprocess, device, analyze_full=True
                )

                # Convert to profile format
                visual_style, pacing_info = convert_style_results_to_profile_format(
                    style_results, pacing_metrics
                )

                # Create profile with all metrics
                profile = {
                    'video_path': uploaded_file.name,
                    'video_embedding': video_embedding,
                    'visual_style': visual_style,
                    'pacing': pacing_info,
                    'content_category': content_category,
                    'hook': hook_metrics,
                    'cta': cta_metrics,
                    'engagement': engagement_prediction
                }

                # Add to database
                creator_display_name = creator_name if creator_name else f"Creator_{len(db.creators)+1}"
                db.add_creator(profile, creator_name=creator_display_name)
                db.build_index()

                # Clean up temp file
                os.unlink(tmp_path)

                # Store analysis results in session state for display in main area
                st.session_state.upload_results = {
                    'creator_name': creator_display_name,
                    'style_results': style_results,
                    'pacing_info': pacing_info,
                    'content_category': content_category,
                    'hook_metrics': hook_metrics,
                    'cta_metrics': cta_metrics,
                    'engagement_prediction': engagement_prediction
                }

                st.rerun()  # Refresh to display results in main area

            except Exception as e:
                st.sidebar.error(f"❌ Error processing video: {str(e)}")

                # Clean up temp file if it exists
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # ========== MAIN CONTENT AREA: Mode-dependent rendering ==========

    if st.session_state.mode == 'welcome':
        # Welcome screen content with hero section
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(2, 132, 199, 0.05) 100%); border-radius: 16px; margin-bottom: 2rem;'>
            <h1 style='font-size: 2.5rem; background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0891b2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: Poppins; font-weight: 800; margin-bottom: 0.5rem;'>
                Welcome to CreatorLens
            </h1>
            <p style='font-size: 1.2rem; color: #64748b; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
                Advanced AI-powered video analysis to match brands with content creators.<br/>
                Analyze style, pacing, hooks, CTAs, and predict engagement—all automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Key Features Grid
        st.markdown("### ✨ Key Features")
        feat_col1, feat_col2, feat_col3 = st.columns(3)

        with feat_col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(2, 132, 199, 0.1) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #0ea5e9; height: 180px; display: flex; flex-direction: column; justify-content: flex-start;'>
                <h4 style='color: #0284c7; margin: 0 0 0.75rem 0; font-size: 1.1rem;'>🎨 Visual AI Analysis</h4>
                <p style='color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;'>CLIP-powered style detection across 8 aesthetic categories with semantic understanding</p>
            </div>
            """, unsafe_allow_html=True)

        with feat_col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(8, 145, 178, 0.1) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #06b6d4; height: 180px; display: flex; flex-direction: column; justify-content: flex-start;'>
                <h4 style='color: #0891b2; margin: 0 0 0.75rem 0; font-size: 1.1rem;'>🎣 Hook & CTA Detection</h4>
                <p style='color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;'>Analyze first 5 seconds for hook effectiveness and detect call-to-action patterns</p>
            </div>
            """, unsafe_allow_html=True)

        with feat_col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #10b981; height: 180px; display: flex; flex-direction: column; justify-content: flex-start;'>
                <h4 style='color: #059669; margin: 0 0 0.75rem 0; font-size: 1.1rem;'>📊 Engagement Prediction</h4>
                <p style='color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;'>Multi-factor scoring combining 6 metrics to predict content performance</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("🎓 How It Works")

        # Feature explainers
        col1, col2 = st.columns(2)

        with col1:
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

            with st.expander("🎣 What is Hook Detection?", expanded=False):
                st.markdown("""
                **Hook analysis** evaluates the first 3-5 seconds of a video - the most critical moment for viewer retention.

                **What we detect (7 hook elements):**
                - Person looking at camera (direct engagement)
                - Close-up face (personal connection)
                - Text on screen (immediate value prop)
                - Bright colorful scene (visual pop)
                - Fast movement (energy/urgency)
                - Surprising element (pattern interrupt)
                - Eye-catching visual (standout moment)

                **Effectiveness ratings:**
                - 🟢 **Excellent** (>0.28): Strong viewer retention expected
                - 🟡 **Good** (0.24-0.28): Above-average hook
                - 🟠 **Moderate** (0.20-0.24): Adequate opening
                - 🔴 **Weak** (<0.20): May lose viewers quickly

                **Why it matters:** 50% of viewers decide to keep watching in the first 3 seconds. A strong hook = better engagement.
                """)

        with col2:
            with st.expander("📢 What is CTA Detection?", expanded=False):
                st.markdown("""
                **CTA (Call-to-Action) detection** identifies engagement prompts in the last 5 seconds of videos.

                **What we detect (7 CTA types):**
                - Text on screen ("Subscribe", "Follow", etc.)
                - Pointing gestures (directing attention)
                - Subscribe buttons (visual elements)
                - Product displays (featured items)
                - Person talking to camera (verbal CTA)
                - Logo or branding (brand reinforcement)
                - Contact information (links, handles)

                **Detection threshold:** Score > 0.22 = CTA detected

                **Why it matters:** Videos with clear CTAs typically see 20-30% higher conversion rates. Knowing if a creator naturally includes CTAs helps predict campaign effectiveness.
                """)

            with st.expander("📊 What is Engagement Prediction?", expanded=False):
                st.markdown("""
                **Engagement prediction** combines 6 factors to estimate how well content will perform.

                **The formula (weighted scoring):**
                1. **Hook Strength (30%)** - Strong opening = higher retention
                2. **CTA Presence (15%)** - Clear calls-to-action boost conversions
                3. **Pacing Variety (20%)** - 3-15 transitions = engaging flow
                4. **Style Consistency (15%)** - Dominant visual identity
                5. **Optimal Length (10%)** - Matches content pacing
                6. **Transition Quality (10%)** - Smooth, dynamic changes

                **Performance categories:**
                - 🟢 **Excellent** (0.75+): High engagement expected
                - 🟡 **Good** (0.60-0.74): Strong performance likely
                - 🟠 **Fair** (0.45-0.59): Average engagement
                - 🔴 **Poor** (<0.45): May underperform

                **Plus:** Get specific strengths and weaknesses for each video to guide content strategy!
                """)

        st.divider()

        # Show example flow
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style='font-size: 0.9rem;'>

            ### 1️⃣ Define Your Brand

            <p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;'>What you do:</p>

            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>Choose visual style preferences
                    <ul><li><em>Example: "bright and colorful" for energetic brands</em></li></ul>
                </li>
                <li>Select content pacing
                    <ul><li><em>Example: "fast" for social media ads</em></li></ul>
                </li>
                <li>Optionally describe content type
                    <ul><li><em>Example: "tech product reviews"</em></li></ul>
                </li>
            </ul>

            <p style='font-size: 0.85rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;'>What happens:</p>

            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>Your preferences are converted into an AI embedding</li>
                <li>This creates a semantic "fingerprint" of your brand vision</li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='font-size: 0.9rem;'>

            ### 2️⃣ AI Analyzes Videos

            <p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;'>What we analyze:</p>

            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li><strong>Visual Style</strong> - 8 aesthetic categories</li>
                <li><strong>Pacing</strong> - Frame-to-frame changes</li>
                <li><strong>Hook</strong> - First 5 seconds effectiveness</li>
                <li><strong>CTA</strong> - Last 5 seconds call-to-action</li>
                <li><strong>Engagement</strong> - Predicted performance</li>
            </ul>

            <p style='font-size: 0.85rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;'>Core matching factors:</p>

            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>🤖 <strong>CLIP Similarity (50%)</strong> - Semantic visual match</li>
                <li>🎨 <strong>Style Match (30%)</strong> - Aesthetic alignment</li>
                <li>⚡ <strong>Pacing Match (20%)</strong> - Tempo compatibility</li>
            </ul>

            <p style='font-size: 0.8rem; font-style: italic; margin-top: 1rem;'>Plus: Hook, CTA, and Engagement metrics for uploaded videos</p>

            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='font-size: 0.9rem;'>

            ### 3️⃣ Get Comprehensive Results

            <p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;'>For each creator:</p>

            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>Overall match score (0-1)</li>
                <li>Top 3 visual style rankings</li>
                <li>Pacing category & transitions</li>
                <li><strong>Hook effectiveness</strong> 🎣</li>
                <li><strong>CTA detection</strong> 📢</li>
                <li><strong>Engagement prediction</strong> 📊</li>
            </ul>

            <p style='font-size: 0.85rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;'>Match score ranges:</p>

            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>🟢 0.80+ = Excellent match</li>
                <li>🟡 0.60-0.79 = Good match</li>
                <li>🟠 0.40-0.59 = Fair match</li>
                <li>🔴 <0.40 = Poor match</li>
            </ul>

            <p style='font-size: 0.8rem; font-style: italic; margin-top: 1rem;'>Uploaded videos include full engagement analysis!</p>

            </div>
            """, unsafe_allow_html=True)

        # Example use case
        st.divider()

        with st.expander("💡 Example: How a Fitness Brand Uses CreatorLens", expanded=False):
            st.markdown("""
            <div style='font-size: 0.9rem;'>

            <h3 style='font-size: 1.1rem;'>Scenario</h3>
            <p style='font-size: 0.85rem; line-height: 1.6;'>A fitness supplement brand wants to find creators for an Instagram campaign promoting their new energy drink.</p>

            <h3 style='font-size: 1.1rem; margin-top: 1rem;'>Their Input</h3>
            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li><strong>Primary Style:</strong> "vibrant and energetic" (matches their brand aesthetic)</li>
                <li><strong>Pacing:</strong> "fast" (for short-form social content)</li>
                <li><strong>Content Type:</strong> "fitness and workout videos"</li>
            </ul>

            <h3 style='font-size: 1.1rem; margin-top: 1rem;'>What CreatorLens Analyzes</h3>
            <ol style='font-size: 0.85rem; line-height: 1.6;'>
                <li><strong>CLIP Analysis (50%):</strong> Compares brand preferences against creator video frames
                    <ul><li>Looks for energetic visuals, bright lighting, dynamic movement</li></ul>
                </li>
                <li><strong>Style Matching (30%):</strong> Identifies creators with "vibrant and energetic" aesthetics
                    <ul><li>Filters out "dark and moody" or "minimalist" styles</li></ul>
                </li>
                <li><strong>Pacing Analysis (20%):</strong> Finds creators with quick cuts and high visual change
                    <ul><li>Matches the fast-paced social media style they need</li></ul>
                </li>
                <li><strong>🎣 Hook Detection:</strong> Checks if creators grab attention in first 3 seconds</li>
                <li><strong>📢 CTA Analysis:</strong> Verifies creators naturally include engagement prompts</li>
                <li><strong>📊 Engagement Prediction:</strong> Estimates content performance potential</li>
            </ol>

            <h3 style='font-size: 1.1rem; margin-top: 1rem;'>Results</h3>
            <p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem;'>Top Match: FitnessGuru (0.87)</p>
            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>Style: vibrant and energetic ✅</li>
                <li>Pacing: fast/dynamic ✅</li>
                <li><strong>Hook: 🟢 Excellent (0.89)</strong> - "Bold text overlay + face close-up"</li>
                <li><strong>CTA: ✅ Detected</strong> - "Subscribe and follow prompts at end"</li>
                <li><strong>Engagement: 🟢 Excellent (0.82)</strong>
                    <ul>
                        <li>Strengths: Strong hook, Clear CTA, Dynamic pacing</li>
                        <li>Perfect for Instagram Reels!</li>
                    </ul>
                </li>
            </ul>

            <p style='font-size: 0.85rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.3rem;'>Good Match: GameStreamer (0.72)</p>
            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>Style: vibrant ✅, but gaming-focused</li>
                <li>Pacing: moderate ⚠️</li>
                <li><strong>Hook: 🟡 Good (0.76)</strong></li>
                <li><strong>CTA: ✅ Detected</strong></li>
                <li><strong>Engagement: 🟡 Good (0.68)</strong></li>
            </ul>

            <h3 style='font-size: 1.1rem; margin-top: 1rem;'>Outcome</h3>
            <p style='font-size: 0.85rem; line-height: 1.6;'>The brand chooses FitnessGuru with high confidence because:</p>
            <ul style='font-size: 0.85rem; line-height: 1.6;'>
                <li>✅ Visual style matches perfectly (CLIP + Style scores)</li>
                <li>✅ Strong hook = high retention expected</li>
                <li>✅ Natural CTAs = better conversion rates</li>
                <li>✅ Excellent engagement prediction = ROI confidence</li>
            </ul>

            <p style='font-size: 0.85rem; font-weight: 600; margin-top: 1rem;'><strong>Result:</strong> Campaign performs 35% above industry benchmarks! 🎉</p>

            </div>
            """, unsafe_allow_html=True)

        # Show database stats with interactive charts
        st.divider()
        st.subheader("📊 Creator Database Analytics")

        # Total creators metric
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Total Creators", len(db.creators), delta="Active")

        # Count categories
        category_counts = {}
        for creator in db.creators:
            cat = creator.content_category or "Uncategorized"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        with col_metric2:
            st.metric("Content Categories", len(category_counts))

        with col_metric3:
            st.metric("Database Status", "Ready", delta="100%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive charts - stacked vertically
        cat_fig, style_fig = create_database_stats_charts(db)

        st.plotly_chart(cat_fig, use_container_width=True, key="db_cat_chart", config={'displayModeBar': False})
        st.plotly_chart(style_fig, use_container_width=True, key="db_style_chart", config={'displayModeBar': False})

    elif st.session_state.mode == 'upload':
        # Upload mode content
        if 'upload_results' in st.session_state:
            # Display uploaded video analysis results
            results = st.session_state.upload_results

            st.success(f"✅ Successfully added **{results['creator_name']}** to the database!")

            st.header("📊 Video Analysis Results")

            # Creator Info Card
            st.subheader("📌 Creator Information")
            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.metric("Creator Name", results['creator_name'])

            with info_col2:
                st.metric("Content Category", results.get('content_category', 'Not specified'))

            st.divider()

            # Style Analysis Card with Interactive Chart
            st.subheader("🎨 Visual Style Analysis")

            # Create style bar chart
            style_chart = create_style_bars(results['style_results'], top_n=3)
            st.plotly_chart(style_chart, use_container_width=True, key="upload_style_chart", config={'displayModeBar': False})

            # Top detected style as metric
            if results['style_results']:
                top_style = results['style_results'][0]
                st.metric(
                    "Primary Style Detected",
                    top_style['style'].title(),
                    delta=f"Confidence: {top_style['score']:.1%}"
                )

            st.divider()

            # Pacing Analysis Card
            st.subheader("⚡ Pacing Analysis")
            pacing_cols = st.columns(3)

            with pacing_cols[0]:
                st.metric("Pacing Category", results['pacing_info']['category'])

            with pacing_cols[1]:
                st.metric("Transitions", results['pacing_info']['num_transitions'])

            with pacing_cols[2]:
                st.metric("Mean Change", f"{results['pacing_info']['mean_change']:.3f}")

            st.divider()

            # Hook Analysis Card
            st.subheader("🎣 Hook Effectiveness")
            hook_data = results['hook_metrics']

            effectiveness_emoji = {
                "Excellent": "🟢",
                "Good": "🟡",
                "Moderate": "🟠",
                "Weak": "🔴"
            }

            hook_col1, hook_col2 = st.columns([1, 2])

            with hook_col1:
                effectiveness = hook_data.get('effectiveness', 'Unknown')
                emoji = effectiveness_emoji.get(effectiveness, "⚪")
                st.metric("Effectiveness", f"{emoji} {effectiveness}")
                st.metric("Hook Score", f"{hook_data.get('hook_score', 0):.3f}")

            with hook_col2:
                if hook_data.get('elements'):
                    st.write("**Detected Hook Elements:**")
                    for elem in hook_data['elements'][:5]:
                        st.write(f"- {elem['element']}: {elem['score']:.3f}")

            st.divider()

            # CTA Detection Card
            st.subheader("📢 Call-to-Action Detection")
            cta_data = results['cta_metrics']

            cta_col1, cta_col2 = st.columns([1, 2])

            with cta_col1:
                cta_detected = cta_data.get('cta_detected', False)
                cta_icon = "✅" if cta_detected else "❌"
                st.metric("CTA Status", f"{cta_icon} {'Detected' if cta_detected else 'Not Detected'}")
                st.metric("CTA Score", f"{cta_data.get('cta_score', 0):.3f}")

            with cta_col2:
                if cta_detected and cta_data.get('elements'):
                    st.write("**Detected CTA Types:**")
                    for elem in cta_data['elements'][:5]:
                        st.write(f"- {elem['type']}: {elem['score']:.3f}")

            st.divider()

            # Engagement Prediction Card with Gauge
            st.subheader("📊 Engagement Prediction")
            engagement_data = results['engagement_prediction']

            perf = engagement_data.get('predicted_performance', 'Unknown')
            score = engagement_data.get('engagement_score', 0)

            # Create two columns: gauge on left, details on right
            gauge_col, details_col = st.columns([1, 1])

            with gauge_col:
                # Interactive gauge chart
                gauge_fig = create_engagement_gauge(score)
                st.plotly_chart(gauge_fig, use_container_width=True, key="upload_engagement_gauge", config={'displayModeBar': False})

            with details_col:
                # Performance badge
                perf_emoji = {
                    "Excellent": "🟢",
                    "Good": "🟡",
                    "Fair": "🟠",
                    "Poor": "🔴"
                }
                emoji = perf_emoji.get(perf, "⚪")

                # Create HTML badge for performance
                badge_class = f"badge-{perf.lower()}" if perf in ["Excellent", "Good", "Fair", "Poor"] else "badge-fair"
                st.markdown(f"""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h3 style='color: #334155; font-family: Poppins;'>Predicted Performance</h3>
                    <span class='score-badge {badge_class}' style='font-size: 1.2rem; padding: 0.5rem 1.5rem;'>
                        {emoji} {perf}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Factor breakdown
                if engagement_data.get('factor_scores'):
                    st.write("**📈 Factor Breakdown:**")
                    for factor, factor_score in engagement_data['factor_scores'].items():
                        # Create a mini progress bar for each factor
                        st.write(f"**{factor}:**")
                        st.progress(factor_score)
                        st.caption(f"Score: {factor_score:.2f}")

            # Strengths and Weaknesses
            strength_col, weakness_col = st.columns(2)

            with strength_col:
                if engagement_data.get('strengths'):
                    st.success("**✅ Strengths:**")
                    for strength in engagement_data['strengths']:
                        st.write(f"- {strength}")

            with weakness_col:
                if engagement_data.get('weaknesses'):
                    st.warning("**⚠️ Areas to Improve:**")
                    for weakness in engagement_data['weaknesses']:
                        st.write(f"- {weakness}")

            # Add action buttons
            st.divider()
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button("📤 Upload Another Video", use_container_width=True):
                    del st.session_state.upload_results
                    st.rerun()

            with action_col2:
                if st.button("🔍 Match This Creator to Brands", use_container_width=True):
                    del st.session_state.upload_results
                    st.session_state.mode = 'match'
                    st.rerun()

        else:
            # Show upload instructions
            st.info("👈 Upload a creator video in the sidebar to see detailed AI analysis!")

            st.subheader("📤 Video Upload Analysis")
            st.markdown("""
            Upload a creator's video to get comprehensive AI-powered analysis including:

            - **🎨 Visual Style** - Identify the aesthetic category and style scores
            - **⚡ Pacing** - Analyze content tempo and transitions
            - **🎣 Hook Detection** - Evaluate first 3-5 seconds effectiveness
            - **📢 CTA Detection** - Identify call-to-action patterns
            - **📊 Engagement Prediction** - Predict content performance

            **Supported Formats:** MP4, MOV, AVI, MKV
            **Recommended Length:** 10-60 seconds
            **Processing Time:** ~10-30 seconds depending on video length
            """)

    elif st.session_state.mode == 'match':
        # Match mode content
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
                Overall Score = (CLIP × 45%) + (Style × 30%) + (Pacing × 20%) + (Content × 5%)
                ```

                #### 🤖 CLIP Similarity (45% weight)
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

                #### 📂 Content Category Match (5% weight)
                Filters creators by their primary content niche (manually tagged):
                - Categories: Cooking/Food, Tech/Reviews, Fitness/Wellness, Gaming, Fashion/Beauty, Education/Tutorial, Vlog/Lifestyle, Business/Finance, Art/Creative, Other
                - **1.0** = Exact category match (e.g., Cooking brand → Cooking creator)
                - **0.5** = No category specified (neutral)
                - **0.0** = Category mismatch (e.g., Cooking brand → Gaming creator)

                ---
                **Score Interpretation:**
                - 🟢 **0.80 - 1.00**: Excellent match - Highly recommended
                - 🟡 **0.60 - 0.79**: Good match - Strong alignment
                - 🟠 **0.40 - 0.59**: Fair match - Moderate alignment
                - 🔴 **0.00 - 0.39**: Poor match - Low alignment

                ---
                ### 📊 Additional Metrics (For Uploaded Videos)

                When you upload a video, we also analyze:

                **🎣 Hook Effectiveness**
                - Analyzes first 3-5 seconds for viewer retention potential
                - Detects 7 hook elements (face presence, text overlays, movement, etc.)
                - Rated: Excellent/Good/Moderate/Weak

                **📢 CTA Detection**
                - Scans last 5 seconds for call-to-action patterns
                - Identifies 7 CTA types (text, gestures, subscribe buttons, etc.)
                - Binary: Detected ✅ or Not detected ❌

                **📊 Engagement Prediction**
                - Combines 6 factors: Hook (30%), CTA (15%), Pacing Variety (20%), Style Consistency (15%), Length (10%), Transitions (10%)
                - Provides predicted performance: Excellent/Good/Fair/Poor
                - Lists specific strengths and areas for improvement

                These metrics help you assess not just brand fit, but content quality and performance potential!
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
                    # Create radar chart and data table side by side
                    radar_col, table_col = st.columns([1, 1])

                    with radar_col:
                        st.subheader("📊 Match Scores")
                        # Interactive radar chart
                        radar_fig = create_score_radar_chart(match)
                        st.plotly_chart(radar_fig, use_container_width=True, key=f"radar_{creator.name}_{i}", config={'displayModeBar': False})

                    with table_col:
                        st.subheader("📈 Score Breakdown")
                        # Score breakdown table
                        score_data = {
                            'Factor': ['Overall', 'CLIP', 'Style', 'Pacing', 'Content'],
                            'Score': [
                                match['overall_score'],
                                match['clip_similarity'],
                                match['style_match'],
                                match['pacing_match'],
                                match['content_match']
                            ],
                            'Weight': ['-', '45%', '30%', '20%', '5%'],
                            'Weighted': [
                                '-',
                                f"{match['clip_similarity'] * 0.45:.3f}",
                                f"{match['style_match'] * 0.30:.3f}",
                                f"{match['pacing_match'] * 0.20:.3f}",
                                f"{match['content_match'] * 0.05:.3f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(score_data), hide_index=True, use_container_width=True)

                    st.divider()

                    # Creator profile below
                    col1, col2 = st.columns([1, 1])
                    
                    with col2:
                        st.subheader("👤 Creator Profile")
                        st.write(f"**Name:** {creator.name}")
                        st.write(f"**Primary Style:** {creator.style['primary']}")

                        # Show content category if available
                        if creator.content_category:
                            st.write(f"**Content Category:** {creator.content_category}")

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

                    with col1:
                        st.subheader("🏆 Match Quality")
                        # Score indicator with badge
                        if match['overall_score'] >= 0.8:
                            badge_class = "badge-excellent"
                            emoji = "🟢"
                            label = "Excellent Match"
                            desc = "Highly recommended - Strong alignment across all factors"
                        elif match['overall_score'] >= 0.6:
                            badge_class = "badge-good"
                            emoji = "🟡"
                            label = "Good Match"
                            desc = "Strong alignment - Good fit for your brand"
                        elif match['overall_score'] >= 0.4:
                            badge_class = "badge-fair"
                            emoji = "🟠"
                            label = "Fair Match"
                            desc = "Moderate alignment - Consider specific factors"
                        else:
                            badge_class = "badge-poor"
                            emoji = "🔴"
                            label = "Poor Match"
                            desc = "Low alignment - May not be the best fit"

                        st.markdown(f"""
                        <div style='text-align: center; margin: 1rem 0;'>
                            <span class='score-badge {badge_class}' style='font-size: 1.1rem; padding: 0.5rem 1.5rem;'>
                                {emoji} {label}
                            </span>
                            <p style='color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;'>{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detailed comparison table
            st.divider()
            st.subheader("📋 Comparison Table")
            
            comparison_data = []
            for i, match in enumerate(matches, 1):
                creator = match['creator']
                row = {
                    'Rank': i,
                    'Creator': creator.name,
                    'Overall Score': f"{match['overall_score']:.3f}",
                    'CLIP': f"{match['clip_similarity']:.3f}",
                    'Style': f"{match['style_match']:.3f}",
                    'Pacing': f"{match['pacing_match']:.3f}",
                    'Creator Style': creator.style['primary'],
                    'Creator Pacing': creator.pacing['category']
                }

                # Add engagement metrics if available
                if creator.hook:
                    row['Hook'] = f"{creator.hook.get('effectiveness', 'N/A')}"
                if creator.cta:
                    row['CTA'] = "✓" if creator.cta.get('cta_detected') else "✗"
                if creator.engagement:
                    row['Engagement'] = f"{creator.engagement.get('predicted_performance', 'N/A')}"

                comparison_data.append(row)

            st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

        else:
            # Show instructions before search
            st.info("👈 Configure your brand requirements in the sidebar and click 'Find Creators' to start!")

            st.subheader("🔍 Find Your Perfect Creator Match")
            st.markdown("""
            Use the sidebar to specify your brand requirements:

            - **Visual Style** - Choose primary and secondary aesthetic preferences
            - **Content Pacing** - Select your preferred tempo (fast, moderate, slow)
            - **Content Type** - Optionally describe the type of content you need
            - **Number of Results** - How many creator matches to display

            Our AI will analyze all creators in the database and rank them by compatibility with your brand!
            """)

    # Footer (shown in all modes)
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>CreatorLens</strong> - Built with Streamlit, CLIP, and FAISS</p>
        <p>Multi-factor AI matching system for creator-brand partnerships</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
