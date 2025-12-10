# CreatorLens 🎥

> AI-powered video analysis and creator-brand matching platform

CreatorLens is a comprehensive Streamlit-based application that uses OpenAI's CLIP model and advanced computer vision techniques to analyze video content and match creators with brand requirements. It provides deep insights into video style, pacing, hooks, CTAs, and predicts engagement potential.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Key Features

- **🎯 AI-Powered Creator-Brand Matching** - Semantic similarity search using CLIP embeddings with multi-factor scoring
- **🎨 Visual Style Analysis** - Classifies videos across 8 semantic style categories (bright/colorful, dark/moody, minimalist, etc.)
- **⚡ Pacing Detection** - Embedding-based analysis of video dynamics and transitions
- **🪝 Hook Detection** - Analyzes first 3-5 seconds for effectiveness (Excellent/Good/Moderate/Weak)
- **📢 CTA Detection** - Identifies call-to-action patterns in video endings
- **📊 Engagement Prediction** - Multi-factor heuristic model predicting video performance
- **🔍 FAISS-Powered Search** - Fast approximate nearest neighbor search for large creator databases
- **🖥️ Interactive UI** - Rich Streamlit interface with visualizations and expandable creator profiles

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Features](#core-features)
- [Content Categories](#content-categories)
- [Scoring Methodology](#scoring-methodology)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Development](#development)
- [Limitations & Future Work](#limitations--future-work)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/priyam-02/creatorlens.git
   cd creatorlens
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note for macOS users:** The project uses `torch>=2.2.0` for compatibility with Apple Silicon and Intel Macs.

---

## Quick Start

1. **Launch the application**

   ```bash
   streamlit run creatorlens.py
   ```

2. **Access the web interface**

   - Open your browser to `http://localhost:8501`

3. **Basic workflow**

   - **Sidebar**: Enter brand requirements (description, preferred style, pacing, content category)
   - **Upload videos**: Add creator videos to build your database
   - **View matches**: See ranked creator matches with detailed analysis
   - **Explore profiles**: Expand creator cards to see style breakdowns, pacing metrics, hook/CTA analysis, and engagement predictions

4. **Demo mode**
   - The app includes 10 synthetic creator profiles for testing
   - Upload real videos for full analysis capabilities

---

## Architecture Overview

CreatorLens is built as a single-file Streamlit application with four main architectural layers:

### 1. Core Matching Engine

- **CreatorProfile**: Dataclass storing creator metadata, embeddings, and analytics
- **CreatorDatabase**: FAISS-based vector database for similarity search
- **BrandBrief**: Converts brand requirements into CLIP embeddings
- **Matching function**: Multi-factor scoring combining CLIP similarity, style, pacing, and content category

### 2. Video Processing Pipeline

- **CLIP-based style analysis**: Text-image similarity across 8 semantic categories
- **Embedding-based pacing**: Cosine distance between consecutive frames
- **Hook detection**: Analyzes first 3-5 seconds for 7 hook elements
- **CTA detection**: Identifies 7 call-to-action patterns in video endings
- **Engagement prediction**: Weighted scoring across 6 factors

### 3. Visualization Layer

- Radar charts for multi-dimensional scoring
- Engagement gauge charts
- Style distribution bar charts
- Database statistics and analytics

### 4. Streamlit UI

- Interactive sidebar for brand requirements
- Video upload with progress tracking
- Rich creator profile cards with expandable details
- Session state management for database persistence

### Technology Stack

- **Deep Learning**: PyTorch, OpenAI CLIP
- **Computer Vision**: OpenCV
- **Vector Search**: FAISS (IndexFlatIP)
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: NumPy, Pandas

---

## Core Features

### Creator-Brand Matching

The matching system uses a weighted scoring algorithm that combines:

- **CLIP Similarity (45%)**: Semantic similarity between brand description and video content
- **Style Matching (30%)**: Alignment of visual style preferences
- **Pacing Matching (20%)**: Compatibility of video pacing with brand requirements
- **Content Category (5%)**: Niche/category alignment

Results are ranked and displayed with detailed breakdowns of each scoring component.

### Video Analysis

#### Style Detection

- Uses CLIP's multimodal understanding to classify videos across 8 categories:
  - Bright and colorful
  - Dark and moody
  - Minimalist and clean
  - Vintage or retro
  - High contrast
  - Soft and pastel
  - Natural and earthy
  - Monochrome or black and white
- Returns ranked scores for ALL styles, not just a single category

#### Pacing Analysis

- Embedding-based approach using cosine distance between consecutive frames
- Categories: Fast, Moderate, Slow, Dynamic, Static
- Thresholds:
  - `<0.15`: Slow/static
  - `0.15-0.25`: Moderate
  - `>0.25`: Fast/dynamic
- Detects transitions where frame change exceeds 0.3
- More robust than pixel-based methods (ignores camera motion and noise)

### Hook Detection

Analyzes the critical first 3-5 seconds of video for effectiveness:

- **Elements detected** (7 types):

  - Face presence
  - Text overlays
  - Bright colors
  - Movement/action
  - Interesting visuals
  - Sound/music cues
  - Rapid cuts

- **Effectiveness ratings**:
  - Excellent: >0.28 CLIP similarity
  - Good: 0.24-0.28
  - Moderate: 0.20-0.24
  - Weak: <0.20

### CTA Detection

Identifies call-to-action patterns in the last 5 seconds:

- **CTA types detected** (7 patterns):

  - Text/caption CTAs
  - Hand gestures (pointing, thumbs up)
  - Subscribe buttons
  - Product displays
  - Link/URL mentions
  - Verbal CTAs
  - End screens

- **Detection threshold**: 0.22 CLIP similarity
- Returns binary detection + confidence score

### Engagement Prediction

Multi-factor heuristic model combining:

| Factor             | Weight | Description                          |
| ------------------ | ------ | ------------------------------------ |
| Hook Strength      | 30%    | Effectiveness of opening 3-5 seconds |
| CTA Presence       | 15%    | Clear call-to-action in ending       |
| Pacing Variety     | 20%    | Optimal transition count (3-15)      |
| Style Consistency  | 15%    | Primary style dominance              |
| Optimal Length     | 10%    | Duration appropriate for pacing      |
| Transition Quality | 10%    | Smooth, dynamic changes              |

**Output**: Score 0-1, categorized as Excellent/Good/Fair/Poor with strengths/weaknesses breakdown

---

## Content Categories

The system supports 10 manually-tagged content categories:

1. **Cooking/Food**
2. **Tech/Reviews**
3. **Fitness/Wellness**
4. **Gaming**
5. **Fashion/Beauty**
6. **Education/Tutorial**
7. **Vlog/Lifestyle**
8. **Business/Finance**
9. **Art/Creative**
10. **Other**

**Note**: Content categories are manually tagged during video upload because CLIP cannot reliably distinguish content topics from visual frames alone (e.g., a cooking video and a chemistry tutorial may look visually similar).

---

## Scoring Methodology

### Final Match Score Formula

```
final_score = (clip_similarity × 0.45) + (style_match × 0.30) + (pacing_match × 0.20) + (content_match × 0.05)
```

### Component Scoring

**Style Matching**:

- 1.0 = Primary style match
- 0.7 = Secondary style match
- 0.0 = No match

**Pacing Matching**:

- 1.0 = Exact match (fast/fast, moderate/moderate, slow/slow)
- 0.5 = Neutral/unspecified
- 0.0 = Opposite (fast/slow mismatch)

**Content Category Matching**:

- 1.0 = Exact category match
- 0.5 = No category specified (neutral)
- 0.0 = Category mismatch

### Engagement Prediction Scoring

Each factor contributes to a 0-1 score:

- **Hook Strength (0.30)**: Linear scaling from weak to excellent
- **CTA Presence (0.15)**: Binary bonus for detected CTA
- **Pacing Variety (0.20)**: Peak score at 3-15 transitions
- **Style Consistency (0.15)**: Ratio of primary style score
- **Optimal Length (0.10)**: Duration match for pacing category
- **Transition Quality (0.10)**: Smooth dynamic changes preferred

---

## Configuration

### Modifying Style Concepts

To add or change style categories, edit the `STYLE_CONCEPTS` list in `creatorlens.py`:

```python
STYLE_CONCEPTS = [
    "bright and colorful video",
    "dark and moody video",
    "your custom style here",
    # ... add more
]
```

### Adjusting Scoring Weights

Modify the `match_brand_to_creators()` function weights:

```python
final_score = (
    clip_similarity * 0.45 +  # Adjust CLIP weight
    style_match * 0.30 +      # Adjust style weight
    pacing_match * 0.20 +     # Adjust pacing weight
    content_match * 0.05      # Adjust content weight
)
```

### Adding Content Categories

Update the `CONTENT_CATEGORIES` list:

```python
CONTENT_CATEGORIES = [
    "Cooking/Food",
    "Your New Category",
    # ... add more
]
```

---

## Technical Details

### FAISS Index Strategy

- **Index Type**: `IndexFlatIP` (inner product)
- **Rationale**: CLIP embeddings are L2-normalized, making inner product equivalent to cosine similarity
- **Rebuild**: Index is rebuilt whenever the database changes (on video upload)

### CLIP Model

- **Model**: `ViT-B/32` (Vision Transformer)
- **Embedding Dimension**: 512
- **Input**: Text descriptions and video frames (224×224 RGB)
- **Device**: Auto-detects CUDA availability, defaults to CPU

### Device Handling

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

- GPU acceleration used when available
- CPU fallback for compatibility
- macOS: Supports both Apple Silicon (M1/M2/M3) and Intel

### macOS Compatibility

- Uses `torch>=2.2.0` and `torchvision>=0.17.0`
- Compatible with Apple Silicon (M1/M2/M3) and Intel Macs
- MPS (Metal Performance Shaders) support for Apple Silicon acceleration

### Database Persistence

- Uses `st.session_state['database']` for in-memory storage
- Persists during active session
- **Resets on browser refresh** (no permanent storage)
- Demo creators loaded as fallback

---

## Project Structure

```
creatorlens/
├── creatorlens.py              # Main application (2,251 lines)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── .git/                       # Git repository
```

### Key Files

- **creatorlens.py**: Single-file application containing all logic
  - Lines 315-413: Core matching engine
  - Lines 484-968: Video processing pipeline
  - Lines 1020-1211: Visualization functions
  - Lines 1212-2250+: Streamlit UI and state management

---

## Development

### Code Organization Philosophy

CreatorLens is intentionally built as a **single-file application** for:

- Rapid development and prototyping
- Easy deployment and distribution
- Simplified dependency management
- Straightforward debugging

### Running Tests

Currently, the project focuses on interactive testing through the Streamlit UI. Automated testing infrastructure is planned for future releases.

---

## Limitations & Future Work

### Current Limitations

1. **Manual Content Tagging**: Content categories must be manually selected during upload (CLIP cannot reliably distinguish content topics from visual analysis alone)

2. **Session-Based Persistence**: Database resets on browser refresh (no permanent storage backend)

3. **Single-File Architecture**: While great for prototyping, may need refactoring for large-scale production use

4. **CPU Processing**: Video analysis can be slow on CPU-only systems

5. **Heuristic Engagement Model**: Engagement prediction uses rule-based scoring rather than trained ML models

### Future Improvements

- **Persistent Storage**: Database backend (PostgreSQL, MongoDB) for permanent creator profiles
- **Automated Content Classification**: Fine-tuned model for content category detection
- **ML-Based Engagement Model**: Train on real engagement data (views, likes, shares)
- **Batch Processing**: API endpoint for bulk video analysis
- **Advanced Analytics**: Trend detection, creator performance tracking over time
- **Multi-Language Support**: Caption and audio analysis for non-English content
- **Audio Analysis**: Speech-to-text, music detection, sound effect classification
- **Export Features**: PDF reports, CSV exports, API access

---

## Dependencies

See [requirements.txt](requirements.txt) for complete list. Key dependencies:

- **streamlit>=1.29.0** - Web UI framework
- **torch>=2.2.0** - Deep learning framework
- **clip** (OpenAI) - Multimodal embeddings
- **faiss-cpu>=1.7.4** - Fast similarity search
- **opencv-python>=4.8.0** - Video processing
- **plotly>=5.18.0** - Interactive visualizations
- **pandas>=2.1.3** - Data manipulation
- **numpy>=1.26.0** - Numerical computing

---

## Acknowledgments

- **OpenAI CLIP** - Multimodal embeddings powering semantic understanding
- **FAISS** - Efficient similarity search by Facebook AI Research
- **Streamlit** - Rapid web application framework
- **PyTorch** - Deep learning infrastructure

---

## Contact & Support

- **Repository**: [https://github.com/priyam-02/creatorlens](https://github.com/priyam-02/creatorlens)

---

**Built using OpenAI CLIP and Streamlit**
