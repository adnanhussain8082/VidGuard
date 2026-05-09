import gradio as gr
from PIL import Image

from classify import classify_media, load_model


model = load_model()

# === Inference Logic ===
def predict_file(file_obj, frame_stride=1):
    if file_obj is None:
        return "⚠️ No file selected", "", [], [], None

    path = file_obj.name
    try:
        result = classify_media(
            path,
            model,
            num_video_frames=5,
            video_frame_stride=max(int(frame_stride), 1),
        )
    except Exception as exc:
        return f"❌ Error: {exc}", "", [], [], None

    ui_label = "🟢 Real" if result["label"] == "REAL" else "🔴 Deepfake"
    ui_confidence = f"{result['confidence'] * 100:.2f}%"

    if result["type"] == "image":
        original = Image.open(path).convert("RGB")
        overlay = Image.fromarray(result["overlay"])
        return ui_label, ui_confidence, [original], [overlay], None

    original_gallery = [Image.fromarray(frame["original"]) for frame in result["frames"]]
    heatmap_gallery = [Image.fromarray(frame["overlay"]) for frame in result["frames"]]
    annotated_video = result.get("annotated_video_path")
    return ui_label, ui_confidence, original_gallery, heatmap_gallery, annotated_video

# === Professional Enterprise-Grade CSS ===
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --navy-900: #0f172a;
    --navy-800: #1e293b;
    --navy-700: #334155;
    --navy-600: #475569;
    --slate-100: #f1f5f9;
    --slate-200: #e2e8f0;
    --slate-300: #cbd5e1;
    --accent-blue: #3b82f6;
    --accent-indigo: #6366f1;
    --accent-success: #22c55e;
    --accent-danger: #ef4444;
    --accent-warning: #f59e0b;
}

* {
    font-family: 'IBM Plex Sans', -apple-system, system-ui, sans-serif;
}

body {
    background: var(--navy-900) !important;
    color: var(--slate-100) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 3rem 2rem !important;
    background: transparent !important;
}

/* Professional Header */
.main-header {
    background: linear-gradient(135deg, var(--navy-800) 0%, var(--navy-700) 100%);
    padding: 2.5rem 3rem;
    border-radius: 16px;
    margin-bottom: 3rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
}

.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.app-title {
    font-size: 2.75rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.app-subtitle {
    font-size: 1.125rem;
    color: var(--slate-300);
    font-weight: 500;
    margin: 0.5rem 0 0 0;
    letter-spacing: 0.01em;
}

.beta-badge {
    display: inline-block;
    background: var(--accent-indigo);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-left: 1rem;
}

/* Section Containers */
.section-container {
    background: var(--navy-800);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
}

.section-container:hover {
    border-color: rgba(255, 255, 255, 0.12);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

/* Typography */
h3 {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    margin: 0 0 1.25rem 0 !important;
    letter-spacing: -0.01em !important;
    border-left: 3px solid var(--accent-indigo);
    padding-left: 0.75rem;
}

p, label {
    color: var(--slate-300) !important;
    line-height: 1.6 !important;
}

label {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 0.5rem !important;
}

/* File Upload - Professional Style */
.file-upload {
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
    background: rgba(99, 102, 241, 0.03) !important;
    padding: 2.5rem 2rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-align: center !important;
}

.file-upload:hover {
    border-color: var(--accent-indigo) !important;
    background: rgba(99, 102, 241, 0.08) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
}

/* Input Fields */
input[type="text"], textarea, .textbox {
    background: var(--navy-700) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: var(--slate-100) !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
    font-family: 'JetBrains Mono', monospace !important;
}

input[type="text"]:focus, textarea:focus, .textbox:focus {
    border-color: var(--accent-indigo) !important;
    background: var(--navy-600) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

/* Buttons */
button {
    background: linear-gradient(135deg, var(--accent-indigo) 0%, var(--accent-blue) 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: white !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3) !important;
}

button:active {
    transform: translateY(0) !important;
}

/* Gallery - Clean Professional Grid */
.gallery {
    background: var(--navy-700) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    overflow: hidden !important;
}

.gallery img {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.2s ease !important;
}

.gallery img:hover {
    transform: scale(1.02) !important;
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

/* Status Indicators */
.status-real {
    color: var(--accent-success) !important;
    font-weight: 600 !important;
}

.status-fake {
    color: var(--accent-danger) !important;
    font-weight: 600 !important;
}

/* Footer */
.footer {
    background: var(--navy-800);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 3rem;
    text-align: center;
    color: var(--slate-400);
    font-size: 0.875rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
}

.footer-links a {
    color: var(--accent-indigo);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
}

.footer-links a:hover {
    color: var(--accent-blue);
}

/* Metrics Display */
.metric-card {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-indigo);
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--slate-400);
    margin-top: 0.25rem;
}

/* Responsive */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1.5rem 1rem !important;
    }
    
    .app-title {
        font-size: 2rem;
    }
    
    .section-container {
        padding: 1.5rem;
    }
}
    margin-bottom: 1rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .app-title {
        font-size: 2.5rem;
    }
    
    .gradio-container {
        padding: 1rem !important;
    }
    
    .input-card, .output-card {
        padding: 1.5rem;
    }
}
"""

# === Gradio UI with Professional Enterprise Design ===
with gr.Blocks(title="VidGuard - AI Deepfake Detection") as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Professional Header
    gr.HTML("""
        <div class="main-header">
            <div class="logo-container">
                <h1 class="app-title">🛡️ VidGuard</h1>
                <span class="beta-badge">Research Preview</span>
            </div>
            <p class="app-subtitle">Enterprise-Grade AI Media Authentication Platform</p>
        </div>
    """)
    
    with gr.Row():
        # Input Section
        with gr.Column(scale=1, elem_classes=["section-container"]):
            gr.Markdown("### 📤 Media Upload")
            file_input = gr.File(
                label="Select File",
                file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
                elem_classes=["file-upload"]
            )
            frame_stride = gr.Slider(
                minimum=1,
                maximum=30,
                step=1,
                value=1,
                label="Video Frame Stride (process every Nth frame)",
            )
            with gr.Row():
                analyze_btn = gr.Button("Start Detection", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
            gr.Markdown("**Supported:** JPG, PNG • MP4, MOV • Max 200MB")
        
        # Output Section
        with gr.Column(scale=1, elem_classes=["section-container"]):
            gr.Markdown("### 🎯 Detection Results")
            with gr.Row():
                prediction = gr.Textbox(
                    label="Classification", 
                    interactive=False,
                    elem_classes=["textbox"]
                )
                confidence = gr.Textbox(
                    label="Confidence", 
                    interactive=False,
                    elem_classes=["textbox"]
                )
    
    # Preview Gallery Section
    with gr.Column(elem_classes=["section-container"]):
        gr.Markdown("### 🖼️ Original Frames")
        gr.Markdown("_Image shows one frame, video shows 5 sampled frames_")
        original_preview = gr.Gallery(
            label="",
            show_label=False,
            columns=5,
            rows=1,
            object_fit="cover",
            height="auto",
            elem_classes=["gallery"]
        )

    with gr.Column(elem_classes=["section-container"]):
        gr.Markdown("### 🔥 Grad-CAM Overlays")
        gr.Markdown("_Red/yellow regions indicate strongest model attention_")
        heatmap_preview = gr.Gallery(
            label="",
            show_label=False,
            columns=5,
            rows=1,
            object_fit="cover",
            height="auto",
            elem_classes=["gallery"]
        )

    with gr.Column(elem_classes=["section-container"]):
        gr.Markdown("### 🎬 Annotated Video")
        gr.Markdown("_Bounding boxes and highlighted regions for easy visual explanation_")
        annotated_video = gr.Video(
            label="",
            format="mp4",
            interactive=False,
            autoplay=False,
            show_label=False,
        )
    
    # Professional Footer
    gr.HTML("""
        <div class="footer">
            <p style="margin: 0; font-weight: 500;">
                ⚡ Powered by <strong>EfficientNet-B0</strong> • Built for Media Forensics Research
            </p>
            <p style="margin: 0.75rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">
                ⚠️ Research Tool • Results require expert verification • Not for production deployment
            </p>
            <div class="footer-links">
                <span>📊 Model: EfficientNet-B0</span>
                <span>🎯 Accuracy: Research Grade</span>
                <span>🔒 Privacy: Local Processing</span>
            </div>
        </div>
    """)

    def handle_input(file_obj, frame_stride_value):
        return predict_file(file_obj, frame_stride=frame_stride_value)

    def handle_clear():
        return None, "", "", [], [], None

    analyze_btn.click(
        fn=handle_input,
        inputs=[file_input, frame_stride],
        outputs=[prediction, confidence, original_preview, heatmap_preview, annotated_video]
    )

    clear_btn.click(
        fn=handle_clear,
        inputs=[],
        outputs=[file_input, prediction, confidence, original_preview, heatmap_preview, annotated_video],
    )
demo.launch()
