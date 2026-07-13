import streamlit as st
import torch
import torch.nn as nn
import time
import os
import html

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(
    page_title="Char RNN Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Custom CSS — dark, glassy, gradient-accented theme
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* App background */
    .stApp {
        background: radial-gradient(circle at 10% 0%, #1b1035 0%, #0d0b1f 40%, #090812 100%);
        color: #eaeaf5;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer {visibility: hidden;}

    /* Hero header */
    .hero-wrap {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        margin-bottom: 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(236,72,153,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #c084fc, #f472b6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #b8b8d0;
        margin-top: 0.6rem;
        font-weight: 400;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f4f4ff;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-caption {
        color: #9c9cc0;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        height: 3.1rem;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        color: white;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(124,58,237,0.35);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 22px rgba(236,72,153,0.45);
    }
    .stDownloadButton>button {
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white;
        border: none;
        box-shadow: 0 4px 14px rgba(59,130,246,0.35);
        transition: all 0.25s ease;
    }
    .stDownloadButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 22px rgba(6,182,212,0.45);
    }

    /* Output text card */
    .output-card {
        background: linear-gradient(135deg, #14101f, #0a0913);
        color: #e6e6fa;
        border-radius: 16px;
        padding: 1.6rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
        white-space: pre-wrap;
        line-height: 1.55;
        border: 1px solid rgba(196,132,252,0.3);
        box-shadow: inset 0 0 30px rgba(124,58,237,0.08), 0 4px 24px rgba(0,0,0,0.4);
        max-height: 420px;
        overflow-y: auto;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.7rem 0.5rem;
    }
    [data-testid="stMetricLabel"] { color: #b8b8d0 !important; }
    [data-testid="stMetricValue"] { color: #f4f4ff !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150f28, #0a0913);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #e6c2ff;
        font-weight: 800;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .badge-warn { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.35); }
    .badge-ok   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.35); }

    hr { border-color: rgba(255,255,255,0.08) !important; }

    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        color: #eaeaf5 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    div[data-baseweb="tab-list"] { gap: 6px; }
    button[data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        background: rgba(255,255,255,0.03);
        color: #b8b8d0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Config & Setup
# =====================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FILE_PATH   = os.path.join(BASE_DIR, "data.txt")
SEQ_LEN     = 100
BATCH_SIZE  = 64
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
EMBED_SIZE  = 128
EPOCHS      = 20
LR          = 0.002
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# =====================================================================
# Load data & vocabulary (Cached)
# =====================================================================
@st.cache_data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)
    return text, chars, vocab_size, char2idx, idx2char, data

text_data, chars, vocab_size, char2idx, idx2char, data_tensor = load_data(FILE_PATH)

def get_batch(data, seq_len, batch_size):
    """Many-to-many training pair: y is x shifted by one character."""
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)

# =====================================================================
# Model
# =====================================================================
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

if 'model' not in st.session_state:
    st.session_state.model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    st.session_state.is_trained = False

# =====================================================================
# Generation function
# =====================================================================
def generate(model, start_str, length, temperature=0.8):
    model.eval()
    chars_generated = list(start_str)

    safe_start = [char2idx[ch] for ch in start_str if ch in char2idx]
    if not safe_start:
        return ""

    input_seq = torch.tensor([safe_start], dtype=torch.long).to(DEVICE)
    hidden = None
    with torch.no_grad():
        logits, hidden = model(input_seq, hidden)
        last_char_logits = logits[0, -1, :]

        for _ in range(length):
            probs = torch.softmax(last_char_logits / temperature, dim=0)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            chars_generated.append(idx2char[next_idx])

            next_input = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
            logits, hidden = model(next_input, hidden)
            last_char_logits = logits[0, -1, :]

    return "".join(chars_generated)

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("# 🧬 Char RNN Studio")
    st.markdown("---")
    st.markdown("### 📊 Dataset & Device")
    st.info(f"**Dataset**  \n`{os.path.basename(FILE_PATH)}`")
    st.info(f"**Device**  \n`{str(DEVICE).upper()}`")

    c1, c2 = st.columns(2)
    c1.metric("Vocab Size", vocab_size)
    c2.metric("Total Chars", f"{len(text_data):,}")

    st.markdown("### 🎛️ Hyperparameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Epochs", EPOCHS)
        st.metric("Batch Size", BATCH_SIZE)
        st.metric("Layers", NUM_LAYERS)
    with col2:
        st.metric("Seq Length", SEQ_LEN)
        st.metric("Hidden Size", HIDDEN_SIZE)
        st.metric("Embed Size", EMBED_SIZE)

    st.metric("Learning Rate", LR)
    st.markdown("---")
    st.caption("Built with 💜 using PyTorch + Streamlit")

# =====================================================================
# Hero Header
# =====================================================================
st.markdown("""
<div class="hero-wrap">
    <p class="hero-title">🧠 Many-to-Many Character RNN</p>
    <p class="hero-subtitle">Train a GRU-based recurrent neural network to generate text, one character at a time.</p>
</div>
""", unsafe_allow_html=True)

# --- Info tabs instead of plain expanders ---
tab1, tab2, tab3 = st.tabs(["ℹ️ About", "🏗️ Architecture", "🔄 How It Works"])
with tab1:
    st.markdown("""
    <div class="glass-card">
    This project demonstrates text generation using a <b>Character-Level Recurrent Neural Network</b>.
    Given a sequence of characters, the model predicts the most likely next character, allowing it to
    generate text that mimics the style and structure of the training data.
    </div>
    """, unsafe_allow_html=True)
with tab2:
    st.markdown("""
    <div class="glass-card">
    The model consists of an <b>Embedding layer</b>, a <b>GRU (Gated Recurrent Unit)</b> RNN layer, and a
    <b>Linear fully-connected layer</b>. The GRU handles the sequential nature of the data and learns
    dependencies over time.
    </div>
    """, unsafe_allow_html=True)
with tab3:
    st.markdown("""
    <div class="glass-card">
    In a <b>Many-to-Many RNN</b>, the model takes a sequence of inputs and produces a sequence of outputs.
    For every character in the input sequence, it predicts the immediate next character — enabling
    efficient training by computing loss over the entire sequence at once.
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================================
# Training Section
# =====================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🏃‍♂️ Model Training</p>', unsafe_allow_html=True)

if not st.session_state.is_trained:
    st.markdown('<span class="badge badge-warn">⚠️ Untrained — output will be random gibberish</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-ok">✅ Trained & ready for generation</span>', unsafe_allow_html=True)

train_clicked = st.button("🚀 Retrain Model", use_container_width=True, type="primary")

if train_clicked:
    st.session_state.model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(st.session_state.model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = max(1, len(data_tensor) // (SEQ_LEN * BATCH_SIZE))

    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        st.session_state.model.train()
        total_loss = 0.0

        for step in range(steps_per_epoch):
            x, y = get_batch(data_tensor, SEQ_LEN, BATCH_SIZE)
            optimizer.zero_grad()
            logits, _ = st.session_state.model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.session_state.model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch

        progress = epoch / EPOCHS
        progress_bar.progress(progress)

        elapsed = time.time() - start_time
        est_total = elapsed / progress if progress > 0 else 0
        rem_time = max(0, est_total - elapsed)

        status_text.markdown(f"**⏳ Training...** Epoch `{epoch}/{EPOCHS}` &nbsp;|&nbsp; Est. remaining: `{rem_time:.0f}s`")

        with metrics_placeholder.container():
            m1, m2 = st.columns(2)
            m1.metric("Current Epoch", f"{epoch}/{EPOCHS}")
            m2.metric("Current Loss", f"{avg_loss:.4f}")

    progress_bar.empty()
    status_text.empty()
    st.session_state.is_trained = True
    st.success("✅ Training completed successfully!")

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# Generation Section
# =====================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">✍️ Text Generation</p>', unsafe_allow_html=True)
st.markdown('<p class="section-caption">Give the model a seed, tune the settings, and watch it write.</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    default_seed = text_data[:100] if len(text_data) >= 100 else text_data
    seed_text = st.text_area("🌱 Seed Text", value=default_seed, height=200, help="Starting text for the model to continue.")

with col_right:
    gen_length = st.slider("📏 Output Length", min_value=50, max_value=1000, value=300, step=50, help="Number of characters to generate.")
    temperature = st.slider("🌡️ Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1, help="Higher values increase randomness, lower values make it more predictable.")
    st.write("")
    generate_btn = st.button("✨ Generate Text", use_container_width=True, type="primary")

if generate_btn:
    if not seed_text:
        st.error("Please enter some seed text!")
    else:
        unknown_chars = [ch for ch in seed_text if ch not in char2idx]
        if unknown_chars:
            st.error(f"Error: These characters in your seed aren't in the training text: {''.join(set(unknown_chars))}")
        else:
            with st.spinner("Generating magic... 🪄"):
                generated_output = generate(st.session_state.model, seed_text, gen_length, temperature)
                time.sleep(0.4)

            st.success("Text generated successfully! 🎉")

            safe_output = html.escape(generated_output)
            st.markdown(f"<div class='output-card'>{safe_output}</div>", unsafe_allow_html=True)

            st.download_button(
                label="📥 Download Generated Text",
                data=generated_output,
                file_name="generated_text.txt",
                mime="text/plain",
                use_container_width=True
            )

st.markdown('</div>', unsafe_allow_html=True)
