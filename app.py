import streamlit as st
import torch
import torch.nn as nn
import time
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Many-to-Many Char RNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2e6c80;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .dark-card {
        background-color: #1e1e1e;
        color: #f1f1f1;
        border-radius: 10px;
        padding: 20px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-family: 'Consolas', 'Courier New', monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Config & Setup
# ----------------------------
FILE_PATH   = "data/text.txt"
SEQ_LEN     = 100
BATCH_SIZE  = 64
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
EMBED_SIZE  = 128
EPOCHS      = 20
LR          = 0.002
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# ----------------------------
# Load data & vocabulary (Cached)
# ----------------------------
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

# ----------------------------
# Model
# ----------------------------
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

# Initialize Model in Session State to retain across reruns
if 'model' not in st.session_state:
    st.session_state.model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    st.session_state.is_trained = False

# ----------------------------
# Generation function
# ----------------------------
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

# ----------------------------
# Streamlit UI
# ----------------------------

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("### Model & Dataset Info")
    st.info(f"**Dataset:** `{FILE_PATH}`")
    st.info(f"**Device:** `{DEVICE}`")
    st.info(f"**Vocab Size:** `{vocab_size}`")
    st.info(f"**Total Chars:** `{len(text_data):,}`")
    
    st.markdown("### Hyperparameters")
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

# --- Main Header ---
st.markdown("<h1 class='main-header'>🧠 Many-to-Many Character-Level RNN Text Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #555;'>Train a GRU-based Recurrent Neural Network to generate text character-by-character.</p>", unsafe_allow_html=True)

# --- Expandable Sections ---
with st.expander("ℹ️ About the Project"):
    st.write("This project demonstrates text generation using a Character-Level Recurrent Neural Network. Given a sequence of characters, the model predicts the most likely next character, allowing it to generate text that mimics the training data.")

with st.expander("🏗️ Model Architecture"):
    st.write("The model consists of an **Embedding layer**, a **GRU (Gated Recurrent Unit) RNN layer**, and a **Linear fully-connected layer**. The GRU handles the sequential nature of the data and learns dependencies over time.")

with st.expander("🔄 How Many-to-Many RNN Works"):
    st.write("In a Many-to-Many RNN, the model takes a sequence of inputs and produces a sequence of outputs. For every character in the input sequence, it tries to predict the immediate next character. This allows efficient training by computing the loss over the entire sequence at once.")

st.divider()

# --- Training Section ---
st.markdown("### 🏃‍♂️ Model Training")
if not st.session_state.is_trained:
    st.warning("The model is currently untrained. Generate output will be random gibberish. Please train the model first.")
else:
    st.success("Model is trained and ready for generation!")

if st.button("🚀 Retrain Model", use_container_width=True, type="primary"):
    # Reset model
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
        
        # Update UI
        progress = epoch / EPOCHS
        progress_bar.progress(progress)
        
        elapsed = time.time() - start_time
        est_total = elapsed / progress if progress > 0 else 0
        rem_time = max(0, est_total - elapsed)
        
        status_text.text(f"Training... Epoch {epoch}/{EPOCHS} | Estimated time remaining: {rem_time:.0f}s")
        
        with metrics_placeholder.container():
            c1, c2 = st.columns(2)
            c1.metric("Current Epoch", f"{epoch}/{EPOCHS}")
            c2.metric("Current Loss", f"{avg_loss:.4f}")
            
    progress_bar.empty()
    status_text.empty()
    st.session_state.is_trained = True
    st.success("✅ Training completed successfully!")

st.divider()

# --- Generation Section ---
st.markdown("### ✍️ Text Generation")

col_left, col_right = st.columns([2, 1])

with col_left:
    # Use text_data[:100] safely
    default_seed = text_data[:100] if len(text_data) >= 100 else text_data
    seed_text = st.text_area("🌱 Seed Text", value=default_seed, height=200, help="Starting text for the model to continue.")
    
with col_right:
    gen_length = st.slider("📏 Output Length", min_value=50, max_value=1000, value=300, step=50, help="Number of characters to generate.")
    temperature = st.slider("🌡️ Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1, help="Higher values increase randomness, lower values make it more predictable.")
    
    st.markdown("<br>", unsafe_allow_html=True)
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
                time.sleep(0.5) 
            
            st.success("Text generated successfully! 🎉")
            
            st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
            st.text(generated_output)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download Generated Text",
                data=generated_output,
                file_name="generated_text.txt",
                mime="text/plain",
                use_container_width=True
            )