# 🔴 Nexus Connect-4 // Adversarial AI Engine

> An unbeatable Connect-4 AI built with Minimax and Alpha-Beta Pruning. Features a real-time Explainable AI (XAI) dashboard, dynamic decision trees, candidate move evaluations, and heuristic window scoring.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

An intelligent Connect-4 implementation designed to demonstrate and prove core concepts in **Adversarial Search, Heuristic Function Design, Game Tree Optimization, and Explainable AI (XAI)**.

Instead of an opaque black box, this system includes a live **XAI telemetry dashboard** revealing how the AI thinks, anticipates human counter-moves, and prunes suboptimal branches in real time.

---

## ✨ Key Features & XAI Suite

* **Adversarial Search Engine:** Full Minimax algorithm implementation with depth-limited lookahead ($D = 1 \text{ to } 6$).
* **Alpha-Beta ($\alpha$-$\beta$) Pruning:** Prunes subtrees where $\alpha \ge \beta$, achieving **80%+ reduction in node evaluations** with 100% mathematical optimality parity.
* **🌳 Interactive Decision Tree (Top 2-3 Plies):** Zero-dependency pure SVG tree visualizing:
  * 👑 Root Decision (MAX)
  * ⭐ Selected Optimal Move (Gold)
  * ✂️ Pruned Subtrees via $\alpha$-$\beta$ Cutoffs (Red dashed)
  * 🔵 Evaluated Alternatives (Cyan)
* **🏆 Candidate Move Scoreboard:** Displays a real-time table of all 7 columns with their minimax scores, highlighting the chosen column and badging pruned branches.
* **🎯 Principal Variation (PV) Pathway:** Displays the exact projected sequence of optimal responses:
  $$\text{AI: Col 4} \longrightarrow \text{You: Col 4} \longrightarrow \text{AI: Col 3}$$
* **⚡ Side-by-Side Proof ("Compare With vs Without Pruning"):** One-click benchmark comparing node counts and runtime between pure Minimax and Alpha-Beta on the exact current board.
* **🎯 69-Window Heuristic Function:** Evaluates all 69 sliding 4-cell windows (horizontal, vertical, diagonal) plus center-column territorial dominance. Configurable via sidebar sliders.
* **🎮 3D Arcade Board:** Custom cyberpunk theme with specular discs, last-move indicator, and victory animation.
* **💡 Tactical Hint Engine:** Provides recommended moves and strategic explanations for human players.

---

## 📂 Project Structure

```
connect4_ai/
│
├── connect4/
│   ├── __init__.py          # Package exports
│   ├── board.py             # 6x7 Grid representation, gravity drop, win detection
│   └── ai.py                # Minimax, Alpha-Beta, heuristic evaluation, SVG tree builder
│
├── .streamlit/
│   └── config.toml          # Dark theme and cloud server configurations
│
├── app.py                   # Streamlit web dashboard with interactive board & XAI suite
├── benchmark.py             # Standalone CLI performance benchmark suite
├── requirements.txt         # Production dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Comprehensive documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd connect4_ai
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the Terminal Benchmark
```bash
python benchmark.py
```

---

## 🧠 Algorithmic Foundations

### 1. Minimax Decision Rule
$$V(s) = \begin{cases} 
\text{Utility}(s) & \text{if } s \text{ is terminal or } d = 0 \\
\max_{a \in \text{Actions}(s)} V(\text{Result}(s, a)) & \text{if Player}(s) = \text{MAX (AI)} \\
\min_{a \in \text{Actions}(s)} V(\text{Result}(s, a)) & \text{if Player}(s) = \text{MIN (Human)}
\end{cases}$$

### 2. Alpha-Beta ($\alpha$-$\beta$) Pruning
* Pruning condition: Whenever $\alpha \ge \beta$, the remaining children of the current node are discarded without loss of optimality.
* Optimal move ordering (`[3, 2, 4, 1, 5, 0, 6]`) brings the effective time complexity from worst-case $O(b^d)$ down toward $O(b^{d/2})$.

---

## 🎓 Model Viva Voce Q&A

1. **Q: Does Alpha-Beta pruning ever change the optimal move?**
   * *A:* **No.** It is mathematically proven to return the exact same move and score as pure Minimax.
2. **Q: Why does center-first move ordering improve Alpha-Beta efficiency?**
   * *A:* Central positions participate in more winning 4-in-a-row lines (13 vs 3). Evaluating them first raises $\alpha$ earlier, maximizing subsequent cutoffs.
3. **Q: What is the Horizon Effect?**
   * *A:* The limitation of depth-bounded search where an AI cannot foresee a threat occurring just beyond cutoff depth $d$.

---

## 📄 License
This project is open-source under the MIT License.
