Customer Support Ticket Multi-Label Classification
==================================================

This project implements and compares two distinct machine learning approaches---**TF-IDF/LinearSVC** and **BERT Transfer Learning**---to classify customer support tickets, assigning multiple relevant categories (labels) to each one.

The core problem addressed is **Multi-Label Text Classification**, where a single ticket can belong to several categories (e.g., "Billing" and "Technical Issue" simultaneously).

🚀 Project Goal
---------------

The primary goal is to build a robust model for automated ticket processing, enabling:

-   **Intelligent Routing:** Directing tickets to the correct support department.

-   **Prioritization:** Assigning urgency based on predicted labels.

-   **Auto-Tagging:** Applying descriptive tags for better analytics.

🛠️ Setup and Installation
--------------------------

### Prerequisites

You need a working Python environment (Python 3.10+ recommended).

### 1\. Clone the Repository

Bash

```
git clone https://github.com/bk-12346/Multi-Label-Customer-Support-Ticket-Classifier
cd Multi-Label-Customer-Support-Ticket-Classifier

```

### 2\. Install Dependencies

This project relies on several major libraries, including TensorFlow/Hugging Face for the BERT component.

Bash

```
# It is highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all required packages
pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers tensorflow

```

### 3\. Data Setup

Ensure the dataset file, **`aa_dataset-tickets-multi-lang-5-2-50-version.csv`**, is placed in the root directory of the project, next to `main.py`.

⚙️ How to Run the Script
------------------------

The script performs all data loading, preprocessing, model training, and evaluation sequentially.

1.  Make sure your Python environment is activated.

2.  Execute the main script from your terminal:

Bash

```
python main.py

```

### ⚠️ Note on BERT Training

The BERT transfer learning phase requires significant computational resources.

-   **GPU Recommended:** For fast training, a dedicated GPU with CUDA support is necessary for the TensorFlow/BERT components.

-   **CPU Warning:** Running the BERT training on a CPU will be **extremely slow** (potentially hours or more). Be prepared to wait, or consider reducing the number of `epochs` in the script if you are limited to a CPU.

📈 Results and Comparison
-------------------------

The script calculates the **Jaccard Score (Sample Average)**, a key metric for multi-label classification, for both models.

| **Model Approach** | **Core Technology** | **Key Metric (Jaccard Score)** | **Computational Cost** |
| --- | --- | --- | --- |
| **TF-IDF Model** | TF-IDF + LinearSVC (Sparse Vectors) | 0.597 | Low (CPU-friendly) |
| **BERT Model** | Transfer Learning (Contextual Embeddings) | 0.315 | High (GPU-intensive) |

### Key Differences Summarized:

-   **TF-IDF** is simpler, faster, and creates **sparse vectors** based on word frequency, treating words independently. It serves as an excellent baseline.

-   **BERT** is more complex, slower to train, but uses pre-trained knowledge to generate **dense, contextual embeddings**, which typically capture semantic relationships better and yield higher performance.

🔗 Dataset Source
-----------------

The dataset used in this project is:

-   **Dataset Name:** multilingual-customer-support-tickets

-   **Source:** Kaggle (by tobiasbueck), https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data

-   **File Used:** `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
