# Multimodal LLM Demo: ECG + X-ray + Symptoms

This repository demonstrates a **multimodal fusion pipeline** using real ECG data, X-ray images, and symptom text, combined into a prompt for a lightweight LLM (OPT-350M) to generate medical reasoning.


---

## Features

- Extracts heart rate and ECG statistics from MIT-BIH ECG dataset.
- Extracts image embeddings from X-ray using ViT.
- Extracts text embeddings from patient symptom description.
- Fuses ECG, image, and text embeddings into a single feature vector.
- Generates concise medical interpretation using OPT-350M LLM.
- Fully runnable on **Google Colab free GPU**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/multimodal-llm-demo.git
cd multimodal-llm-demo

Recommended: run in Google Colab to use GPU.

Usage

Open notebooks/multimodal_demo.ipynb in Colab.

Run the notebook cells in order.

The notebook will:

Download sample ECG data (or use your own in data/).

Extract ECG features.

Process the X-ray image to embeddings.

Embed symptom text.

Fuse features and generate reasoning via LLM.

The last cell prints the medical interpretation output.


ECG heart rate and signal mean are scaled for clarity.

The LLM is lightweight (OPT-350M) for free Colab GPU; you can switch to larger LLMs for better output quality.

