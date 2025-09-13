# LLM_training
To training large language models

🚀 **Just wrapped up a hands-on NLP project using Hugging Face Transformers and 🤗 Accelerate on Google Colab!**

Over the past few days, I’ve been diving deep into the world of custom training loops, distributed training, and evaluation workflows—all within the constraints of the free-tier Colab environment. Here's a quick walkthrough of what I built and learned:

### 🧠 Project Overview
I trained a BERT-based model for sentence-pair classification using the **GLUE MRPC dataset**, which focuses on identifying whether two sentences are semantically equivalent. The goal was to build a **custom training loop** from scratch using PyTorch and Hugging Face libraries, and then scale it using 🤗 Accelerate for efficient GPU utilization.

---

### 🔧 Key Steps & Highlights

**1️⃣ Dataset Preparation**
- Used `datasets.load_dataset("glue", "mrpc")` to load the benchmark dataset.
- Tokenized sentence pairs using `AutoTokenizer` with truncation and padding to ensure uniform input size.

**2️⃣ Data Formatting**
- Removed unnecessary columns and renamed the label column to `labels`.
- Set the dataset format to PyTorch tensors for seamless integration with the training loop.

**3️⃣ Model & Optimizer Setup**
- Loaded `bert-base-uncased` via `AutoModelForSequenceClassification` with 2 output labels.
- Initialized the optimizer using `AdamW` with a learning rate of 3e-5.

**4️⃣ Accelerator Integration**
- Wrapped the model, dataloaders, and optimizer using `accelerator.prepare()` to enable GPU acceleration.
- This made the code hardware-agnostic and ready for scaling—even on multi-GPU or TPU setups.

**5️⃣ Training Loop**
- Implemented a custom loop with gradient backpropagation using `accelerator.backward()`.
- Included a learning rate scheduler for smoother convergence across 3 epochs.

**6️⃣ Evaluation Workflow**
- Integrated the `evaluate` library to compute metrics like accuracy and F1 score.
- Ran evaluation after each epoch to monitor performance improvements.

---

### 💻 Environment
All of this was executed on **Google Colab’s free GPU backend**, proving that even resource-constrained setups can support robust experimentation with modern NLP tools.

---

### 📊 Chart Description
The line chart compares **Accuracy** and **F1 Score** over three training epochs. It highlights the upward trend in both metrics, showing how the BERT model improved with each pass through the data.

- **Epoch 1**: Accuracy = 0.82, F1 = 0.79  
- **Epoch 2**: Accuracy = 0.85, F1 = 0.83  
- **Epoch 3**: Accuracy = 0.87, F1 = 0.85  

---
### 🌟 Takeaway
This project was a great exercise in understanding the inner workings of training loops, optimizing for hardware, and evaluating model performance—all while staying modular and scalable. Hugging Face’s ecosystem continues to impress with its flexibility and ease of use.

---
#NLP #MachineLearning #HuggingFace #Transformers #GoogleColab #Accelerate #BERT #AI #DeepLearning #Python #DataScience

