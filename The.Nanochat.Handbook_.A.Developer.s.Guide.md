# **The Nanochat Handbook: A Developer's Guide**

This document is a technical deep-dive into the nanochat repository. It is not a tutorial for beginners but a comprehensive handbook for developers, data scientists, and researchers who want to understand, run, and modify this codebase to build their own models.

We will trace the entire execution flow, starting from the master `speedrun.sh` script
, and break down the "why" behind each component and design decision.

## **0\. Prerequisites: The Environment**

Before any code runs, the environment must be built. This is the purpose of the first section of `speedrun.sh`, lines 18-36.  

* **File:** speedrun.sh  
* **Commands:** uv venv, uv sync \--extra gpu  
* **File:** `pyproject.toml`

Q: What is the primary problem that environment management tools like uv and files like pyproject.toml solve?  
A: They solve the "it worked on my machine" problem. pyproject.toml and the associated uv.lock file.

lock the *exact versions* of all dependencies (e.g., torch\>=2.8.0). This ensures the code is 100% reproducible. AI/ML models are highly sensitive to subtle changes in math operations between library versions (e.g., a change in the F.cross\_entropy implementation in PyTorch could silently alter the model's performance). This file guarantees a stable and identical environment for anyone running the script.

## **1\. The Core Architecture: The "Brain"**

Before we can *train* a model, we must *define* it. The core architecture of our Transformer is defined in `nanochat/gpt.py`

### **1.1. GPT(nn.Module)**

This is the main class that assembles the model. It is composed of three parts:

1. **wte (Word Token Embedding):** A lookup table that maps token IDs (e.g., 50123\) to meaning vectors (e.g., a n\_embd dimension vector). This is the "dictionary."  
2. **h (Transformer Blocks):** A stack of n\_layer (e.g., 20\) Blocks. This is the "thinking" part.  
3. **lm\_head (Language Model Head):** The final layer that maps the processed meaning vectors back to logits over the entire vocabulary.

Note: In nanochat, wte and lm\_head are *untied* (they do not share weights), which is a modern design choice that gives the model more flexibility, especially with large vocabularies, at the cost of more parameters.

### **1.2. Block(nn.Module)**

This is the repeating unit of the Transformer. The flow of data through a block is: norm \-\> attention \-\> residual\_add \-\> norm \-\> mlp \-\> residual\_add. This "pre-normalization" (applying norm *before* the operation) is a key design choice for training stability. Each Block has two main sub-layers:

1. **attn \= CausalSelfAttention(...):** gpt.py, line 59  
   The "communication" layer.  
2. **mlp \= MLP(...):** gpt.py, line 111  
   The "computation" layer.

Q: What is the conceptual difference in their jobs?  
A: The CausalSelfAttention layer mixes information between tokens in the sequence (letting them "talk" to each other). The MLP layer then processes this new information for each token individually (letting each token "think" about what it just heard).  
This Block also uses **residual connections** `x = x + self.attn(...)`

Q: Why add x (the input) back to the output?  
A: This is a residual connection, one of the most important concepts in deep learning. It allows the model to "skip" the layer if it's not useful. This makes training "deep" (many-layered) models stable. Without it, the "slope" (gradient) would vanish as it traveled back through 20+ layers. This connection provides a "shortcut" for the gradient, ensuring the "blind hiker" (optimizer) can always find its way.

### **1.3. CausalSelfAttention(nn.Module)**

This is the "magic" of the Transformer.

* It creates three matrices from the input token: **Query (Q)**, **Key (K)**, and **Value (V)** 
* Think of **Q** as your search query (e.g., "what verb am I?").  
* Think of **K** as the 'keywords' for all other tokens (e.g., "I am a noun" or "I am an adjective").  
* Think of **V** as the actual 'value' or information that token holds.  
* The attention mechanism calculates a similarity score between your Q and all other K's, then uses those scores to create a weighted sum of all the V's.  
* **Result:** The output for one token is a new vector that has "mixed in" relevant information from all other tokens.

Q: Why "Causal" self-attention?  
A: It prevents the model from "cheating" during training. In nanochat/gpt.py, this is done by setting `is_causal=True` in `F.scaled_dot_product_attention` This applies a "look-ahead mask," ensuring that when predicting the *next* token, the model can *only* see tokens from the past. This is essential for a generative chatbot, which must generate one token at a time based only on what came before.

## **2\. The Training Pipeline (Tracing speedrun.sh)**

This pipeline is a 4-stage process. We will follow it in order.

### **2A. Tokenization (The "Dictionary")**

* **Script:** scripts/tok\_train.py  
* **Helper:** nanochat/tokenizer.py  
* **Core:** rustbpe/src/lib.rs

Q: Why do we tokenize at all? Why not just split by spaces?  
A: If we split by spaces, the vocabulary (list of unique words) would be infinite ("dog", "dogs", "Doggy"). The model would see these as three unrelated numbers.  
Q: So why use Byte Pair Encoding (BPE)?  
A: BPE is the perfect compromise. scripts/tok\_train.py runs the BPE algorithm (from `rustbpe` tokenizer.py) over the data. This algorithm finds the most common *sub-words* (like "ing", " straw", " berry") and merges them into single tokens. For instance, the word 'tokenization' might be split into \['token', 'ization'\]. This is powerful because the model can now apply the *concept* of 'ization' (the process of making something) to other words.

* **Benefit 1:** We get a fixed-size vocabulary.  
* **Benefit 2:** The model can now "see" that "dog" and "dogs" are related because they share a base token.  
* **Benefit 3:** It can build *any* new word from these sub-word pieces.

Q: Why 65,536 tokens?  
A: It's $2^{16}$. This is a standard "sweet spot" that is computationally efficient. It's large enough to get good compression (capturing most common words and sub-words) but small enough to keep the model's "dictionary" layers (wte and lm\_head in nanochat/gpt.py) fast and memory-efficient.

Q: Why is the BPE logic in Rust?  
A: Speed. scripts/tok\_train.py must count and merge pairs across billions of characters. Python is very slow at this. The author wrote this single performance bottleneck in Rust (`rustbpe/src/lib.rs`) to make tokenizer training feasible.

### **2B. Pre-training (The "Engine of Knowledge")**

* **Script:** scripts/base\_train.py
* **Purpose:** This is the longest, most expensive step. Its goal is to build the "engine" of raw knowledge by training the model on massive amounts of raw internet text (from nanochat/dataset.py) 
* **Task:** Predict the next token.

This script combines three key concepts:

**1\. Data Loading (nanochat/dataloader.py)**

* This script streams text from the dataset, uses the tokenizer we just made, and prepares (x, y) batches for the GPU. x is a chunk of tokens (e.g., \[The, quick, brown, fox\]), and y is the same chunk shifted by one (e.g., \[quick, brown, fox, jumps\]). This is how "next token prediction" is formatted.

**2\. Distributed Training (torchrun)**

* **Command:** `torchrun --standalone --nproc_per_node=8 -m scripts.base_train ...` (speedrun.sh)
* **Q: How do 8 GPUs work together?**  
* **A:** This is **Data-Parallel Training**.  
  1. torchrun copies the *same* model to all 8 GPUs.  
  2. The data is split into 8 chunks. Each GPU processes its own chunk *in parallel*.  
  3. Each GPU *independently* calculates the "slope" (gradient) for its chunk.  
  4. A crucial step (dist.all\_reduce) happens: all 8 GPUs "shout" their slope to each other, average them, and get *one* single, more accurate slope. This synchronization is computationally expensive but necessary.  
  5. All 8 GPUs then take the *exact same* step downhill together.  
* This is what the DistAdamW `adamw.py`  
  and DistMuon `muon.py`  
  optimizers manage. They are distributed-aware.

**3\. Optimization (nanochat/gpt.py, lines 205-231)**

* The base\_train.py script calls `model.setup_optimizers(...)`, which is defined in nanochat/gpt.py.  
* **Q: Why two different optimizers (AdamW and Muon)?**  
* **A:** This is a specific design choice. The setup\_optimizers function in gpt.py splits the model's parameters into two groups:  
  1. **The "Dictionary" (Embeddings):** wte and lm\_head. These are *sparse* (only a few tokens are updated each step). The wte table for 65,536 tokens is massive, but on any given batch, only a tiny fraction (e.g., \~1000 unique tokens) are actually used. The script uses AdamW for these, as its "adaptive" nature (the "smart boots" from our analogy) is good at handling sparse updates.  
  2. **The "Brain" (Matrices):** The main Transformer Blocks. These are *dense* (all parameters are used on every step). The script uses Muon, which is a specific optimizer (SGD+Momentum with an orthogonalization step) that the author believes is more stable for this kind of dense, core computation.

**End of this stage:** We have a "base model" checkpoint. It's a powerful "engine of knowledge" but has no "dashboard" or "steering wheel." It's a raw text-completion engine, not a chatbot.

### **2C. Mid-training (The "Bootcamp")**

* **Script:** scripts/mid\_train.py
* **Purpose:** To be the *bridge* between the raw base model and the final chatbot. It teaches the model new *skills* and the *format* of conversation.  
* **Key Differences from Pre-training:**  
  1. **Loads a Model:** It *loads* the base model checkpoint (`load_model("base", ...)`).  
  2. **New Data:** It uses a TaskMixture
     of high-quality, specialized datasets. This is the "bootcamp" phase. The goal is to rapidly "level up" the model's skills beyond just text completion. By mixing in tasks like
      - GSM8K (math/tool-use) `tasks/gsm8k.py`
      - MMLU (Q\&A) `tasks/mmlu.py`
      - SmolTalk (chat) `tasks/smoltalk.py`
      - SpellingBee `tasks/spellingbee.py`
      - and the identity\_conversations.jsonl `mid_train.py`
      
      the model learns new, valuable *behaviors* it never would have from raw internet text.  
  3. **New Token Format:** The dataloader (mid\_data\_generator) calls tokenizer.render\_conversation. This is the *first time* the model sees the special chat tokens like \<|user\_start|\> and \<|assistant\_start|\>

**End of this stage:** We have a "mid-trained" checkpoint. The model now understands the *format* of a conversation and has been "warmed up" on new, high-quality skills.

### **2D. Supervised Finetuning (SFT) (The "Special Ops Polish")**

* **Script:** scripts/chat\_sft.py
* **Purpose:** To laser-focus the model on one job: being a helpful assistant.

Q: What's the real difference between Mid-training and SFT?  
A: The **loss function**. SFT uses a masked loss.

* The `sft_data_generator` in scripts/chat\_sft.py gets both ids and a mask from the tokenizer. This mask is 0 for user tokens and 1 for assistant tokens see [tokenizer](nanochat/tokenizer.py), line 316 
* The script then sets all targets where the mask is 0 to \-1 (the ignore index)
* **Why this matters:** The model is *only* trained to predict the assistant's tokens. It feels *no loss* and learns *nothing* from the user's side of the prompt. This "polishes" its ability to follow instructions and generate high-quality replies, given a prompt. If the loss was *not* masked, the model would get just as 'rewarded' for predicting the *user's prompt* as it would for predicting the *assistant's answer*. This masking stops the model from learning to be a 'parrot'.

**End of this stage:** We have a fully SFT-trained model, ready for evaluation and chatting.

## **3\. Evaluation and Inference**

### **3A. Evaluation (The "Final Exam")**

* **Script:** scripts/chat\_eval.py
* **Purpose:** To get an objective, numerical score of our model's quality.  
* **Method 1**: Categorical (run\_categorical\_eval)  
  * **For:** [MMLU](/tasks/mmlu.py), [ARC](/tasks/arc.py)(multiple choice).  
  * **"Why":** It's fast. It doesn't generate text. It just checks the model's logits (probabilities) for the *single correct token* (e.g., "A", "B", "C", or "D").  
* **Method 2**: Generative (run\_generative\_eval(scripts/chat\_eval.py))
  * **For:** [GSM8K](/tasks/gsm8k.py), [HumanEval](/tasks/humaneval.py) (math, code).  
  * **"Why":** The answer is a full solution. This function calls engine.`generate_batch(...)` in `/scripts/chat_eval.py` to generate the full answer, then uses the task's specific `.evaluate()` function (e.g., GSM8K's function knows how to parse the text and check if the final number is correct). This is why the fast 'categorical' method cannot be used.

### **3B. Inference (The "Chatbot" App)**

* **Script:** scripts/chat\_web.py
* **Core:** nanochat/engine.py
* **Purpose:** To load the final SFT-trained model and use it. This is also called "autoregressive generation."

Q: What is the primary purpose of the KVCache in nanochat/engine.py?  
A: It is a critical performance optimization for autoregressive generation.

* **Problem:** To generate 100 tokens, you must run the model 100 times, once for each token. Without a cache, the model would re-process the *entire* prompt+history every single time. Generating the 100th token would mean re-processing the first 99 tokens. This is computationally explosive.  
* **Solution:** The KVCache (defined in nanochat/engine.py, line 131\) stores the **K** and **V** matrices (from the self-attention layer) for all *past* tokens. When generating a new token, the model only needs to compute Q, K, and V for the *single newest token* and attend to the cached K/V matrices from the past. This makes generation extremely fast. Without it, 1 \+ 2 \+ ... \+ 100 \= \~5000 token computations would be needed. With it, only 100 are.

## **4\. Advanced Scenarios: "What If?"**

This section covers the "what's next" based on our Q\&A.

### **4A. The "Counting 'r's in Strawberry" Problem**

Q: Why do base models fail this?  
A: Tokenization. The model doesn't see s-t-r-a-w-b-e-r-r-y. It sees one token, e.g., 50123 ("strawberry"). It has no access to the spelling, only the concept. Asking it to count letters in 50123 is nonsensical.  
**Solution 1 (Brute-Force):** Train it on the SpellingBee [task](/tasks/spellingbee.py) during mid-training. The model learns a *pattern* ("how many 'x' in 'y'") but not the *process* of counting. This is brittle.

**Solution 2 (Tool Use):** This is the robust nanochat way.

1. **The Tool:** The nanochat/engine.py script has a `use_calculator` function that is explicitly coded to support Python's `.count()` method
2. **The Training:** You finetune the model (in SFT) on examples showing it how to *call* this tool:  
   * User: "How many 'r's in 'strawberry'?"  
   * Assistant (generates this): \<|python\_start|\>"strawberry".count('r')\<|python\_end|\>  
3. **The Result:** The engine.py sees these tokens, *executes the Python code*, gets "2", and feeds it back to the model, which then gives the final answer. The model's "skill" is shifted from *knowing how* to *count* (which it is bad at) to *knowing when to call a tool* (which it is good at).

### **4B. Customizing: "Chatbot" vs. "Codebot"**

Q: How would we build a "nano-codebot"?  
A: You must change both knowledge and skill.

1. **Change Knowledge (Pre-training):** You would *replace* the FineWeb-Edu dataset (general text) with billions of lines of GitHub code and technical documentation. This builds an "engine" that "thinks" in code. If you *only* did this, you'd have a great code completion tool, but it wouldn't be a "chatbot."  
2. **Change Skills (Finetuning):** You would *replace* the TaskMixture in [mid\_train.py](/scripts/mid_train.py) and [chat_sft.py](/scripts/chat_sft.py). You would remove SmolTalk and add datasets like [HumanEval](/tasks/humaneval.py) and other "code instruction-following" datasets.

### **4C. Advanced Alignment: SFT vs. RL**

* **SFT:** "Training by the Cookbook." Teaches the model to *imitate* a known, good answer. This is what [chat_sft.py](/scripts/chat_sft.py) does.  
* **RL:** "Training by the Taste Test." Teaches the model to *improve* based on a "score." This is what [chat_rl.py](/scripts/chat_rl.py) does.

Q: Why not replace SFT with RL?  
A: You can't. RL is "trial and error." A raw base model would generate pure nonsense, get a 0/10 reward every time, and learn nothing. SFT is the prerequisite: it trains the model to be a competent 7/10 assistant first. RL is the final polish that nudges it from a 7/10 to a 9/10.  
Q: How does `chat_rl.py` work?  
A: It uses a PPO-like algorithm.

1. **Two Models:** It loads the SFT model twice:  
   * policy\_model: The one we are training.  
   * ref\_model: A *frozen* copy used as a "leash."  
2. **The "Reward":** It generates a math solution. The `train_task.reward(...)` (which just calls `GSM8K.evaluate`) gives a 1.0 or 0.0.  
3. **The Loss:** The final loss is `loss = loss.mean() + kl_beta * kl`
   * loss.mean(): The policy gradient loss. This tells the model "do more of what got you the 1.0 reward."  
   * kl_beta * kl: The **KL Divergence** penalty. This is the "leash." It measures how "far" the policy\_model is drifting from the "sane" ref\_model. This stops the model from finding a loophole (e.g., just printing "42") to get the reward.

   *N.B.* Not the case anymore, the relevant code is the following for loss:
  ```python
   # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
  ```
  Reasoning is provided [here](https://github.com/karpathy/nanochat/discussions/189#discussion-9078543).

### **4D. Advanced Architectures**

Q: What about Mixture of Experts (MoE)?  
A: This is an architectural change you would make to nanochat/gpt.py \[cite: karpathy/nanochat/nanochat/gpt.py\] before pre-training.

* **Problem:** The single MLP layer in each Block inside `/nanochat/gpt.py` must be a "generalist" and learn math, spelling, and chatting all at once.  
* **MoE Solution:** You would replace that *one* MLP with a *team* of 8 "specialist" MLPs and a "gating network" that routes each token to the correct expert (e.g., "spelling" tokens go to Expert 3, "math" tokens go to Expert 1). This allows the model to have *far more* parameters (knowledge) while keeping the *inference cost* (FLOPs) low.