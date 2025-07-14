# Transformer (nano-GPT) from Scratch in PyTorch

This repository contains my **from-scratch implementation of GPT**, built using **PyTorch**, inspired by the [Andrej Karpathy](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=OZU5Tu36Cb59U_YW) video series. This was a hands-on project where I wrote the entire transformer-based GPT model by myself — without copying code to deepen my understanding of how large language models (LLMs) work under the hood.

---

## Highlights

* **Model**: Mini-GPT with \~**5.8 million parameters** (5,857,857 to be precise ;))
* **Training Data**: Complete works of **William Shakespeare**
* **Training Platform**: Google Colab on **NVIDIA T4 GPU**
* **Training Time**: \~15 minutes / 5,000 iterations
* **Tokenization**: Character-level (vocab size was just 65)
* **Loss**: Ended up at 0.8154 (Overfitted)

---

## Model Architecture

This project follows a **GPT-style decoder-only transformer**, composed of:

* **Embedding Layer**: Token and positional embeddings
* **Multiple Transformer Blocks**:

  * LayerNorm (Pre-Norm)
  * Multi-Head Self-Attention (with causal masking)
  * Feedforward MLP with ReLU (instead of GeLU)
  * Residual connections
* **Output Head**: Linear projection to vocabulary logits
* **Dataset**: `input.txt` (Shakespeare) [Here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
* **Context Size**: 128
* **Optimizer**: AdamW
* **Loss**: Cross-entropy loss


---


### Example Output (after training):

*Still gibberish in structure, but demonstrates that the model is learning Shakesperean english-like rhythm and semantic flow.*

```
KING EDWARD IV:
Soft, tell me, is it doth. Answer I,
Let it remembers smarves fetch at roar'd,
As you shall lay the battle subject and deceive
And dissolute your lusty for your debt.

ROMEO:
That she is tooth touch by my descent.
Sir, bear the swines he knew my cousins
Before that thy father rises with thy cheeks, which before
My company may half made special we in our mistress.

OXFERDIUS:
Come, the fault
That I know the chances of judgment did
Dream out that which is a the house,
Most dear, by the mire you almost for
The labour in the plebeiring home. Sirrah, be gone
To prevent the fiery eye:self our friends, the blood of
Claudious lifer, whom a dispatch for a purpose
Worth that point where the duke us innocenct.
I hope it too. Where is that frighted,
And when cease that broughts me shall both remove
But from this turning to attendings:
Some could ye it offended by a subject,
She drink me frights my inforce times be with heaven,
Not shall come to strange. Your nation
Starts doings 
```

---

## Learnings & Takeaways

This project was a **huge learning experience**, especially in these areas:

* The inner workings of **multi-head self-attention**
* Proper use of **causal attention masks**
* Understanding **transformer blocks** and **residual paths**
* Writing clean **training loops**, batching, and generation logic
* Debugging gradients, model size, and learning stability
* PyTorch (Phew!)

---

## Things to Improve (Next Steps)

* Average the loss over batch rather than per iteration
* Add metrics like perplexity
* Use gradient clipping for better stability
* Save/load model checkpoints
* Scale to larger models (e.g. **GPT-2 with 124M parameters**)
* Explore LLaMA-style architecture:

  * KV Cache
  * Rotary embeddings
  * FlashAttention
