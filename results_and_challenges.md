
## Challenges and Resolutions

Throughout the development of this RAG-based IEP goal generator, several challenges were addressed. Below is a summary of each and how they were resolved.


### Choosing the Right LLM

Ran into several issues when selecting the right LLM for this project. 
- Many low quality or smaller models resulted in vague and repetitive output - simply repeated the prompt and lacked reasoning.
- The LLM, llama-cpp-python didn't seem to work on windows - I couldn't figure out how to debug quickly so I avoided this local set up entirely.
- Hit quota limit for Open AI - then decided to switch to a hugging face model.
- Calling `mistralai/Mistral-7B-Instruct-v0.1` using `InferenceClient` caused a `StopIteration` error because the model is not hosted on Hugging Face’s TGI backend. 

**Resolution**  
- Swapped the model to one compatible with the Inference API: `HuggingFaceH4/zephyr-7b-beta`, which supports the `text_generation` task and is freely accessible.  

---

### Long Runtime for Document Processing

**Problem**  
The initial document preprocessing pipeline (`load_and_embed.py`) was slow due to:
- Sequential HTTP and PDF parsing
- Synchronous chunking and embedding
- No skip logic for repeat runs

**Resolution**  
- Used `aiohttp` for asynchronous URL requests
- Introduced parallel chunking via `ThreadPoolExecutor`
- Batched document embeddings for efficiency
- Added logic to skip rebuilding the vector store if it already exists

---

### Kernel Crashing During Text Generation

**Problem**  
Using the Zephyr-7B model locally caused kernel crashes due to memory exhaustion (~13–16GB RAM required).

**Resolution**  
- Switched to the Hugging Face Inference API with `InferenceClient`
- Authenticated using an API token
- Trimmed prompt context to avoid excessive input size

---

### Incomplete or Empty Model Outputs

**Problem**  
The model sometimes returned only part of the response (e.g., only “Transition Services”), likely due to unclear prompt structure or truncation.

**Resolution**  
- Rewrote the prompt with a clearly structured template
- Added explicit role instructions and categories
- Ended the prompt with a signal (`Response:`) to guide generation
- Trimmed retrieved context length to avoid token overflow

---

