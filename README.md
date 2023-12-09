
### Retrieval Augmented LLM Assistant

**Adam Newman**

#### Executive summary

By using a LLM and a vector database, we should be able to allow users to have a conversational interface to talk to a document corpus. The user will have a natural language interface to a corpus of documents they have never seen before. The ML model wouldn't have to be retrained any time the documents change, the system will be able to update the vector database on the fly and immediately use the new data for retrieval. Without any special technical skills users should be able to find answers to their questions that are both factual and helpful for the user to accomplish their goals.

#### Rationale

There are a lot of applications for natural language interfaces to a document corpus. Here are a few benefits that a system like this could provide:

* Onboard new-hires to a complex system more quickly. When a company's scattered documents can be accessed with an LLM's knack for summarization, the user can find the most relevant information and get on with their job.
* Many companies don't feel comfortable or legally cannot upload their documents into the cloud and want to unlock the power of LLMs for their on-premises hardware and data. When tools like these only index the public Internet or require uploading your data to their tool, these companies are left out.
* Personal users that take a lot of notes like daily journals or facts that are specific to them or more important to them can't always find the notes after they've taken them. A tool that can aggregate multiple notes across time and provide a more complete answer could provide a "second brain" for these people.

#### Research Questions

* Can a large language model use a local corpus of notes or documentation to provide answers to natural language questions?
* How do off the shelf models compare with a locally fine tuned model?

#### Data Sources

The data source only needs to be a collection of documents with information that either wasn't present for training or contains information very specific to the author or their organization.

I chose the Apache 2.0 licensed notes PKM at https://github.com/simonw/til. This corpus has some high quality notes that represent personal documentation of a broad range of topics.

#### Methodology

I plan to use the Apache licensed `mistralai/Mistral-7B-v0.1` model as the basis LLM for fine tuning. That model has a large 8k token context and low 7B parameters which they claim performs as well as larger Llama models. That enables it to run well on consumer grade 24GB VRAM GPUs. I will do fine tuning on that model with examples of ideal responses to questions from https://huggingface.co/datasets/timdettmers/openassistant-guanaco. That will turn the raw base model into something that is tuned to do interactive chat.

I will also use the off the shelf models/APIs `mistralai/Mistral-7B-v0.1` (to compare base with tuned models), `mistralai/Mistral-7B-Instruct-v0.1`, `meta-llama/Llama-2-7b-chat-hf`, ChatGPT 3.5 Turbo, and ChatGPT 4 Turbo (preview).

The UX should follow something like:

* User asks a natural language question relevant to the documents.
* Use the ChromaDB vector database to find top N relevant documents.
* Once we've retrieved relevant context documents, feed those documents into the question for the model.
* The model replies to the users question with context from the documents along with links to the source documents used as citations.

#### Results
* Loading a corpus of notes into a vector database is a very effective way to provide context for an LLM.
* Requests without the aid of the vector database gave generic answers from the original training data. When given a corpus of notes to frame the response and a vector database, the models performed better at providing specific information in the response.
* The LoRA fine tuning works. You'll need a decent sized training set of example interactions to tune the baseline model from a generic prediction mode into a chat bot. Using small training sets tends to bias the fine tuned model to memorize exact training examples and reply with them even when they're out of context.

##### Comparison of models
Here is a somewhat subjective comparison of the various models including their performance both without RAG (Retrieval-Augmented Generation) and with RAG for some sample queries.

| Model | Inference Time (s) | Non-RAG Quality | RAG Quality | Context Window | CapEx | OpEx |
|--|--|--|--|--|--|--|
|Locally Tuned |  | C | D | E | $$$$ | $ |
|Mistral-7B-Instruct | 4 - 18 | 3/5 | 2/5 | E | $$$ | $ |
|Llama-2-7B-Chat | 12 - 15 | 3/5 | 4/5 | E | $$$ | $ |
|GPT 3.5 Turbo | 8 - 9 | 3.5/5 | 4/5 | E | 0 | $$ |
|GPT 4 Turbo | 42 - 51 | 4.5/5 | 5/5 | E | 0 | $$$ |

* Inference Time - Based on the two example RAG queries, this was the range of time each took.
* Non-RAG Quality - Without access to the corpus, how well did it explain how to loop a GIF? The more options or depth in the answer, the better.
* RAG Quality - How well did it adhere to the source data when providing the answer? Did it summarize the source well or just pull from its original training? Alternatively did it provide no summarization and show memorized content?
* Context Window - The size of the model's context window in tokens. A larger context window allows for more documents to be input.
* CapEx - Capital expenditure or how much you must spend before getting started. If you're buying your own hardware, there is a large one time up-front cost. The training run also uses quite a bit of energy up front.
* OpEx - Operational cost per inference operation. If you're using cloud GPU or SaaS with per-token inference costs, OpEx is a bit higher. If you're using a local GPU, then you're only paying for electricity.  However, for business use, you will need staff to maintain and upgrade your hardware, redundant infra, etc. so on-prem hardware is never a one time cost.

##### Best Model


#### Next steps

* Determine the minimum `distance` score for a relevant document and exclude documents out of that range. Mention that in the result and provide a best effort answer from the corpus. Providing an arbitrary document that isn't very related to the question isn't useful.
* Reuse the response in the context window for follow up questions.

#### Outline of project

To run this, you'll need a CUDA compatible GPU with 24GB VRAM or some experience in tuning the models to fit into less VRAM. You'll need almost all 24GB, so running a headless machine or using the integrated GPU for your desktop environment is a good idea if you aren't using a cloud VM.

Run the `nvidia-smi` command to see allocated VRAM. Ideally this should be 3MB with your kernels unloaded.

Specs of the machine used for this process:
```
CPU: AMD Ryzen 9 7900X3D
GPU: MSI Suprim Liquid X GeForce 4090 24GB
RAM: 128GB
OS: Ubuntu 23.10
CUDA info: Driver Version: 535.129.03   CUDA Version: 12.2
```

##### Install Requirements

Installing CUDA drivers are out of scope for this project and system/card dependent, but they're required to run these notebooks without tweaking for some other ML acceleration drivers.

Python setup:
```bash
# You don't have to create a virtual environment, but I highly recommend it to prevent polluting your global setup.
python3 -m venv jupyter
jupyter/bin/pip install nltk matplotlib chromadb scikit-learn \
  InstructorEmbedding sentence_transformers peft datasets wandb \
  bitsandbytes trl transformers tensorboardX
```

##### These notebooks need to be executed first to setup the environment:

**IMPORTANT:** Be sure to kill any other kernels before running each notebook to ensure VRAM is cleared.

1. [Load Vector Database](load_vector_database.ipynb) - This populates the local vector database with the document corpus. This step is mandatory.
1. [Fine Tuning](fine_tune.ipynb) - Performs local fine tuning based on the `data/fine_tune.json` file. This takes hours to complete and will produce a locally fine tuned model. You may skip this if you don't want to run the `retrieval_augmented_chat_mistral_tuned.ipynb` notebook.
1. [Create Fine Tuning Data](fine_tune_ganaco.ipynb) - You can skip this notebook. This will regenerate the `data/fine_tune.json` based on data from [openassistant-ganaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco). 

##### Primary notebooks:

**IMPORTANT:** Be sure to kill any other kernels before running each notebook to ensure VRAM is cleared.

* [Retrieval Augmented Chat Tuned](retrieval_augmented_chat_mistral_tuned.ipynb) - Uses the local GPU and locally fine tuned version of `Mistral-7B`.
* [Retrieval Augmented Chat Mistral-7B-Instruct](retrieval_augmented_chat_mistral_instruct.ipynb) - Uses the local GPU and instruction pre-tuned model `Mistral-7B-Instruct`.
* [Retrieval Augmented Chat Llama-2-7B-Chat](retrieval_augmented_chat_llama.ipynb) - Uses the local GPU and instruction pre-tuned model `Llama-2-7b-chat-hf`.
* [Retrieval Augmented Chat ChatGPT 3.5 Turbo](retrieval_augmented_chat_gpt_3_5_turbo.ipynb) - Uses the  ChatGPT API with instruction pre-tuned model `gpt-3.5-turbo`.
* [Retrieval Augmented Chat ChatGPT 4 Turbo](retrieval_augmented_chat_gpt_4_turbo.ipynb) - Uses the  ChatGPT API with instruction pre-tuned model `gpt-4-1106-preview`.
* [Retrieval Augmented Chat Mistral-7B](retrieval_augmented_chat_mistral_base.ipynb) - Only interesting to see how poorly it performs without instruction tuning. Uses the local GPU and raw `Mistral-7B` model without any instruction tuning.

##### Other notebooks:

* [Shared Code](shared_code.ipynb) - Shared classes used by the various primary notebooks to reduce repetition of code and ensure consistency.
* [Tuned model inference test](tuned_model_inference.ipynb) - Useful as a playground for the locally tuned model. 

#### Contact and Further Information

Adam Newman
* Mastodon - @txdevgeek@mastadon.social
* Bluesky - @txdevgeek.bsky.social