### Retrieval Augmented LLM Assistant

**Adam Newman**

#### Executive summary

By using a LLM and a vector database, we should be able to allow users to have a conversational interface to talk to a document corpus. The user will have a natural language interface to a corpus of documents the user has never seen before. The ML model wouldn't have to be retrained any time the documents change, the system will be able to update the vector database on the fly and immediately use the new data for retrieval. Without any special technical skills users should be able to find answers to their questions that are both factual and helpful for the user to accomplish their goals.

#### Rationale

There are a lot of applications for natural language interfaces to a document corpus. Here are a few benefits that a system like this could provide:

* Onboard new-hires to a complex system more quickly. When a company's scattered documents can be accessed with an LLM's knack for summarization, the user can find the most relevant information and get on with their job.
* Many companies don't feel comfortable or legally cannot upload their documents into the cloud and want to unlock the power of LLMs for their on-premises hardware and data. When tools like these only index the public Internet or require uploading your data to their tool, these companies are left out.
* Personal users that take a lot of notes like daily journals or facts that are specific to them or more important to them can't always find the notes after they've taken them. A tool that can aggregate multiple notes across time and provide a more complete answer could provide a "second brain" for these people.

#### Research Question
Can a large language model use a local corpus of notes or documentation to provide answers to natural language questions?

#### Data Sources

The data source only needs to be a collection of documents with information that either wasn't present for training or contains information very specific to the author.

I chose the Apache 2.0 notes PKM at https://github.com/simonw/til. This corpus has some high quality notes that represent personal documentation of a broad range of topics.

#### Methodology
I plan to use the Apache licensed Mistral 7B model as the basis LLM. That model has a large 8k token context and low 7B parameters which they claim performs as well as larger Llama models. I hope to do fine tuning on that model with a few examples of ideal responses to questions. That should save some context window if the tuned model is pre-tuned to be one shot or zero shot. There will be at least two modes:

* Tool use for querying the document database.
* Responding to the user with informative summaries.

The UX should follow something like:

* User asks a natural language question relevant to the documents.
* The model may rephrase the question or extract important key words to then turn into embeddings.
* Use the ChromaDB vector database.
* Once we've retrieved relevant context documents, replay the users question with the documents or segments of the documents in the context window.

Two stretch goals:

* Provide source links or the ability to see quoted text from the source that the model used as citations below the answer. Showing the original documents allows the user to ensure that the answer wasn't hallucinated. Keeping the temperature pretty low should help.
* Retain context for follow up questions while appending the next search. This may strain the context window depending on how efficient I can be with search.

#### Results
What did your research find?

#### Next steps
What suggestions do you have for next steps?

#### Outline of project

To run this, you'll need a CUDA compatible GPU with 24GB VRAM or some experience in tuning the models to fit into less VRAM. You'll need almost all 24GB so running a headless machine or using the integrated GPU for your desktop environment is a good idea if you aren't using a cloud VM.
Run the `nvidia-smi` command to see allocated VRAM. Ideally this should be 3MB with your kernels unloaded.

##### Install Requirements

Installing CUDA drivers are out of scope and system/card dependent, but they're required to run these notebooks as-is.

Python setup:
```bash
# You don't have to create a virtual environment, but I highly recommend it to prevent polluting your global setup.
python3 -m venv jupyter
jupyter/bin/pip install nltk matplotlib chromadb scikit-learn \
  InstructorEmbedding sentence_transformers peft datasets wandb \
  bitsandbytes trl transformers tensorboardX
```

##### These notebooks need to be executed first to setup the environment:

1. [Fine Tuning](fine_tune.ipynb) - Be sure to kill any other kernels to ensure VRAM is cleared. This will use almost all 24GB.
1. [Load Vector Database](load_vector_database.ipynb)

##### Primary notebook:

* [Retrieval Augmented Chat](retrieval_augmented_chat.ipynb) - Be sure to kill any other kernels to ensure VRAM is cleared.

##### Other notebooks:

* [Tuned model inference test](tuned_model_inference.ipynb)

#### Contact and Further Information

Adam Newman
* Mastodon - @txdevgeek@mastadon.social
* Bluesky - @txdevgeek.bsky.social