
Today, it's not just about building bigger models.
It's about building smarter systems.

**Retrieval-Augmented Generation (RAG)** is a breakthrough architecture that combines: 🔹 The knowledge of external data sources (retrieval)
🔹 With the creativity and language understanding of LLMs (generation)

Instead of depending solely on what a model "remembers," RAG fetches the most relevant and up-to-date information in real time — leading to: 

1.More accurate answers

2.Fewer hallucinations

3.Faster adaptation to new knowledge

4.More reliable enterprise applications

**Why RAG matters:**

You don't need to retrain massive LLMs every time your data changes.

You can ensure your AI is always grounded in facts.

You can scale smarter solutions in domains like healthcare, finance, legal, and customer support.

In the world of AI, "retrieval + generation" > just "generation."
Smart architecture wins over brute force.

**Simple RAG example with Langchain, Ollama and and open-source LLM model**

Select the LLM model to use: The model must be downloaded locally to be used, so if you want to run llama3, you should run: "ollama pull llama3"

Check the list of models available for Ollama here: https://ollama.com/library

**Instanciate the LLM model and the Embedding model**

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

model.invoke("Give me an inspirational quote")

Using a parser provided by LangChain, we can transform the LLM output to something more suitable to be read

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
response_from_model = model.invoke("Give me an inspirational quote")
parsed_response = parser.parse(response_from_model)
print(parsed_response)

Generate the template for the conversation with the instruct-based LLM.
We can create a template to structure the conversation effectively.

This template allows us to provide some general context to the Language Learning Model (LLM), which will be utilized for every prompt. This ensures that the model has a consistent background understanding for all interactions.

Additionally, we can include specific context relevant to the particular prompt. This helps the model understand the immediate scenario or topic before addressing the actual question. Following this specific context, we then present the actual question we want the model to answer.

By using this approach, we enhance the model's ability to generate accurate and relevant responses based on both the general and specific contexts provided.

The model can answer prompts based on the context.
But it can't answer what is not provided as context.
Even previously known info!

**RAG** - DocArrayInMemorySearch

Next Load an example PDF to do Retrieval Augmented Generation (RAG). For the example, you can select your own PDF.
Store the PDF in a vector space.
From Langchain docs:DocArrayInMemorySearch is a document index provided by Docarray that stores documents in memory. It is a great starting point for small datasets, where you may not want to launch a database server.

The execution time of the block depends on the complexity and longitude of the PDF provided.

1.Create retriever of vectors that are similar to be used as context.

2.Generate conversation with the document to extract the details.

3.Loop to ask-answer questions continously.
