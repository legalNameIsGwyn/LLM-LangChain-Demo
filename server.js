import express from "express";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

const llm = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "llama3",
});

const app = express();
const PORT = 3000;

// return vector store of document 
export const loadDocuments = async (fileName) => {
  // Load document
  const loader = new TextLoader(`./documents/${fileName}.txt`); // CHANGE to Terms.txt
  const docs = await loader.load();

  // split documents
  const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 700,
      chunkOverlap: 100,
      separators: ["\n\n", "."],
  });

  const allSplits = await textSplitter.splitDocuments(docs);
    
  return await MemoryVectorStore.fromDocuments(
      allSplits,
      new OllamaEmbeddings()
  );
}

// creates RAG chain given a vector store
export const createChain = async (vectorStore) => {

  const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the users question in no more tha 35 words.
    Context: {context}
    Question: {input}
  `);

  const chain = await createStuffDocumentsChain({
    llm,
    prompt
  })

  const retriever = vectorStore.asRetriever();

  return await createRetrievalChain({
    combineDocsChain: chain,
    retriever
  });
}

// sample way to use
const store = await loadDocuments("chatbot")
console.log('Document loaded');
const model = await createChain(store)

console.log(await model.invoke({
  input: "What is baguio infamous for?"
}))

app.get("/", (req, res) => {
  res.send("Hello from Express!");
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});