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
  model: "mistral",
});


// Load document
const loader = new TextLoader("./documents/chatbot.txt"); // CHANGE to Terms.txt
const docs = await loader.load();

const app = express();
const PORT = 3000;

// split documents
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 10,
    separators: ["\n\n", "."],
});

const allSplits = await textSplitter.splitDocuments(docs);
  
const vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits,
    new OllamaEmbeddings()
);
const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the users question.
  Context: {context}
  Question: {input}
`);

const chain = await createStuffDocumentsChain({
  llm,
  prompt
})

const retriever = vectorStore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever
});

console.log('Document loaded');

const res = await retrievalChain.invoke({
  input: "What is Baguio infamous for?"
})

console.log(res)

app.get("/", (req, res) => {
  res.send("Hello from Express!");
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});