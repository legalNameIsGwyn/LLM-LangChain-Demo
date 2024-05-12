import express from "express";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { 
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { 
  AIMessage,
  HumanMessage
} from "@langchain/core/messages";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

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
      new OllamaEmbeddings({
        model: "nomic-embed-text",
        maxConcurrency: 5,
      })
  );
}

// returns a chain that includes chat history

// sample chat history
// let chatHistory = [
//   new HumanMessage("Does baguio have any problems?"),
//   new AIMessage("Yes!")
// ] 
// const store = await loadDocuments("chatbot");
// const model = await context(store)
// const res = await model.invoke({
//   chat_history: chatHistory,
//   input: "What are they?"
// })
// console.log(res)
export const context = async (vectorStore) => {
  const retriever = vectorStore.asRetriever();

  // Contextualize question
  const contextualizeQSystemPrompt = `
  Given a chat history and the latest user question
  which might reference context in the chat history,
  formulate a standalone question which can be understood
  without the chat history. Do NOT answer the question, just
  reformulate it if needed and otherwise return it as is.`;
  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: contextualizeQPrompt,
  });

  // Answer question
  const qaSystemPrompt = `
  Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Don't mention you got it from what you've read or the context.
  \n\n
  {context}`;
  const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  const questionAnswerChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
  });
  
  return await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: questionAnswerChain,
  });
}




// creates RAG chain given a vector store

// sample way to use
// const store = await loadDocuments("chatbot")
// console.log('Document loaded');
// const model = await createChain(store)
export const createChain = async (vectorStore) => {

  const prompt = ChatPromptTemplate.fromTemplate(`Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Use three sentences maximum and keep the answer as concise as possible.
  Always say "thanks for asking!" at the end of the answer.
  
  {context}
  
  Question: {input}
  
  Helpful Answer:`);

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




app.get("/", (req, res) => {
  res.send("Hello from Express!");
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});