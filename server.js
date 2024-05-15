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
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import fs from 'fs'
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
      separators: ["\n\n", "\n", "."],
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
export const context = async (vectorStore, ai) => {
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
    llm: ai,
    retriever,
    rephrasePrompt: contextualizeQPrompt,
  });

  // Answer question
  const qaSystemPrompt = `
  Use the following pieces of retrieved context to answer the questions about Baguio City. IF the question is unrelated to the context, say you don't know or it's not a relevant question. Keep the answer short and concise.".
  \n\n
  {context}`;
  const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  const questionAnswerChain = await createStuffDocumentsChain({
    llm: ai,
    prompt: qaPrompt,
  });
  
  return await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: questionAnswerChain,
  });
}
//"llama2", "llama3", "llama3-chatqa", "llava-llama3"
const models = ["llama2", "llama3", "llama3-chatqa", "llava-llama3"]
const questions = [
  "What are must-visit tourist locations in the city",
  "When should I visit",
  "Are there any events I should look into",
  "How do I get around the city",
  "is there anything I should be weary of",
  "Overall, would you say the city is worth visiting",
]


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

const store = await loadDocuments("chatbot2")

let data = []

for(let i = 0; i < models.length; i++){
  let startTime, endTime
  let modName = models[i]
  const ai = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: modName,
  });
  console.log(`\n============================================`,ai["model"])

  const chain = await context(store, ai)
  let chatHistory = [] 
  let outHistory = []
  
  for(let j = 0; j < questions.length; j++){

    let question = questions[j]
    console.log(`\nYou: `,question)
    startTime = performance.now()
    const res = await chain.invoke({
      chat_history: chatHistory,
      input: question
    })
    endTime = performance.now()

    let ans = res["answer"]
    let perf = (endTime-startTime)

    console.log(`AI: `,ans)
    console.log(`Time: `, perf)

    chatHistory.push(new HumanMessage(question))
    chatHistory.push(new AIMessage(ans))

    outHistory.push({
      question: `Q${j}: ${question}`,
      answer: `A${j}: ${ans}`,
      time: `T${j}: ${perf}`
    })
  }
  
  data.push(outHistory)
  const dataString = outHistory.map(entry => `${entry.question}\n${entry.answer}\n${entry.time}`).join('\n\n')

  fs.writeFile(`outputs2/${modName}Output.txt`, dataString, (err) => {
    if (err) {
      console.error('Error writing to file:', err);
    } else {
      console.log('Data saved to file successfully!');
    }
  });
}
 
console.log("\nCOMPLETED")



// const ai = new ChatOllama({
//   baseUrl: "http://localhost:11434", 
//   model: "llama2",
// })

// const chain = await context(store, ai)
// let chatHistory = [] 

// for(let j = 0; j < questions.length; j++){

//   let question = questions[j]
//   console.log(`\nYou: `,question)

//   const res = await chain.invoke({
//     chat_history: chatHistory,
//     input: question
//   })

//   let ans = res["answer"]

//   console.log(`\nAI: `,ans)

//   chatHistory.push(new HumanMessage(question))
//   chatHistory.push(new AIMessage(ans))
// }
// console.log(chatHistory)

app.get("/", (req, res) => {
  res.send("Hello from Express!");
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}/`);
});