# Baguio Environment ChatBot Demo
This chatbot uses a local instance of ollama using the langchain framework to answer questions about Baguio City's environment based on data gathered by [Insert Group Members]. Its main purpose is to show how LLM's.

# Python Setup Guide

1. Clone the repo.
2. Make sure to have [Docker](https://www.docker.com/) running. 
3. Run the command below in CMD. 
```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
4. Run the command below to install and run the phi3.
```
docker exec -it ollama ollama run phi3
```
5. Run ```demo.ipynb``` via VSCode.


# Javascript Setup Guide 

1. Clone the repo.
2. Make sure to have [npm](https://www.npmjs.com/)installed and an [Ollama](https://ollama.com/download) LLM running. 
3. Navigate to the cloned repo through CMD.
6. Run the command below to install required node modules.
```
npm install
```
7. Run the command below to start.
```
npm run dev
```