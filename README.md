# Baguio Environment ChatBot Demo
This chatbot uses a local instance of ollama using the langchain framework to answer questions about Baguio City's environment based on data gathered by [Insert Group Members]. Its main purpose is to show how LLM's.

# Setup Guide
This setup guide is intended for those who wish to try the project.

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
