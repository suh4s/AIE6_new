<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 17: On Prem Agents</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE5/00_AIM_Quicklinks)

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 17: On-Prem Agents](https://www.notion.so/Session-17-On-Prem-Agents-1c8cd547af3d81559270c07763f674c9?pvs=4#1c8cd547af3d8138ac71ca2992574113) | [Session 17: On-Prem Agents](https://www.notion.so/Session-17-On-Prem-Agents-1c8cd547af3d81559270c07763f674c9) | Coming soon! | [Session 17 Slides](https://www.canva.com/design/DAGjaQa4tas/31ABc-J6xw1CfHH9_XY4Vw/edit?utm_content=DAGjaQa4tas&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [OPTIONAL Session 17: On-Prem Agents](https://forms.gle/yPRmeTk2bQGiMpFh8) | [AIE6 Feedback 5/27](https://forms.gle/SstF4CecDSUTJuXZ8) |

# Build 🏗️

First, let's pull and run our DeepSeek-R1 distilled model through Ollama.

### Mac

1. Download the Ollama app for Mac [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/deepseek-r1:8b):
```bash
ollama pull deepseek-r1:8b
```

### Windows

1. Download the Ollama app for Windows [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/deepseek-r1:8b):
```powershell
ollama pull deepseek-r1:8b
```

### Testing

Once you'd completed that task - please run the `test_ollama.ipynb` notebook to confirm your Ollama instance is working correctly. 

### Ollama Deep Research

Once you've completed that step - please clone the [Ollama Deep Research repository](https://github.com/AI-Maker-Space/Ollama-Deep-Research-Modified) (outside of the AIE6 repository). 

```bash
cd ~
git clone git@github.com:AI-Maker-Space/Ollama-Deep-Research-Modified.git
```

Then you can follow the instructions in that README.md to launch the application. 

### Assignment: 

Add a local RAG instance powered by QDrant into the flow of the existing Agent. 

First, you'll need to: 

```bash
ollama pull mxbai-embed-large
```

Then you will need to: 

1) Launch [QDrant through Docker](https://qdrant.tech/documentation/quickstart/)
2) Create a VectorStore leveraging QDrant as a backend (you can use [this notebook](./qdrant_setup.ipynb) as inspiration)
3) Add a new node in the Agent Graph
4) Modify the graph to accomodate your new node. 

## ADVANCED BUILD:

Implement Semantic Caching with Redis (or similar solution), and add tracing through LangSmith, or WandB.

# Ship 🚢

- 5min. Loom Video

# Share 🚀
- Walk through your app and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned
