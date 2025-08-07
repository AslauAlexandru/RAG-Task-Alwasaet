# RAG-Task-Alwasaet



## Setup and run instructions
You can run in different ways.

**Google Colab**

Run in Google Colab link: [RAG_Task_Alwasaet.ipynb](https://colab.research.google.com/drive/162AdC8tO8Hsu55gUEX63Vm-cooV2wvlK?usp=sharing).

**Dockerfile**

Run with dockerfile (in terminal in VS Code or Github Codespaces or more):

```
docker build -f Dockerfile -t app:latest .
```
```
docker run -p 8501:8501 app:latest
```

Dockerfile install the ``` requirements.txt ``` or ```requirements_pip_freeze.txt```.

If you want to rerun the previous dockerfile in terminal, you need to stop for rerun the container if you don't do this you will have problems with port, just run this in terminal:

```
docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
```

**Streamlit**
Run with streamlit (in terminal in VS Code or Github Codespaces or more):

```
streamlit run app_llamaparse.py --server.port 8501
```

```
streamlit run app_pdfplumber.py --server.port 8501
```

## Architecture overview

**Technical Architecture**

**a.Frontend Development**

The application’s frontend is built using Streamlit, which offers:

-Clean, responsive interface

-PDF upload functionality

-Interactive chat or chatbot interface


**b.Document Processing Pipeline**



## Environment variables and configuration details



## References

How I Built a Local RAG App for PDF Q&A | Streamlit | LLAMA 3.x | 2025: https://medium.com/@seotanvirbd/how-i-built-a-local-rag-app-for-pdf-q-a-streamlit-llama-3-x-2025-383db1ed1399


Linkedin: https://www.linkedin.com/pulse/how-i-built-local-rag-app-pdf-qa-streamlit-llama-3x-2025-bd-0wk6c



Gthub: https://github.com/seotanvirbd/Local-PDF-RAG-Assistant



