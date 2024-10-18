# Sample-Set

## Overview

<p>In that Project I have utilized the langgraph to develop the complete RAG Application in which i have utilized the agents to decide when to fetch the data from the vectordatabase and generate response for provide the user interface i have used the streamlit module to for the user interface where user interact with our chatbot data

For easy integration and Development i have used the Docker Contanarization for easy deployment 
</p>

## Directory Structure

    '''

    /project_directory_structure/
    │
    ├── /app/            
    │   │   ├── /Data_processing/    
    │   │   │   ├── __init__.py
    │   │   │   ├── pdf_ingestion.py
    │   │   │   └── ..                 
    │   │
    │   ├── /backend/             
    │   │   ├── __init__.py
    │   │   ├── vector_db_embeddings.py
    │   │   └── response_gen_model.py                        
    │   │
    │   ├── main.py              
    │   ├── requirements.txt      
    │   ├── .env                 
    │   ├── config.py            # Configuration settings
    │   ├── logging.py           # Logging setup
    │   └── ...                                   
    │                
    ├── Dockerfile             
    └── README.md
    │
    ├── /docs/                 
    │   ├── Gen AI Assignment.pdf           
    │   └── project.pdf             
    
    '''

## How to Run The Application


### 3. Local Development (without Docker)

For development purposes, you can directly run the streamlit app without Docker:

#### Install Dependencies (Optional for Local Development)

First, create a Python virtual environment and install the dependencies from `requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Run the Streamlit Application Locally

You can start the streamlit server locally:

```bash
run python app/main.py
```

The app will be accessible at `http://localhost:8501` locally.


### Docker  Development (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/Ansh9728/Sample-Set.git
cd Sample-Set
```

### 2. Docker Setup

#### Step 1: Build the Docker Image

Run the following command to build the Docker image:

Go to app directory where docker file present
```bash
docker build -t <image_name> .
```

This will create a Docker image tagged as `<image_name>`.

#### Step 2: Run the Docker Container

Once the image is built, you can run the container by executing:

```bash

docker run --name <contianer_name> <image_name>

docker run -d -p 8000:8000 --name <contianer_name> <image_name>
```

- `-d`: Runs the container in detached mode (in the background).
- `-p 8000:8000`: Maps port 8000 of your local machine to port 8000 of the container.
- `--name container_name`: Names the running container as `container_name`.

#### Step 3: Access the Application

Once the container is running, you can access the FastAPI app at:

- **API Base URL**: `http://localhost:8501`
- **Network Url** : ` http://172.17.0.2:8501`
- **External Url** : ` http://157.35.71.55:8501`

### 4. Docker Container Management

#### Stop the Container

To stop the running Docker container:

```bash
docker stop container_name
```

#### Restart the Container

To restart a stopped Docker container:

```bash
docker start <container_name>
```

#### Remove the Container

To remove the Docker container after stopping it:

```bash
docker rm <container_name>
```


## Built With

- [Streamlit](https://streamlit.io/) - Streamlit is an open-source Python framework for data scientists and AI/ML engineers to deliver dynamic data apps -- in only a few lines of code.
- [Docker](https://www.docker.com/) - Containerization platform for running and deploying apps.

---

This `README.md` file provides all necessary steps to run the FastAPI application using Docker, 