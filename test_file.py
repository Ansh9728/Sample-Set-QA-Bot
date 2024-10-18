import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


"""################## util function #############"""
# Loading the LLM model

def get_llm_model(model="llama3-70b-8192"):
    model = ChatGroq(
        temperature=0,
        model=model,
        api_key = os.getenv('GROQ_API_KEY'),
        streaming=True
    )

    return model


# Create a Retrivel tool

def get_retrivel_tool(retriever):
    retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_documents",
    "Efficiently search and return relevant documents based on the user's query, providing accurate and timely information to support decision-making."
    )
    tools = [retriever_tool]

    return tools

# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_documents",
#     "Efficiently search and return relevant documents based on the user's query, providing accurate and timely information to support decision-making."
#     )

# tools = [retriever_tool]



"""##################### Retrivel Agent Defining #########################"""
# Defining the Agent state

class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]


# Agent Node

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """

    print("--- Call Agent ---")

    messages = state['messages']

    model = get_llm_model()

    model.bind_tools(tools)

    response = model.invoke(messages)

    return {"messages": [response]}


# Grade Edge

def grade_documents(state) -> Literal['generate', 'rewrite']:
    """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
            
    """
     
    print("---CHECK RELEVANCE---")

    # Data Model
    class grade(BaseModel):
        """Binary score for the relevence check"""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    
    model = get_llm_model(model="llama3-groq-70b-8192-tool-use-preview")

    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(

        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state['messages']
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score=="yes":
        print("---DECISION: DOCS RELEVANT---")
        return 'generate'
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return 'rewrite'

# Question Rewriter

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = get_llm_model(model="gpt-4-0125-preview")
    response = model.invoke(msg)
    return {"messages": [response]}



# generate Node
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """

    print("---GENERATE---")

    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    llm = get_llm_model()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})

    return {"messages": [response]}

# Check Hallazunation in response
def check_hallucination(state):

    # Data Model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    # LLM with fucntion call

    llm = get_llm_model()

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    messages = state["messages"]
    generation = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    hallucination_grader = hallucination_prompt | structured_llm_grader
    response = hallucination_grader.invoke({"documents": docs, "generation": generation})

    return {'messages': [response]}


def chatbot(question: str, retriver_vector_db):
    
    workflow = StateGraph(AgentState)

    workflow.add_node('agent', agent)

    retriver_tool = get_retrivel_tool(retriver_vector_db)

    retrieve = ToolNode(retriver_tool)

    workflow.add_node('retriver', retrieve)

    workflow.add_node('grade', grade_documents)

    workflow.add_node('rewrite', rewrite)

    workflow.add_node('generate', generate)

    workflow.add_node('hallucination', check_hallucination)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        'agent',
        tools_condition,
        {
            "tools": 'retriver',
            END:END
        }
    )

    workflow.add_conditional_edges(
        "retriver",
        grade_documents,

    )

    workflow.add_edge('generate', "hallucination")

    workflow.add_conditional_edges(
        'generate',
        {
            'hallucination': "hallucination",
            'rewrite': 'rewrite'
        }
    )
    workflow.add_edge('generate', END)

    graph  = workflow.compile()

    return graph