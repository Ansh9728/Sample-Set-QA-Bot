import os
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import Field, BaseModel
from pprint import pprint
from langchain import hub
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import Annotated, Sequence
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


memory = MemorySaver()


def get_llm_model(model="llama3-70b-8192"):
    model = ChatGroq(
        temperature=0,
        model=model,
        api_key = os.getenv('GROQ_API_KEY'),
        streaming=True
    )

    return model


# Graph State
class GraphState(TypedDict):
    """
    Represent the state of Our Graph

    Attributes:

        question: question
        generation: LLM Generation
        documents: list of documents
        retriver: object to retrive data from vectordatabase

    """

    question: str
    generation: str
    documents: List[str]
    retriever: object
    intent: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


"""##################### Define Graph Nodes"""

# Retrieve Node
def retrieve(state):

    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")

    question = state['question']
    retriever = state['retriever']

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}



# grade Node

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENTS RELEVENT TO THE QUESTION")

    question  = state['question']
    documents = state['documents']


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

    retrieval_grader= prompt | llm_with_tool

    # Store Filtered doc
    filterd_docs = []

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "context": d.page_content}
        )

        grade = score.binary_score

        if grade=='yes':
            print("---GRADE: DOCUMENT RELEVENT")
            filterd_docs.append(d)

        else:
            print("---GRADE: DOCUMENT NOT RELEVENT")
            continue

    return {"documents": filterd_docs, 'question': question}



# Generate Node
def generate(state):
    """
    Generate Answer

    Args:
        state(dict): The current graph state

    Return:
        state(dict): New  key added to state, generation, that contains LLM Answers

    """

    print('---GENERATE---')

    question = state['question']

    documents = state['documents']

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    llm = get_llm_model()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}


# Rewrite Question Node

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]


    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    llm = get_llm_model()

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# Out of COntext Node
def out_of_context(state):
    """
    Determine the user intent. If it is a greeting, reply; otherwise, return "I don't know" and end the graph.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for the next node to call
    """
    print("---OUT OF THE CONTEXT")
    question = state['question']
    documents = state['documents']

    class Intent(BaseModel):
        """Determine the user intent."""
        intent: str = Field(
            description="Answer is grounded in the facts of greeting: 'yes' or 'no'"
        )

    # Prompt for intent classification
    system_message = """You are a question intent classifier. If the question is related to a greeting, return 'yes' and provide a greeting message; otherwise, return 'no'."""

    intent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "human",
                "Here is the initial question: \n\n {question} \n and Documents {documents}",
            ),
        ]
    )

    llm = get_llm_model()
    structured_intent_model = llm.with_structured_output(Intent)

    # Invoke the intent model
    question_intent = intent_prompt | structured_intent_model
    intent_score = question_intent.invoke({"documents": documents, "question": question})

    intent_result = intent_score.intent
    print('intetn_result', intent_result)
    if intent_result == 'yes':
        # Generate and return a greeting message
        return {"intent": generate_greeting_message(), "question": question, 'documents':documents}
    else:
        return {"intent": "Out OF the Context", "question": question, 'documents':documents}

def generate_greeting_message():
    """Generates a greeting message."""
    return "Hello! How can I assist you today?"



"""###################### Define Edges ####################"""

# Edges

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]



    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
        )


    # LLM with function call
    llm = get_llm_model("llama3-70b-8192")
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

    hallucination_grader = hallucination_prompt | structured_llm_grader

    score = hallucination_grader.invoke(
       {"documents": documents, "generation": generation}
    )

    grade = score.binary_score


    # Answer Grader

        # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )


    # LLM with function call
    llm = get_llm_model()
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader


     # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY ONLY 3 times---")
        return "not supported"


def decide_to_generate(state) -> Literal['generate', 'transform_query']:
    """
    Determine wheather to generate an answer, or regenerate a question

    Args:
        state(dict): The current graph state

    Returns:
        str: Binary Decision for next node to call

    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


"""################# Build Graph ################"""

# Build Graph
workflow = StateGraph(GraphState)


# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("out_of_context", out_of_context)


workflow.add_edge(START, 'retrieve')
workflow.add_edge('retrieve', 'grade_documents')
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "out_of_context",
        "useful": END,
        "not useful": "transform_query",
    },
)

workflow.add_edge('out_of_context', END)

# Compile
app = workflow.compile()



def chatbot_response_generation(question, retriever):

    inputs = {"question": question, 'retriever': retriever}

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    if 'generation' in value:  # Ensure 'generation' exists
        pprint(value["generation"])
        return value["generation"]
    else:
        pprint("No generation found.")
        return "question is out of the context"



# def chatbot_response_generation(retriever):

#     while True:
#       question = input('Enter your question (type "exit" to quit): ')
#       if question == 'exit':
#           break
#       inputs = {"question": question, 'retriever': retriever}

#       for output in app.stream(inputs):
#           for key, value in output.items():
#               # Node
#               pprint(f"Node '{key}':")
#               # Optional: print full state at each node
#               # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#           pprint("\n---\n")

#       if 'generation' in value:  # Ensure 'generation' exists
#           pprint(value["generation"])
#           return value["generation"]
#       else:
#           pprint("No generation found.")
#           return "question is out of the context"