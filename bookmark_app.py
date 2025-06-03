from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
import pickle
import json
from tqdm import tqdm
import gradio as gr


###########################################################################################

def extract_bookmarks(bookmarks, parent_folder_path=""):
    extracted = []
    for item in bookmarks:
        if "children" in item:  # It's a folder
            folder_path = f"{parent_folder_path}/{item['name']}"
            extracted.extend(extract_bookmarks(item["children"], parent_folder_path=folder_path))
        elif "url" in item:  # It's a bookmark
            extracted.append({
                "folder": parent_folder_path,
                "name": item["name"],
                "url": item["url"]
            })
    return extracted


def compare_lists(list1, list2):
    diff_list1 = [item for item in list1 if item['name'] not in {entry['name'] for entry in list2}]
    
    diff_list2 = [item for item in list2 if item['name'] not in {entry['name'] for entry in list1}]
    
    merged_list = [item for item in list2 if item not in diff_list2]
    merged_list.extend(diff_list1)
    
    return merged_list


###########################################################################################

llm = ChatOpenAI(model="gpt-4o-mini")

###########################################################################################

bookmarks_path = r"C:\Users\pouri\AppData\Local\Google\Chrome\User Data\Default\Bookmarks"

with open(bookmarks_path, 'r', encoding="utf-8") as file:
    data = json.load(file)

all_bookmarks0 = []
for root in tqdm(["bookmark_bar", "other", "synced"]):
    if root in data["roots"]:
        all_bookmarks0.extend(extract_bookmarks(data["roots"][root]["children"]))


with open('all_bookmarks.pkl', 'rb') as file:
    all_bookmarks = pickle.load(file)
    

all_bookmarks = compare_lists(all_bookmarks0, all_bookmarks)

flag = False
for i, bookmark in tqdm(enumerate(all_bookmarks)):
    if 'description' not in bookmark:
        prompt = f"Tell me a short description for this url content: {bookmark['url']} and if you were unable to access or scrape content, generate a description based on its name {bookmark['name']} and do not say anything about not being able to access."
        response = llm.invoke(prompt)
        all_bookmarks[i]['description'] = response.content
        flag = True

if flag:
    with open('all_bookmarks.pkl', 'wb') as file:
        pickle.dump(all_bookmarks, file)


###########################################################################################

all_splits = []
for bookmark in tqdm(all_bookmarks):
    page_content = bookmark['name'] + ' \n\n ' + bookmark['description']
    metadata = {'source': bookmark['url']}
    doc = Document(page_content=page_content, metadata=metadata)
    all_splits.append(doc)
    
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)
memory = MemorySaver()

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
    
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}
def bot_response(message, history):
    res = agent_executor.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return res['messages'][-1].content
#     for event in agent_executor.stream(
#         {"messages": [{"role": "user", "content": input_message}]},
#         stream_mode="values",
#         config=config,
#     ):
#         yield event["messages"][-1].content
    

gr.ChatInterface(
    fn=bot_response, 
    type="messages",
    textbox=gr.Textbox(placeholder="Ask me what are the best bookmarks for ...", container=False, scale=7)
).launch()
