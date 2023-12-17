__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os
import json
import logging
import time
import queue
import threading
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.vectorstores import Chroma
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain.embeddings import OctoAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from dotenv import load_dotenv

os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

load_dotenv()

endpoint_url = os.getenv("ENDPOINT_URL")

# Constants
OCTOAI_TOKEN = os.environ.get("OCTOAI_API_TOKEN")
OCTOAI_JSON_FILE_PATH = "data/octoai_docs_urls.json"
K8_JSON_FILE_PATH = "data/k8_docs_urls_setup.json"
K8_DB_NAME = "chroma_k8_docs"
OCTOAI_DB_NAME = "chroma_octoai_docs"

if OCTOAI_TOKEN is None:
    raise ValueError("Environment variables not set.")

logging.basicConfig(level=logging.CRITICAL)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_urls(file_path):
    with open(file_path, "r") as file:
        return [item["url"] for item in json.load(file)]


def scrape_with_playwright(urls):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    return BeautifulSoupTransformer().transform_documents(docs, tags_to_extract=["div"])


def extract(content):
    return {"page_content": str(BeautifulSoup(content, "html.parser").contents)}


def tokenize(text):
    return text.split()


def find_common_phrases(contents, phrase_length=30):
    reference_content = contents[0]["page_content"]
    tokens = tokenize(reference_content)
    return {
        " ".join(tokens[i : i + phrase_length])
        for i in range(len(tokens) - phrase_length + 1)
        if all(
            " ".join(tokens[i : i + phrase_length]) in content["page_content"]
            for content in contents
        )
    }


def remove_common_phrases_from_contents(contents, common_phrases):
    for content in contents:
        for phrase in common_phrases:
            content["page_content"] = content["page_content"].replace(phrase, "")
    return contents


def process_documents(urls):
    docs_transformed = scrape_with_playwright(urls)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1300, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)
    return [extract(split.page_content) for split in splits]


def get_vector_store(db_name=OCTOAI_DB_NAME):
    return Chroma(
        embedding_function=OctoAIEmbeddings(
            endpoint_url="https://instructor-large-f1kzsig6xes9.octoai.run/predict"
        ),
        persist_directory=f"./{db_name}",
        collection_name=db_name,
    )


def get_language_models():
    return (
        OctoAIEndpoint(
            octoai_api_token=OCTOAI_TOKEN,
            endpoint_url="https://text.octoai.run/v1/chat/completions",
            model_kwargs={
                "messages": [
                    {
                        "role": "system",
                        "content": "Write a response that appropriately completes the request. Be clear and concise. Format your response as bullet points whenever possible.",
                    }
                ],
                "model": "llama-2-13b-chat-fp16",
                "presence_penalty": 0,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 400,
                "seed": 123,
            },
        ),
        OctoAIEndpoint(
            octoai_api_token=OCTOAI_TOKEN,
            endpoint_url="https://text.octoai.run/v1/chat/completions",
            model_kwargs={
                "model": "mistral-7b-instruct-fp16",
                "messages": [
                    {
                        "role": "system",
                        "content": "Write a response that appropriately completes the request. Be clear and concise. Format your response as bullet points whenever possible.",
                    }
                ],
                "presence_penalty": 0,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 400,
                "seed": 123,
            },
        ),
    )


def add_documents_to_vectorstore(extracted_contents, vectorstore):
    for item in extracted_contents:
        doc = Document.parse_obj(item)
        doc.page_content = str(item["page_content"])
        vectorstore.add_documents([doc])


def execute_and_print(llm, retriever, question, model_name):
    start_time = time.time()
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, max_tokens_limit=2000)
    response = qa({"question": question, "chat_history": []})
    end_time = time.time()
    result = f"\n{model_name}\n"
    result += response["answer"]
    result += f"\n\nResponse ({round(end_time - start_time, 1)} sec)"
    return result


def predict(data_source="octoai_docs", prompt="how to avoid cold starts?"):
    schema = {
        "properties": {"page_content": {"type": "string"}},
        "required": ["page_content"],
    }
    db_name = (
        K8_DB_NAME
        if ("k8" in data_source or "kubernetes" in data_source)
        else OCTOAI_DB_NAME
    )
    vectorstore = get_vector_store(db_name)
    llm_llama2_13b, llm_mistral_7b = get_language_models()

    if vectorstore._collection.count() < 32:
        url_file = K8_JSON_FILE_PATH if db_name == K8_DB_NAME else OCTOAI_JSON_FILE_PATH
        urls = load_urls(url_file)

        extracted_contents = process_documents(urls)
        common_phrases = find_common_phrases(extracted_contents)
        extracted_contents_modified = remove_common_phrases_from_contents(
            extracted_contents, common_phrases
        )
        add_documents_to_vectorstore(extracted_contents_modified, vectorstore)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    results = []
    results.append(execute_and_print(llm_llama2_13b, retriever, prompt, "LLAMA2-13B"))
    results.append(execute_and_print(llm_mistral_7b, retriever, prompt, "MISTRAL-7B"))

    # Return the combined results
    return "\n".join(results)


def handler(event, context):
    data_source = event.get("data_source", "octoai_docs")
    prompt = event.get("prompt", "How to reduce cold starts?")
    answer = predict(data_source, prompt)
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": answer,
            }
        ),
    }
