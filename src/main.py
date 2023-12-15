from dotenv import load_dotenv
import os

from llama_index.llms import AzureOpenAI
from llama_index.llms import ChatMessage
from llama_index.embeddings import AzureOpenAIEmbedding

from llama_index import (
    VectorStoreIndex,
    StorageContext,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)

# Example that reads the pages with the `page_ids`
from llama_hub.confluence import ConfluenceReader

from llama_index.vector_stores import CognitiveSearchVectorStore
from llama_index.vector_stores.cogsearch import (
    IndexManagement,
    CognitiveSearchVectorStore,
)

# set up Azure Cognitive Search
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


def run():
    llm = AzureOpenAI(
        engine=AZURE_OPENAI_CHAT_DEPLOYEMENT, model=AZURE_OPENAI_CHAT_MODEL, temperature=0.7, top_p=0.95
    )
    
    llm_predictor = LLMPredictor(llm=llm)

    embed_model = AzureOpenAIEmbedding(
        model=AZURE_OPENAI_EMBEDDING_MODEL,
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYEMENT,
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION,
    )

    # max LLM token input size
    max_input_size = 128000
    # set number of output tokens
    num_output = 2048
    # set maximum chunk overlap
    max_chunk_overlap = 0.25

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embed_model,
        prompt_helper=prompt_helper
    )
    
    # set_global_service_context(service_context)

    cognitive_search_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_APIKEY)

    service_endpoint = AZURE_COGNITIVE_SEARCH_ENDPOINT

    # Use index client to demonstrate creating an index
    index_client = SearchIndexClient(
        endpoint=service_endpoint,
        credential=cognitive_search_credential,
    )

    vector_store = CognitiveSearchVectorStore(
        search_or_index_client=index_client,
        index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="id",
        chunk_field_key="content",
        embedding_field_key="embedding",
        metadata_string_field_key="li_jsonMetadata",
        doc_id_field_key="li_doc_id",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        [],
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

    confluence_spaces = CONFLUENCE_SPACE_LIST.split(",")

    for space_key in confluence_spaces:
        reader = ConfluenceReader(base_url=CONFLUENCE_BASE_URL)
        documents = reader.load_data(space_key=space_key, include_attachments=False, page_status="current")
        documents.extend(reader.load_data(page_ids=[], include_children=False, include_attachments=False))
        for doc in documents:
            index.insert(doc)


if __name__ == "__main__":

    load_dotenv()

    CONFLUENCE_PASSWORD  = os.getenv("CONFLUENCE_PASSWORD")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
    CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
    CONFLUENCE_SPACE_LIST = os.getenv("CONFLUENCE_SPACE_LIST")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
    AZURE_OPENAI_CHAT_DEPLOYEMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYEMENT")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    AZURE_OPENAI_EMBEDDING_DEPLOYEMENT = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYEMENT")
    
    AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT")
    AZURE_COGNITIVE_SEARCH_APIKEY = os.getenv("AZURE_COGNITIVE_SEARCH_APIKEY")
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")

    run()
