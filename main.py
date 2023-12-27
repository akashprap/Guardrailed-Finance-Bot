import gradio as gr
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer,util
import numpy as np
import faiss
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import dotenv
import os
from langchain import OpenAI
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
import spacy
from collections import Counter

dotenv.load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
nlp = spacy.load("en_core_web_sm")

def guardrail(docc):
    # Using Genism Library
    text_genism="".join(docc)
    processed_text = preprocess_string(text_genism)
    dictionary = corpora.Dictionary([processed_text])
    corpus = [dictionary.doc2bow(processed_text)]
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary)
    topics = lda_model.print_topics()
    topic_only=[]
    for i in topics:  
        for j in i[1].split("+"):
            topic_only.append((j.split("*")[1][1:-2]).strip())
    
    #Using Spacy
    text_spacy=" ".join(topic_only)
    complete_doc = nlp(text_spacy)
    words = [
        token.text
        for token in complete_doc
        if not token.is_stop and not token.is_punct
    ]
    output_topic=(Counter(words).most_common(10))

    top_topic=output_topic[0][0]
    # If data is finance related or not

    finance_keywords = [
        'trade', 'market', 'invest', 'stock', 'credit', 'fund', 'bank', 'asset',
        'portfolio', 'equity', 'bond', 'insurance', 'currency', 'dividend', 'loan',
        'financial', 'economy', 'interest', 'capital', 'wealth','brokerage', 'exchange',
        'commodities', 'derivative', 'securities', 'liabilities','transaction', 'valuation',
        'revenue', 'profit', 'loss', 'liquidity', 'mortgage','savings', 'retirement', 'audit',
        'inflation', 'deflation', 'risk', 'hedge'
    ]

    word_frequencies = output_topic

    finance_related_count = sum(freq for word, freq in word_frequencies if word in finance_keywords)

    threshold = 10

    is_finance_related = (finance_related_count >= threshold)

    if is_finance_related:
        return top_topic
    else:
        return False

def semantic_search(Question, *urls):
    urls = [url for url in urls if url]
    loader = UnstructuredURLLoader(urls=list(urls))
    data=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators='\n',chunk_size=600,chunk_overlap=100)
    docs=text_splitter.split_documents(data)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    docc=[]
    metadata=[]
    for i in range(len(docs)):
        docc.append(docs[i].page_content)
        metadata.append(docs[i].metadata)
    embed=model.encode(docc)
    # Call preprocessing
    is_valid=guardrail(docc)
    if not is_valid:
        response="Data not related to finance"
        source=[["After Preprocessing"]]
        return response, source
    else:
        top_topic=is_valid

        dim=embed.shape[1]
        index=faiss.IndexFlatL2(dim)
        index.add(embed)
        index_filename = 'index_flat_l2.index'
        faiss.write_index(index, index_filename)
        index_filename = 'index_flat_l2.index'
        loaded_index = faiss.read_index(index_filename)
        search_index=model.encode(Question)
        search_index=search_index.reshape(1,-1)
        _,I=loaded_index.search(search_index,k=5)
        result_context=""
        source=[]
        for _,ind in enumerate(I[0]):
            result_context+=docc[ind]
            if [metadata[ind]['source']] not in  source:
                source.append([metadata[ind]['source']])
        llm=OpenAI(temperature=0.7,)
        
        prompt_template = """ Answer in very detail and try to remain close to the topic and Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Top Topic: {top_topic}
        Question: {question}
        Answer:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context","top_topic","question"])

        query_llm=LLMChain(llm=llm,prompt=prompt)
        response=query_llm.run({"context":result_context,"top_topic":top_topic,"question":Question})
        return response, source

url_inputs = [gr.Textbox(lines=2, label=f"URL {i+1}") for i in range(2)]
iface = gr.Interface(fn=semantic_search, inputs=["text"]+url_inputs, outputs=["text", "list"])
iface.launch()
