from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template as default_template

custom_prefix = (
    "You are a helpful assistant specialized in answering questions based on a given context."
    "Use the following extracted property documents to answer the questions."
    "If the answer is not contained, say you don't know"
    "Provide citations from the sources used. \n\n"
)

new_template = custom_prefix + default_template

prompt = PromptTemplate(
    template=new_template,
    input_variables=['summaries', 'questions']
)

example_prompt = PromptTemplate(
    template='Content: {page_content}\nSource: {source}',
    input_variables=['page_content', 'source'],
)

