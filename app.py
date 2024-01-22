import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from LLAMA2 model
def getLLAMAResponse(input_text, no_words, blog_style):

    ### LLAMA2 model
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## Prompt Template

    template = """
    Write about {input_text} such that it is informative to {blog_style} around {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)

    ## Generating the response from LLAMA 2 model

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)

    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon = "<3",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Generate Blogs")

input_text = st.text_input("Enter the Blog topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox("Writing blog for:", ("Researchers", "Data Scientist", "Common People", "Professionals"), index=0)

submit = st.button("Generate")


## Final Response
if submit:
    st.write(getLLAMAResponse(input_text, no_words, blog_style))