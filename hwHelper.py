import streamlit as st
import boto3 
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import StreamlitCallbackHandler

## function to extract text via aws textract
def extract_text(file):
    client = boto3.client('textract')
    response = client.detect_document_text(
        Document={'Bytes': file}
    )
    text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + '\n'
    return text

## LLM Functions

bedrockClient = boto3.client('bedrock-runtime')

llm = Bedrock(
    model_id="anthropic.claude-v2", 
    client=bedrockClient,
    streaming=True,
    model_kwargs={'temperature': .99, 'max_tokens_to_sample': 2000}
)

summaryPrompt = PromptTemplate(
    template="""
    
    Human:
    Summarize the below text: \n"{inputText}\"
    Assistant:""",
    input_variables=["inputText"]
)

questionsPrompt = PromptTemplate(
    template="""
    
    Human:
    Create 5 multiple choice questions from the input text. 
    Use the below output

    <br>
    <b>Question:</b> Question1 <br>
    a) answer1 <br> b) answer2 <br> c) answer3 <br> d) answer4 <br>
    <details>
    <summary><b>Answer</b></summary>
    Answer to the question
    </details>


    <inputText> 
    {inputText}
    </inputText>
    Assistant:""",
    input_variables=["inputText"]
)

if 'rawText' not in st.session_state:
    st.session_state['rawText'] = ''

## Streamlit UI 
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    extracted_text = extract_text(bytes_data)
    st.session_state['rawText'] = extracted_text

tab1, tab2, tab3 = st.tabs(["Text", "Summary", "Questions"])

with tab1:
    st.header("Raw Text from file")
    st.write(st.session_state['rawText'])

with tab2:
    st.header("Summary")
    st_callback = StreamlitCallbackHandler(st.container())
    summaryChain = LLMChain(llm=llm,prompt=summaryPrompt,callbacks=[st_callback])
    st.write(summaryChain.run(inputText = st.session_state['rawText']))

with tab3:
    st.header("Questions")
    st_callback = StreamlitCallbackHandler(st.container())
    questionChain = LLMChain(llm=llm,prompt=questionsPrompt,callbacks=[st_callback])
    st.session_state['questions'] = questionChain.run(inputText = st.session_state['rawText'])
    st.markdown(st.session_state['questions'], unsafe_allow_html=True)
