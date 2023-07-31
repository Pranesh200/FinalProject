import os
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
import streamlit as st
import json
import openai

def display_responses():
    if len(st.session_state.selected_answers) == 3:
        st.write("Feedback:")
        question_obj = st.session_state.question_data
        for i in range(len(question_obj["questions"])):
            question_obj["questions"][i] = question_obj["questions"][i].replace("'", "\"")
            question_obj["questions"][i] = json.loads(question_obj["questions"][i])
        mcq_questions = [q["Question"] for q in question_obj["questions"]]
        selected_answers = [st.session_state.selected_answers.get(f"Question {i+1}", "") for i in range(3)]

        correct_answers = [q["Answer"] for q in question_obj["questions"]]

        prompt = "Given the following multiple-choice questions and answers:\n\n"
        for i in range(3):
            prompt += f"{i + 1}. {mcq_questions[i]}\nSelected answer: {selected_answers[i]}\nCorrect answer: {correct_answers[i]}\n\n"

        prompt += "Please act like an intelligent tutor and suggest topics the student should review and provide some learning resources or exercises based on the questions.\n"

        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=400
        )

        st.write("Intelligent Tutor Suggestions:")
        st.write(response["choices"][0]["text"])

def main():
    st.set_page_config(page_title="Upload your PDF")
    openai_api_key = st.sidebar.text_input('OpenAI API Key')
    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.header("Upload your PDF (after inputting OpenAI Key to the right)")

    uploadFile = st.file_uploader("Upload your PDF", type="pdf")
    if "question_generated" not in st.session_state:
        st.session_state.question_generated = False

    ques_response = "" 

    st.session_state.setdefault("sum_response", "")

    if uploadFile is not None:
        processedPDF = PdfReader(uploadFile)
        pdfText = ""
        for page in processedPDF.pages:
            pdfText += page.extract_text()

        textChunks = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        embedChunks = textChunks.split_text(pdfText)

        embeddings = OpenAIEmbeddings()
        simGraph = FAISS.from_texts(embedChunks, embeddings)

        summarizePrompt = "Summarize this paper in one paragraph"
        questionPrompt1 = "Give me one multiple choice question based on a random section of the paper with an answer key. Generate a response in key-value pair format with the following structure: {'Question': {}, 'Options': [], Answer: {}}"
        suggestPrompt = "Generate one multiple choice question and Suggest additional learning material or exercises based on the question"

        st.session_state.setdefault("question_data", None)
        st.session_state.setdefault("selected_answers", {})

        if not st.session_state.question_generated:
            docs = simGraph.similarity_search(summarizePrompt)
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
            with get_openai_callback() as apiResponse:
                st.session_state.sum_response = chain.run(input_documents=docs, question=summarizePrompt)
                st.session_state.question_data = {
                    "questions": [chain.run(input_documents=docs, question=questionPrompt1) for _ in range(3)],
                    "suggest_response": chain.run(input_documents=docs, question=suggestPrompt)
                }
            st.session_state.question_generated = True

        if st.session_state.question_data:
            st.write("Summary:")
            st.write(st.session_state.sum_response)
            st.write("---")

            for idx, question in enumerate(st.session_state.question_data["questions"]):
                input_text = question.replace("'", "\"")
                output_data = json.loads(input_text)
                with st.form(f"my_form_{idx}"):
                    st.write(output_data["Question"])
                    checkbox_val1 = st.checkbox(output_data["Options"][0], key=f"checkbox1_{idx}")
                    checkbox_val2 = st.checkbox(output_data["Options"][1], key=f"checkbox2_{idx}")
                    checkbox_val3 = st.checkbox(output_data["Options"][2], key=f"checkbox3_{idx}")
                    checkbox_val4 = st.checkbox(output_data["Options"][3], key=f"checkbox4_{idx}")
                    submitted = st.form_submit_button("Submit")

                    if submitted:
                        selected_answer = None
                        if checkbox_val1:
                            selected_answer = output_data["Options"][0]
                        elif checkbox_val2:
                            selected_answer = output_data["Options"][1]
                        elif checkbox_val3:
                            selected_answer = output_data["Options"][2]
                        elif checkbox_val4:
                            selected_answer = output_data["Options"][3]

                        st.session_state.selected_answers[f"Question {idx+1}"] = selected_answer

                        correct_answer = output_data["Answer"]
                        if selected_answer == correct_answer:
                            st.success("You got the answer right!")
                        else:
                            st.error(f"Oops! That's incorrect.")
        if len(st.session_state.selected_answers) == 3:
            display_responses()

if __name__ == '__main__':
    main()