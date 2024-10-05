import streamlit as st

from dotenv import load_dotenv
from src.graph_function import setup_react_graph

load_dotenv()


def main():
    st.title("Hello ReAct with LangGraph")

    # get user input
    user_input = st.text_input(
        "Enter your query:",
        value="What is the weather in San Francisco? Write it and then Triple it."
    )

    if st.button("Submit"):
        with (st.spinner("Processing...")):
            flow = setup_react_graph()
            app = flow.compile()
            res = app.invoke(
                input={
                    "input": user_input
                }
            )
            output = res["agent_outcome"].return_values["output"]
            st.write(output)


if __name__ == "__main__":
    main()
