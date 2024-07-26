import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from main import ChatBot

# Set page configuration at the top of the script
st.set_page_config(page_title="Meeting Information Bot")

# Load environment variables
load_dotenv()

bot = None

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            passwd=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
    except Error as e:
        st.error(f"Error: '{e}'")
    return connection

def query_database(query, params=None):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        result = cursor.fetchall()
        return pd.DataFrame(result)
    except Error as e:
        st.error(f"Error: '{e}'")
        return None
    finally:
        cursor.close()
        connection.close()

def get_user_role(username, password):
    query = "SELECT role FROM Users WHERE UserID = %s AND PW = %s"
    params = (username, password)
    df = query_database(query, params)
    if not df.empty:
        return df.iloc[0]['role']
    else:
        return None

def get_access_level(role):
    query = "SELECT access_levels FROM Roles WHERE role = %s"
    params = (role,)
    df = query_database(query, params)
    if not df.empty:
        return df.iloc[0]['access_levels']
    else:
        return None

def main():
    global bot
    st.title("Meeting Information Bot")
    if bot is None:
        bot = ChatBot(llm_type='Local (PHI3)', api_key='')

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.access_levels = ""

    if not st.session_state.logged_in:
        # Login Page
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            role = get_user_role(username, password)
            if role:
                access_string = get_access_level(role)
                access_levels = [level.strip() for level in access_string.split(",")]
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.access_levels = access_levels
                st.success(f"Welcome {username}! Your role is {role} with access levels: {', '.join(access_levels)}")
                run_app(access_levels)
            else:
                st.error("Invalid username or password")
    else:
        run_app(st.session_state.access_levels)

def run_app(access_levels):
    global bot
    with st.sidebar:
        st.title('Meeting Information Bot')

        llm_selection = st.selectbox(
            "Select LLM",
            ["Local (PHI3)", "External (OpenAI)"],
            key="llm_selection"
        )

        if llm_selection == "External (OpenAI)":
            st.session_state.api_key = st.text_input("Enter OpenAI API Key", type="password")
        else:
            st.session_state.api_key = ""
            
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    if st.session_state.llm_selection == "External (OpenAI)" and not st.session_state.api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
    else:
        try:
            bot = ChatBot(llm_type=st.session_state.llm_selection, api_key=st.session_state.api_key)
        except Exception as e:
            st.error(f"Failed to initialize the chatbot: {e}")
            st.stop()

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Welcome, what can I help you with?"}]
            st.session_state.context_history = []

        def generate_response(input_dict):
            try:
                result = bot.rag_chain.invoke(input_dict)
                return result
            except Exception as e:
                st.error(f"Error generating response: {e}")
                return "An error occurred while generating the response."

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if input := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": input})
            with st.chat_message("user"):
                st.write(input)

            try:
                context = bot.get_context_from_collection(input, access_levels=access_levels)
                st.session_state.context_history.append(context)
            except Exception as e:
                st.error(f"Error retrieving context: {e}")
                context = "An error occurred while retrieving context."

            input_dict = {"context": context, "question": input}
            with st.chat_message("assistant"):
                with st.spinner("Grabbing your answer from database..."):
                    response = generate_response(input_dict)
                    st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == '__main__':
    main()
