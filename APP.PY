# Import necessary libraries
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the conversational model and tokenizer, and move the model to the GPU if available
model_name = "microsoft/DialoGPT-small"  # You can try larger models like DialoGPT-medium if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Initialize conversation history in Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []

# Define a function to generate AI-related responses
def ai_chatbot_conversation(input_text):
    # Tokenize the input and previous conversation history
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
    bot_input_ids = torch.cat([torch.LongTensor(st.session_state.history).to(device), new_user_input_ids], dim=-1) if st.session_state.history else new_user_input_ids

    # Generate a response from the model with adjusted settings
    output = model.generate(
        bot_input_ids,
        max_length=200,          # Increase max_length if responses are too short
        temperature=0.7,         # Add slight randomness for varied responses
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update the conversation history
    st.session_state.history.append(new_user_input_ids)
    st.session_state.history.append(output[:, bot_input_ids.shape[-1]:])

    return response

# Streamlit app layout
st.title("AI Chatbot")
st.write("Hello! I can provide information about Artificial Intelligence. Type your message below and hit Enter.")

# Input box for user input
user_input = st.text_input("You: ", key="input")

# Display bot response if there is user input
if user_input:
    response = ai_chatbot_conversation(user_input)
    st.write("AI Chatbot:", response)

# Button to clear conversation history
if st.button("Clear Chat"):
    st.session_state.history = []
    st.write("Chat history cleared.")
