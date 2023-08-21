import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# App title
st.set_page_config(page_title="💬 afnan-chatbot")

# Function for generating response using T5 model
def generate_response(prompt_input):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    
    input_ids = tokenizer.encode(prompt_input, return_tensors="pt")
    outputs = model.generate(input_ids)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": f"chatbot: {prompt}"})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
