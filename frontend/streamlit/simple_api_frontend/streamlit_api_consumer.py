import streamlit as st
import requests

# Set the base URL of  API
base_url = ""

# Define the available endpoints and their corresponding fields
endpoints = {
    "/1": ["input_1", "input_2", "input_3"],
    "/2": ["input_1", "input_2", "input_3"]
}

# Streamlit app
def main():
    st.title("Title goes here")

    # Endpoint selection
    selected_endpoint = st.sidebar.selectbox("Select Endpoint", list(endpoints.keys()))

    # API key input (faux-password)
    api_key = st.sidebar.text_input("API Key")

    # Display fields based on the selected endpoint
    fields = endpoints[selected_endpoint]
    input_data = {}

    # Fields below - configure based on examples
    for field in fields:
        if field == "input_1":
            # Special handling for input_1 field which uses a long text box
            st.subheader("input_1")
            input_data[field] = st.text_area("Enter input_1")
        elif field == "input_2":
            # Special handling for input_2 field which is a number input using a selector
            st.subheader("input_2")
            input_data[field] = st.number_input("input_2 value", min_value=0, max_value=100, value=80, step=1)
        else:
            # Regular input fields
            input_data[field] = st.text_input(field)

    # Make API request when the user clicks the "Submit" button
    if st.button("Submit"):
        # Prepare the request data
        request_data = {key: value for key, value in input_data.items() if value}

        # Add the API key to the request headers
        headers = {"X-Api-Key": api_key}

        # Make the API request
        response = requests.post(base_url + selected_endpoint, json=request_data, headers=headers)

        # Display the response
        st.subheader("API Response")
        st.json(response.json())

if __name__ == "__main__":
    main()