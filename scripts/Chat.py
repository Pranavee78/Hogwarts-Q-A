import requests
from datetime import datetime
import json
import os
import re

class ChatMemoryModel:
    def __init__(self, model_name="llama3:latest", api_url="http://localhost:11434/api/chat", file_name = None):
        """
        Initialize the ChatMemoryModel class.

        Args:
            model_name: The name of the model to use for generating responses.
            api_url: The API endpoint for Ollama (default is localhost).
        """
        self.model_name = model_name
        self.api_url = api_url
        self.history = []
        self.file_name = file_name

    def add_to_history_as_user(self, message):
        """
        Add a user message to the conversation history.

        Args:
            message: The user's message.
        """
        self.history.append({"role": "user", "content": message})

    def add_to_history_as_model(self, message):
        """
        Add a model's response to the conversation history.

        Args:
            message: The model's response.
        """
        self.history.append({"role": "assistant", "content": message})

    def get_history(self):
        """
        Retrieve the conversation history.

        Returns:
            A list of message dictionaries containing roles and content.
        """
        return self.history

    def send_message(self, user_message):
        """
        Send a user message to the model and receive a response.

        Args:
            user_message: The user's message to send to the model.

        Returns:
            The model's response.
        """
       
        
        self.add_to_history_as_user(user_message)
        
        payload = {
            "model": self.model_name,
            "messages": self.history, 
            "stream": False,
            "options":{
                        "num_ctx": 4096*2
                    }
        }

        try:
           
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # exception for HTTP errors

            # Check for a valid response
            response_data = response.json()
            print("Response from server:", response_data)

            
            model_response = response_data.get("message", {}).get("content", "") # get response

            if not model_response:
                return "Error: The response from the model was empty or invalid."

            
            self.add_to_history_as_model(model_response)
            self.save_history()

            return model_response

        except requests.exceptions.RequestException as e:
            return f"An error occurred while contacting the model: {e}"

    def clear_history(self):
        """
        Clear the conversation history. 
        """
        self.history = []
        
    def save_history(self):
        """
        Save the conversation history in a history folder as a JSON file.
        If no filename is provided, it will create a dynamic one like 'Untitled_1'.
        """
        # Ensure the 'history' directory exists
        if not os.path.exists('history'):
            os.makedirs('history')

        # If file_name is None, assign a dynamic name like 'Untitled_1'
        if self.file_name is None:
            self.file_name = self._generate_sequential_filename()

        # Create the full path for the file
        file_path = f"history/{self.file_name}.json"

        # Write the history to a JSON file
        with open(file_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved successfully as {file_path}")

    def _generate_sequential_filename(self, base_name="Untitled"):
        """
        Generate a sequential filename like 'Untitled_1', 'Untitled_2', etc.

        Args:
            base_name: The base name to use for the files (default is 'Untitled').

        Returns:
            The next available sequential filename.
        """
        # List existing files in the 'history' directory
        existing_files = os.listdir('history')
        pattern = re.compile(f"{base_name}_(\\d+)\\.json")

        # Extract numbers from filenames, e.g., 'Untitled_1.json' -> 1
        numbers = [int(pattern.search(f).group(1)) for f in existing_files if pattern.search(f)]

        # Find the next available number
        next_number = max(numbers, default=0) + 1

        return f"{base_name}_{next_number}"

            
    def jump_to_history(self, index):
        """
        Jump to a specific point in the conversation history by index.

        Args:
            index: The index number to jump back to.

        Returns:
            A message indicating whether the jump was successful.
        """
        if 0 <= index < len(self.history):
            self.history = self.history[:index + 1]  # Keep history up to the selected index
            self.save_history()
            return f"Jumped back to message {index}."
        else:
            return f"Error: Index {index} is out of range."
        
    def load_history(self, path):
        """
        Load conversation history from a specified file path.

        Args:
            path: The file path to load the history from.

        Returns:
            A message indicating whether the load was successful or if an error occurred.
        """
        if not os.path.exists(path):
            return f"Error: The file {path} does not exist."

        try:
            with open(path, 'r') as f:
                self.history = json.load(f)
            return "History loaded successfully."
        except Exception as e:
            return f"An error occurred while loading the history: {e}"

        
if __name__ == "__main__":
    
    chat_model = ChatMemoryModel(model_name="llama3:latest")

    print("Start chatting with the model! Type /bye to exit.")
    
    while True:
       
        user_input = input("You: ")
        
        
        if user_input.strip().lower() == "/bye":
            print("Goodbye!")
            break
        elif user_input.strip().lower() == "/history":
            print("History:", chat_model.get_history())
            continue
        elif user_input.strip().lower() == "/clear":
            chat_model.clear_history()
            print("clearing history......")
        elif user_input.strip().lower() == "/jump":
            try:
                index = int(input("Enter a valid index"))
                print(chat_model.jump_to_history(index))
            except ValueError:
                print("Invalid index. Usage: /jump [index]")
        elif user_input.strip().lower() == "/load":
            file_path = input("Enter the file path to load history: ")
            print(chat_model.load_history(file_path))
        else:
            
            response = chat_model.send_message(user_input)
            print("Model:", response)
            

    
    chat_model.clear_history()