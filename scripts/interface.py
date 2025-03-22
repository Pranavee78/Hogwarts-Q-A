from query import query_2 as query
from Chat import ChatMemoryModel
from store import load_vectorstore




def send_prompt(question,vectorstore, top_n = 10, chat_model = ChatMemoryModel(model_name="llama3.1"), character="wizard"):
    context = query(question, vectorstore, top_n=top_n, send_prompt=False)
    prompt = f"As a {character} in the magical world of Harry Potter, respond to the following question only using the references provided. Make sure to include specific quotes and references from the context to explain your answer in a style that J.K. Rowling might use.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
    print(prompt)
    response = chat_model.send_message(user_message=prompt)
    return response


def generate_history(chat_model:ChatMemoryModel):
    return chat_model.get_history()

def load_point(chat_model:ChatMemoryModel, point):
    return chat_model.jump_to_history(point)

def load_history(chat_model:ChatMemoryModel, file_path):
    return (chat_model.load_history(file_path))


def clear_history(chat_model:ChatMemoryModel):
    return chat_model.clear_history()


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    send_prompt("What is the relationship between Harry Potter and Sirius Black", vectorstore)