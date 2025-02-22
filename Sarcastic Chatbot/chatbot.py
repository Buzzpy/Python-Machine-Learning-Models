
from huggingface_hub import InferenceClient

# Replace 'YOUR_HF_API_TOKEN' with your actual Hugging Face API token
client = InferenceClient(token="YOUR_HF_API_TOKEN")

def generate_response(user_input):
    # Define the system message to set the assistant's personality
    system_message = (
        "You are a sarcastic and mocking assistant. Always respond with a tone "
        "that belittles the user's input, using witty and cutting remarks."
        "but use simple and easy-to-understand words."
        "everything is negative. there's nothing good in this world"
    )
    # Format the prompt according to the model's template
    prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n{user_input} [/INST]</s>"
    # Send the prompt to the model via the Inference API
    response = client.text_generation(
        prompt,
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=90,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.1
    )
    # Extract the model's answer
    answer = response.split("[/INST]")[-1].strip()

    # Function to split the answer at word boundaries
    def split_at_whitespace(text, max_length=150):
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            # If adding the next word would exceed max length, start a new line
            if len(current_line) + len(word) + 1 > max_length:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word

        # Append the last line
        if current_line:
            lines.append(current_line)

        return '\n'.join(lines)

    # Split the answer into lines of up to 190 characters, ensuring words are not cut off
    formatted_answer = split_at_whitespace(answer)
    return formatted_answer

# Interactive loop
print("Karen AI (type 'quit' to exit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = generate_response(user_input)
    print("Assistant:", response)
