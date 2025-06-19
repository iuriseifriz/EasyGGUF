import os
from llama_cpp import Llama

# ==========================
# PATH CONFIGURATION
# ==========================
BASE_DIR = "your/directory/here/EasyGGUF"
MODEL_INPUT_DIR = os.path.join(BASE_DIR, "MODEL_INPUT")
MODEL_PATH = os.path.join(MODEL_INPUT_DIR, "model.gguf")

# ==========================
# PROMPT TEMPLATE
# ==========================
def create_chat_prompt(user_message, conversation_history=""):
    """
    Creates a structured prompt for conversation.
    Adjust this template according to your model's expected format.
    """
    system_prompt = "Put here your initial prompt!"
    
    # Basic template – adjust according to your model
    prompt = f"""<|system|>
{system_prompt}
<|end|>
{conversation_history}<|user|>
{user_message}
<|end|>
<|assistant|>
"""
    
    return prompt

# ==========================
# LOAD MODEL AND RUN
# ==========================
def main():
    print("[INFO] Initializing model...")

    try:
        # Initialize the model with optimized parameters
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,  # Larger context for conversations
            n_threads=8,  # Adjust according to your CPU
            verbose=False
        )
        print("[OK] Model successfully loaded.\nType 'exit' to quit.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    conversation_history = ""
    
    while True:
        # Capture user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        
        if not user_input:
            continue
            
        # Create the structured prompt
        full_prompt = create_chat_prompt(user_input, conversation_history)
        
        try:
            # Generate response
            output = llm(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|user|>", "<|end|>", "You:", "\n\nYou:"],
                echo=False,  # Do not repeat the prompt in output
                stream=False
            )

            # Extract and clean the response
            if output and "choices" in output and len(output["choices"]) > 0:
                response = output["choices"][0]["text"].strip()
                
                # Remove leftover formatting tokens
                response = response.replace("<|assistant|>", "").replace("<|end|>", "").strip()
                
                if response:
                    print(f"Chatbot: {response}")
                    
                    # Update conversation history (keep limited to avoid context overflow)
                    conversation_history += f"<|user|>\n{user_input}\n<|end|>\n<|assistant|>\n{response}\n<|end|>\n"
                    
                    # Limit history to prevent overly long context
                    if len(conversation_history) > 2000:
                        # Keep only the last interactions
                        lines = conversation_history.split('\n')
                        conversation_history = '\n'.join(lines[-20:])
                else:
                    print("Chatbot: [No response generated]")
            else:
                print("Chatbot: [Error in response generation]")
                
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")

if __name__ == "__main__":
    main()
