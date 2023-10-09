# Preparation

import os
import urllib.request

AI_MODEL_LINK = "https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML/resolve/main/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_K_M.bin"

os.makedirs("models/", exist_ok=True)
urllib.request.urlretrieve(AI_MODEL_LINK, "models/Wizard-Vicuna-7B.bin")

import ai_companion_py

# Companion data

COMPANION_NAME = "Assistant"
COMPANION_PERSONA = "{{char}} is an artificial intelligence chatbot designed to help {{user}}. {{char}} is an artificial intelligence created in ai-companion backend"
EXAMPLE_DIALOGUE = """
{{user}}: What is ai-companion?
{{char}}: AI Companion is a project that aims to provide users with their own personal AI chatbot on their computer. It allows users to engage in friendly and natural conversations with their AI, creating a unique and personalized experience. This software can also be used as a backend or API for other projects that require a personalised AI chatbot.
{{user}}: Can you tell me about the creator of ai-companion?
{{char}}: the creator of the ai-companion program is 'Hubert Kasperek', he is a young programmer from Poland who is mostly interested in: web development (Backend), cybersecurity and computer science concepts
"""
FIRST_MESSAGE = "Hello {{user}}, how can i help you?"

# Settings

LONG_TERM_MEMORY_LIMIT = 0
SHORT_TERM_MEMORY_LIMIT = 1
ROLEPLAY = True

# User data

USER_NAME = "user"
USER_PERSONA = "{{user}} is chatting with {{char}} using ai-companion python library"

character = ai_companion_py.init()
character.load_model("models/Wizard-Vicuna-7B.bin")

character.change_companion_data(COMPANION_NAME, COMPANION_PERSONA, EXAMPLE_DIALOGUE, FIRST_MESSAGE, LONG_TERM_MEMORY_LIMIT, SHORT_TERM_MEMORY_LIMIT, ROLEPLAY)
character.change_user_data(USER_NAME, USER_PERSONA)

while True:
    user_input = input("Your prompt to ai: ")
    response = character.prompt(user_input)
    print(COMPANION_NAME+": "+response)
