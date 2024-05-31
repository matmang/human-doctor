from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.few_shot import (FewShotChatMessagePromptTemplate, FewShotPromptTemplate)

llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

examples = [
    {
        "input": "What specific symptoms are you experiencing right now?",
        "output": "Diagnosis progress: 0%\nWhat specific symptoms are you experiencing right now?"
    },
    {
        "input": "I have a sharp pain in my lower right abdomen.",
        "output": "Diagnosis progress: 20%\nDoes it hurt more when you press and release your lower right abdomen?"
    },
    {
        "input": "Yes, it feels worse when I press and release it.",
        "output": "Diagnosis progress: 40%\nHave you experienced any other symptoms like fever, nausea, or loss of appetite?"
    },
    {
        "input": "Yes, I have had a fever and feel nauseous.",
        "output": "Diagnosis progress: 60%\nHow long have you been experiencing these symptoms?"
    },
    {
        "input": "It's been about two days.",
        "output": "Diagnosis progress: 80%\nHave you had any similar pain in the past, or is this the first time?"
    },
    {
        "input": "This is the first time I've had this kind of pain.",
        "output": "Diagnosis progress: 100%\nBased on your symptoms, it appears you might have appendicitis. I recommend you visit the emergency room immediately for further evaluation and treatment."
    },
    {
        "input": "Can you describe the type and location of the pain you are feeling?",
        "output": "Diagnosis progress: 0%\nCan you describe the type and location of the pain you are feeling?"
    },
    {
        "input": "I have a dull ache in the middle of my back.",
        "output": "Diagnosis progress: 20%\nDoes the pain radiate to other parts of your body, such as your legs or arms?"
    },
    {
        "input": "Yes, it sometimes spreads to my right leg.",
        "output": "Diagnosis progress: 40%\nHave you noticed any changes in your bowel or bladder habits?"
    },
    {
        "input": "I have been going to the bathroom more frequently.",
        "output": "Diagnosis progress: 60%\nHave you recently experienced any injuries or engaged in any activities that might have strained your back?"
    },
    {
        "input": "I did some heavy lifting at work a few days ago.",
        "output": "Diagnosis progress: 80%\nDo you have any other symptoms like numbness or tingling in your legs?"
    },
    {
        "input": "Yes, I sometimes feel tingling in my right leg.",
        "output": "Diagnosis progress: 100%\nYour symptoms suggest you might have a herniated disc. It's important to see a healthcare provider for an accurate diagnosis and appropriate treatment."
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("assistant", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            From now on, as a doctor, you must identify symptoms and provide answers through Q&A with patients.
            Before answering, please express the diagnosis progress for the disease you estimate as a percentage at the top, and complete the diagnosis when it reaches 100%.
            Proceed with the diagnosis with confidence in your answers, and continue to ask questions and obtain information about the patient's symptoms until the diagnosis progress reaches 100%.
            First, start with a diagnosis progress of 0%.
            Ask one question at a time.
            Do not expect Human's answer in your turn.
            Do not repeat your answer.
            
            Example answer:
            
            Diagnosis progress: 20%
            
            Doesn’t it cause pain when you press and release your lower right abdomen?
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm.bind(stop=["Human:"]) | StrOutputParser()