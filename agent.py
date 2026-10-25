
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()

def build_agent():
    return Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        markdown=True,
        instructions="You are a helpful and expert travel agent."
    )

if __name__ == "__main__":
    agent = build_agent()
    agent.print_response(
        "My budget is 1 lakh INR. Should I travel to Goa or Phuket?"
    )
