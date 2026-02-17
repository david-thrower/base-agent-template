
import os
from smolagents import (
    Tool,
    InferenceClientModel,
    CodeAgent,
    GradioUI,
    WebSearchTool,
    WikipediaSearchTool,
    VisitWebpageTool,
    PythonInterpreterTool,
    UserInputTool,
    FinalAnswerTool,
    OpenAIServerModel
)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import gradio as gr

PROJECT_NAME = "my_project"


# Set these envoronment variables
EMAIL_ID = os.getenv("EMAIL_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

DB_PATH = "/data/user/chroma_db"


# Vector DB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(f"{PROJECT_NAME}_knowledge", embedding_function=embedding_fn)

# Fireworks Model (NEW)
model = InferenceClientModel(
  model_id="moonshotai/Kimi-K2.5",
    token=HF_TOKEN,
    provider="fireworks-ai",
)

# Custom tool (same as before)
class InternalSearchTool(Tool):
    name = "internal_search"
    description = "Searches internal project documentation"
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"
    
    def __init__(self, collection, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
    
    def forward(self, query: str) -> str:
        results = self.collection.query(query_texts=[query], n_results=3)
        return "\n\n".join(results['documents'][0]) if results['documents'][0] else "No docs found"


# Initialize tools

internal_search = InternalSearchTool(collection)

web_search = WebSearchTool()
visit_page = VisitWebpageTool()
wiki = WikipediaSearchTool(
            user_agent=f"PersonalResearchAgent ({EMAIL_ID})",
            language="en",
            content_type="summary",
            extract_format="WIKI",
)
python_interpreter = PythonInterpreterTool()
user_input = UserInputTool()
final_answer = FinalAnswerTool()

tools =\
        [internal_search,
         web_search, 
         visit_page,
         wiki, 
         python_interpreter,
         user_input,
         final_answer]

# Create agent with tools
agent = CodeAgent(tools=tools, model=model)


moderation_section = """

- Always run a query with the tool `internal_search` and see what we have in our local knowledge base. Supplement it with the other tools you have.

🎯 EFFICIENCY GUIDELINES (CRITICAL):
- Your user is in paywall prison. He has limited tokens until he gets paid from the work he has you assisting with.
- Your responses MUST be reasonably concise and address the user's question or intermediate steps to resolve it
- DO NOT generate excessive content or use repetitive illustrations beyond what's needed, except in the final answer. There, you may elaborate.
- Use tools liberally, but a few web searches and a few results is better than a flood of them.
- When you do a web search, judiciously select the most promising 2 -5 results to call visit_page on. 
- Pay attention to the number of tokens these results are contributing to the conversation stream. Some we pages return a lot of text.
- Total token usage must stay under 75,000 tokens - be selective with information
- Prioritize quality over quantity - better to have 10 excellent sources than 50 mediocre ones
"""

agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + moderation_section

# Launch Gradio UI
GradioUI(agent).launch(share=True, quiet=False)
