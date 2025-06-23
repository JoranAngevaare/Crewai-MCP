from crewai import Agent, Task, Crew, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import sys
import platform
from pathlib import Path
import os
import warnings
from pydantic import PydanticDeprecatedSince20
from dotenv import load_dotenv
import traceback
import subprocess
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


def get_available_llm():
    """Get the first available LLM from environment variables"""
    llm_configs = [
        {
            "name": "Ollama Local",
            "model": "ollama/qwen3:8b",
            "api_key_env": None,  # No API key needed for local
            "temperature": 0.0,
            "kw": {"base_url": "http://localhost:11434"},
        },
        {
            "name": "Ollama Local",
            "model": "ollama/qwen3:4b",
            "api_key_env": None,  # No API key needed for local
            "temperature": 0.7,
            "kw": {"base_url": "http://localhost:11434"},
        },
    ]

    print("🔍 Checking available LLM providers...")

    for config in llm_configs:
        try:
            if config["api_key_env"] is None:
                # For local models like Ollama, try without API key
                print(f"⚡ Trying {config['name']} (Local)...")
                llm = LLM(
                    model=config["model"],
                    temperature=config["temperature"],
                    max_tokens=10000,
                    **config.get("kw", {}),
                    # stream=True
                )
                print(f"✅ Using {config['name']}: {config['model']}")
                return llm
            else:
                api_key = os.getenv(config["api_key_env"])
                if api_key:
                    print(f"⚡ Trying {config['name']}...")
                    llm = LLM(
                        model=config["model"],
                        temperature=config["temperature"],
                        api_key=api_key,
                    )
                    print(f"✅ Using {config['name']}: {config['model']}")
                    return llm
                else:
                    print(f"⚠️  {config['name']} API key not found in environment")
        except Exception as e:
            print(f"❌ {config['name']} failed: {str(e)[:100]}...")
            continue

    raise ValueError(f"No LLM model found")


# Configure LLM with fallback options
llm = get_available_llm()

# Suppress warnings
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

# Get current directory
base_dir = Path(__file__).parent.resolve()

print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Base directory: {base_dir}")


def check_search_server():
    """Check if the Python search server exists"""
    server_path = base_dir / "servers" / "search_server.py"
    if server_path.exists():
        print(f"✓ Python search server found: {server_path}")
        return True
    else:
        print(f"✗ Python search server not found: {server_path}")
        return False


def get_working_servers():
    """Get list of working server configurations"""
    working_servers = []

    print("\n" + "=" * 50)
    print("DIAGNOSING MCP SERVERS")
    print("=" * 50)

    # Check Python search server
    search_server_available = check_search_server()
    if search_server_available:
        search_server_params = StdioServerParameters(
            command="python",
            args=[
                str(base_dir / "servers" / "search_server.py"),
            ],
            env={"UV_PYTHON": "3.13", **os.environ},
        )
        working_servers.append(("Python Search Server", search_server_params))
        print("✓ Python search server configured")
    else:
        print("✗ Skipping Python search server (server file not found)")

    print(f"\nFound {len(working_servers)} server configurations")
    return working_servers


class CustomMCPServerAdapter(MCPServerAdapter):
    """Custom MCP Server Adapter with increased timeout"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = 90  # Increase timeout to 90 seconds


def test_servers_individually(server_configs):
    """Test each server individually to identify problematic ones"""
    working_servers = []

    print("\n" + "=" * 50)
    print("TESTING SERVERS INDIVIDUALLY")
    print("=" * 50)

    for name, server_params in server_configs:
        print(f"\nTesting {name}...")
        try:
            with CustomMCPServerAdapter([server_params]) as tools:
                print(f"✓ {name} connected successfully!")
                print(f"  Available tools: {[tool.name for tool in tools]}")
                working_servers.append(server_params)
        except Exception as e:
            print(f"✗ {name} failed: {str(e)[:100]}...")
            continue

    return working_servers


class Summary(BaseModel):
    summary: str = Field(description="A detailed summary of the research findings")
    list_of_references: str = Field(description="Where are answers based on")


def create_agent_and_tasks(tools=None):
    """Create agent and tasks with or without tools"""
    tools_list = tools or []

    # Adjust role and tasks based on available tools
    if tools_list:
        tool_names = [getattr(tool, "name", "unknown") for tool in tools_list]
        print(f"Agent will have access to: {tool_names}")

        role = "LocalBreeze AI documentation reader"
        goal = "Find reliable answers in KNMI documentation"
        backstory = "An AI interface to reliably search KNMI's data sources and provide reliable, and tracible answers to KNMI employees"
    else:
        raise NotImplementedError("We could, but don't want to run like this")

    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools_list,
        llm=llm,
        verbose=True,
    )

    tasks = [
        Task(
            description="Find answers to the topic '{topic}' using available MCP tools. Answer in a consise manner. Keep track of the source of the answer.",
            expected_output="A detailed summary of research findings with references. Point to the sources used. The final response should be in the format of a pydantic model Summary",
            agent=agent,
            output_pydantic=Summary,
        ),
        # Task(
        #     description="Create a detailed summary of your research findings. Include references.",
        #     expected_output="A detailed summary of research findings, preferably saved as a text file if filesystem access is available. The final response should be in the format of a pydantic model Summary",
        #     agent=agent,
        #     output_pydantic=Summary
        # )
    ]

    return agent, tasks


def main():
    """Main function to run the CrewAI application"""
    # Get available server configurations
    server_configs = get_working_servers()

    if not server_configs:
        print("\n⚠️  No MCP servers available. Running in fallback mode only.")
        run_fallback_mode()
        return

    # Test servers individually to find working ones
    working_server_params = test_servers_individually(server_configs)

    if not working_server_params:
        print("\n⚠️  No MCP servers are working. Running in fallback mode.")
        run_fallback_mode()
        return

    try:
        print(f"\n✓ Using {len(working_server_params)} working MCP server(s)")
        print("Initializing MCP Server Adapter...")

        with CustomMCPServerAdapter(working_server_params) as tools:
            print(f"Successfully connected to MCP servers!")
            print(f"Available tools: {[tool.name for tool in tools]}")

            # Create agent and tasks with MCP tools
            agent, tasks = create_agent_and_tasks(tools)

            # Create crew with error handling
            crew = Crew(
                agents=[agent],
                tasks=tasks,
                verbose=True,
                reasoning=True,
            )

            # Get user input
            import argparse

            parser = argparse.ArgumentParser(description="make plots")
            parser.add_argument("-Q", "--question", type=str)
            args = parser.parse_args()

            topic = (
                args.question
                or input(
                    "\nPlease provide a search for KNMI documentation resources: ",
                ).strip()
            )

            # Execute crew with retry mechanism
            max_retries = 0
            for attempt in range(max_retries + 2):
                try:
                    print(f"\nStarting research on: {topic} (Attempt {attempt + 1})")
                    response = crew.kickoff(inputs={"topic": topic})
                    # print("\n" + "="*50)
                    # print("FINAL RESULT FROM THE AGENT")
                    # print("="*50)

                    print(f"Summary task output :{tasks[0].output}")
                    # print(f"Summary task output :{tasks[1].output}")
                    return response
                except Exception as e:
                    if attempt < max_retries:
                        print(f"⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}...")
                        print(f"🔄 Retrying... ({attempt + 2}/{max_retries + 1})")
                        continue
                    else:
                        print(f"❌ All attempts failed. Error: {e}")
                        raise e

    except Exception as e:
        print(f"Error running with MCP tools: {e}")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    print("🚀 Starting CrewAI Localbreeze")
    result = main()
    # print(result)
