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
from pydantic import BaseModel,Field

# Load environment variables
load_dotenv()

def get_available_llm():
    """Get the first available LLM from environment variables"""
    llm_configs = [
        {
            "name": "Ollama Local",
            "model": "ollama/qwen3:1.7b",
            "api_key_env": None,  # No API key needed for local
            "temperature": 0.7, 
            "kw":{"base_url":"http://localhost:11434"}
            
        }
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
                    max_tokens=1000,
                    **config.get('kw', {}),
                    stream=True  
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
                        api_key=api_key
                    )
                    print(f"✅ Using {config['name']}: {config['model']}")
                    return llm
                else:
                    print(f"⚠️  {config['name']} API key not found in environment")
        except Exception as e:
            print(f"❌ {config['name']} failed: {str(e)[:100]}...")
            continue
    
    # Fallback to basic configuration if all else fails
    print("⚠️  Using fallback LLM configuration...")
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY", "")
    )

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
    
    print("\n" + "="*50)
    print("DIAGNOSING MCP SERVERS")
    print("="*50)
    
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
        self.timeout = 1  # Increase timeout to 90 seconds

def test_servers_individually(server_configs):
    """Test each server individually to identify problematic ones"""
    working_servers = []
    
    print("\n" + "="*50)
    print("TESTING SERVERS INDIVIDUALLY")
    print("="*50)
    
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

def create_agent_and_tasks(tools=None):
    """Create agent and tasks with or without tools"""
    tools_list = tools or []
    
    # Adjust role and tasks based on available tools
    if tools_list:
        tool_names = [getattr(tool, 'name', 'unknown') for tool in tools_list]
        print(f"Agent will have access to: {tool_names}")
        
        role = "AI Research Creator with Tools"
        goal = "Research topics thoroughly using available MCP tools, create comprehensive diagrams, and save summaries"
        backstory = "An AI researcher and creator that specializes in using MCP tools to gather information, create visual representations, and save findings."
    else:
        role = "AI Research Creator"
        goal = "Research topics using built-in knowledge and create comprehensive analysis"
        backstory = "An AI researcher that specializes in analyzing topics and providing detailed insights using available knowledge."
    
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools_list,
        llm=llm,
        verbose=True,
    )
    
    if tools_list:
        research_task = Task(
            description="Research the topic '{topic}' thoroughly using available MCP tools.",
            expected_output="A comprehensive research summary and, if possible, a successfully created diagram/image illustrating the topic.",
            agent=agent,
        )
        
    else:
        research_task = Task(
            description="Research and analyze the topic '{topic}' thoroughly using your knowledge. Provide detailed insights about how it works, including key components, processes, and relationships.",
            expected_output="A comprehensive analysis and explanation of the topic with detailed insights.",
            agent=agent,
        )
        
    return agent, [research_task,]

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
            topic = input("\nPlease provide a topic to research: ").strip()
            if not topic:
                topic = "artificial intelligence"
                print(f"No topic provided, using default: {topic}")
            
            # Execute crew with retry mechanism
            max_retries = 0
            for attempt in range(max_retries + 1):
                try:
                    print(f"\nStarting research on: {topic} (Attempt {attempt + 1})")
                    response = crew.kickoff(inputs={"topic": topic})
                    # print("\n" + "="*50)
                    # print("FINAL RESULT FROM THE AGENT")
                    # print("="*50)
                
                    print(f"Summary task output :{tasks[0].output}")
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
        print("\nFalling back to basic agent without MCP tools...")
        run_fallback_mode()

def run_fallback_mode():
    """Run the application without MCP tools"""
    print("\n" + "="*50)
    print("RUNNING IN FALLBACK MODE")
    print("="*50)
    
    # Create fallback agent without MCP tools but with LLM
    agent, tasks = create_agent_and_tasks()
    
    crew = Crew(
        agents=[agent],
        tasks=tasks,
        verbose=True,
        reasoning=True,
    )
    
    # Get user input for fallback
    topic = input("Please provide a topic to research (fallback mode): ").strip()
    if not topic:
        topic = "artificial intelligence"
        print(f"No topic provided, using default: {topic}")
    
    print(f"\nStarting research on: {topic} (without MCP tools)")
    result = crew.kickoff(inputs={"topic": topic})
    print("\n" + "="*50)
    print("FINAL RESULT (Fallback Mode):")
    print("="*50)
    print(result["summary"])
    return result["summary"]

if __name__ == "__main__":
    print("🚀 Starting CrewAI MCP Demo")
    print("\n📋 Setup Instructions:")
    print("   For more MCP servers, update Node.js to v18+: https://nodejs.org")
    print("   Add API keys to .env file for additional LLM providers")
    print("   Supported: GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, BRAVE_API_KEY")
    result = main()
    #print(result)