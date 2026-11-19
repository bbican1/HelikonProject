from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: str = "ollama"
    llm_api_key: str = ""
    llm_model: str = "llama3"

    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    embedding_api_key: str = ""

    chromadb_path: str = "./data/chromadb"

    class Config:
        env_file = ".env"
        case_sensitive = False

    gen_pipeline_planner_model:str = "llama3.2:latest"
    gen_pipeline_validator_model:str = "llama3.2:latest"
    gen_pipeline_models: dict[str, str] = {
        "mistral-nemo:latest":
            "Leistungsfähiges Allround-Modell mit Unterstützung für Tool-Calling. Geeignet für komplexe Fragen, Agentensteuerung und strukturierte Aufgaben.",

        "qwen2.5-coder:7b":
            "Technisch orientiertes Modell mit Tool-Calling-Unterstützung. Besonders geeignet für Programmierung, technische Erklärungen und Aufgaben mit externen Tools.",

        "mistral:7b":
            "Kompaktes Allround-Modell mit Tool-Calling-Unterstützung. Geeignet für allgemeine Fragen und einfache Agentenaufgaben.",

        "qwen3.5:2b":
            "Kleines und schnelles Modell mit Tool-Calling-Unterstützung. Geeignet für einfache Aufgaben mit geringem Ressourcenverbrauch.",
    }

settings = Settings()
