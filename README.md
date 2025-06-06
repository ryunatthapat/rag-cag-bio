# RAG & CAG Biographies Demo CLI

## Environment Variables

The following environment variables must be set (see `.env` example):

- `OPENAI_API_KEY`: Your OpenAI API key for accessing OpenAI models (required for RAG).
- `HF_TOKEN`: Your HuggingFace token for accessing HuggingFace models (required for CAG).
- `QDRANT_URL`: The URL of your Qdrant vector database instance (required for RAG retrieval).
- `QDRANT_API_KEY`: The API key for authenticating with your Qdrant instance.

Place these variables in a `.env` file in the project root. The application will load them automatically using `python-dotenv`.

---

(Additional setup, usage, and architecture documentation will be added as the project progresses.) 