# Authentication

## Getting an API key

Sign in to the [Pinecone console](https://app.pinecone.io), navigate to **API Keys**,
and create or copy an existing key.

## Environment variable (recommended)

Set `PINECONE_API_KEY` before running your application:

```bash
export PINECONE_API_KEY=your-key-here
```

The client reads it automatically when you call `Pinecone()` with no arguments:

```python
from pinecone import Pinecone

pc = Pinecone()
```

## Explicit argument

Pass the key directly if you manage secrets through your own mechanism:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key-here")
```

## Missing key error

If no API key can be resolved from either the argument or the environment variable,
the client raises `PineconeValueError` on construction:

```python
from pinecone import Pinecone, PineconeValueError

try:
    pc = Pinecone()
except PineconeValueError as e:
    print(e)  # "No API key provided. Pass api_key='...' or set the PINECONE_API_KEY environment variable."
```

## Security best practices

Never hardcode API keys in source files. Use environment variables or a secrets
manager. As a lightweight alternative, the `python-dotenv` package loads a local
`.env` file (which you add to `.gitignore`):

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()  # reads PINECONE_API_KEY from .env
pc = Pinecone()
```
