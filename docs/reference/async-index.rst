AsyncIndex
==========

Obtain an ``AsyncIndex`` via :meth:`pinecone.AsyncPinecone.index`, which resolves
the host for you:

.. code-block:: python

   from pinecone import AsyncPinecone

   pc = AsyncPinecone(api_key="your-api-key")

   async with await pc.index("my-index") as idx:
       stats = await idx.describe_index_stats()

Constructing one directly is the other option, and it needs no client — pass the
index host and an API key yourself:

.. code-block:: python

   from pinecone import AsyncIndex

   async with AsyncIndex(
       host="my-index-abc123.svc.pinecone.io",
       api_key="your-api-key",
   ) as idx:
       stats = await idx.describe_index_stats()

``AsyncIndex`` mirrors :class:`~pinecone.index.Index` but every method is an
``async def``. It is an async context manager; call
:meth:`~pinecone.async_client.async_index.AsyncIndex.close` (or use ``async with``)
to release the underlying HTTP connection pool.

**Method groups:**

- **Vectors** — :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert_from_dataframe`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.upsert_records`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.query`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.query_namespaces`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.fetch`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.fetch_by_metadata`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.update`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.delete`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_paginated`
- **Stats** — :meth:`~pinecone.async_client.async_index.AsyncIndex.describe_index_stats`
- **Documents** — :attr:`~pinecone.async_client.async_index.AsyncIndex.documents`, a
  lazily-instantiated :class:`~pinecone.async_client.documents.AsyncDocuments`
  namespace for schema-based indexes (``index.documents.upsert``, ``.search``,
  ``.fetch``, ``.delete``, ``.update``, ``.list``, ``.batch_upsert``).
- **Integrated Inference** — :meth:`~pinecone.async_client.async_index.AsyncIndex.search`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.search_records`
- **Namespaces** — :meth:`~pinecone.async_client.async_index.AsyncIndex.create_namespace`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.describe_namespace`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.delete_namespace`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_namespaces`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_namespaces_paginated`
- **Bulk Import** — :meth:`~pinecone.async_client.async_index.AsyncIndex.start_import`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.describe_import`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.cancel_import`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_imports`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.list_imports_paginated`
- **Lifecycle** — :attr:`~pinecone.async_client.async_index.AsyncIndex.host`,
  :meth:`~pinecone.async_client.async_index.AsyncIndex.close`

.. autoclass:: pinecone.async_client.async_index.AsyncIndex
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__, __aenter__, __aexit__
   :member-order: bysource


AsyncDocuments
--------------

.. autoclass:: pinecone.async_client.documents.AsyncDocuments
   :members:
   :undoc-members: False
   :show-inheritance:
