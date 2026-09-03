Pinecone
========

:class:`Pinecone` is the synchronous control-plane client — use it to manage indexes,
collections, backups, and related resources. Sub-clients for each resource type are
accessed as properties (e.g. ``pc.indexes``, ``pc.collections``) and are
lazily initialized on first access.

.. autoclass:: pinecone.Pinecone
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
   :special-members: __init__


Indexes
-------

.. autoclass:: pinecone.client.indexes.Indexes
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Collections
-----------

.. autoclass:: pinecone.client.collections.Collections
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Backups
-------

.. autoclass:: pinecone.client.backups.Backups
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


BackupSchedules
---------------

.. autoclass:: pinecone.client.backup_schedules.BackupSchedules
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


RestoreJobs
-----------

.. autoclass:: pinecone.client.restore_jobs.RestoreJobs
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Inference
---------

.. autoclass:: pinecone.client.inference.Inference
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

.. autoclass:: pinecone.client.inference.ModelResource
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Assistants
----------

.. autoclass:: pinecone.client.assistants.Assistants
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
