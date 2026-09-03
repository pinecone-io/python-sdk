Admin
=====

The ``Admin`` client manages organizations, projects, API keys, users, invites, service
accounts, and role bindings.  It uses OAuth2
client credentials (service account) rather than an API key, and is the right tool
for control-plane operations such as creating projects and rotating keys.  It is
synchronous only — there is no async form of this client.

The users, invites, service-account, and role-binding namespaces are new in ``2026-07``.
See the :ref:`admin-oauth` section of :doc:`../migration/v10-migration` for the
per-operation release notes and an end-to-end RBAC walkthrough.

.. autoclass:: pinecone.admin.Admin
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
   :special-members: __init__


Organizations
-------------

.. autoclass:: pinecone.admin.organizations.Organizations
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Projects
--------

.. autoclass:: pinecone.admin.projects.Projects
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


API Keys
--------

.. autoclass:: pinecone.admin.api_keys.ApiKeys
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Users
-----

.. autoclass:: pinecone.admin.users.Users
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Invites
-------

.. autoclass:: pinecone.admin.invites.Invites
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Service Accounts
----------------

The OAuth principals the ``Admin`` client itself authenticates as.  ``create`` and
``rotate_secret`` are the only operations that return a ``client_secret``, and each returns it
exactly once — capture it or rotate again.

.. autoclass:: pinecone.admin.service_accounts.ServiceAccounts
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource


Role Bindings
-------------

The whole of Pinecone's authorization model: one ``role`` granted to one principal — user,
service account, API key, or pending invite — at one scope, either the organization or a single
project.  Nothing else confers permissions, and bindings are immutable, so a role change is a
``create`` followed by a ``delete``.

.. autoclass:: pinecone.admin.role_bindings.RoleBindings
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource
