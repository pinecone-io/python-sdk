Exceptions
==========

All exceptions raised by the Pinecone SDK derive from :class:`~pinecone.errors.exceptions.PineconeError`
so that a single ``except PineconeError`` block can catch every SDK error if desired.

.. contents:: Sections
   :local:
   :depth: 1

Class Hierarchy
---------------

.. code-block:: text

   PineconeError
   ├── PineconeValueError  (also ValueError)
   ├── PineconeTypeError   (also TypeError)
   ├── PineconeConnectionError
   ├── PineconeTimeoutError  (also TimeoutError)
   ├── ResponseParsingError
   ├── IndexInitFailedError
   ├── IndexTerminatedError
   └── ApiError
       ├── ConflictError (409)
       ├── NotFoundError (404)
       ├── ForbiddenError (403)
       ├── UnauthorizedError (401)
       ├── PaymentRequiredError (402)
       ├── FailedPreconditionError (412)
       ├── RateLimitError (429)
       └── ServiceError (5xx)

Base & Configuration Errors
----------------------------

.. autoexception:: pinecone.errors.exceptions.PineconeError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.PineconeValueError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.PineconeTypeError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.ResponseParsingError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.IndexInitFailedError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.IndexTerminatedError
   :members:
   :show-inheritance:

Network Errors
--------------

.. autoexception:: pinecone.errors.exceptions.PineconeConnectionError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.PineconeTimeoutError
   :members:
   :show-inheritance:

API Errors
----------

.. autoexception:: pinecone.errors.exceptions.ApiError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.NotFoundError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.ConflictError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.UnauthorizedError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.ForbiddenError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.PaymentRequiredError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.FailedPreconditionError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.RateLimitError
   :members:
   :show-inheritance:

.. autoexception:: pinecone.errors.exceptions.ServiceError
   :members:
   :show-inheritance:

Deprecated Aliases
------------------

The following names are kept for backwards compatibility and will be removed in a future
major release.  New code should use the canonical names listed above.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Alias
     - Canonical class
     - Notes
   * - ``PineconeException``
     - :class:`~pinecone.errors.exceptions.PineconeError`
     - Legacy base-class name
   * - ``PineconeApiException``
     - :class:`~pinecone.errors.exceptions.ApiError`
     - Legacy API error name
   * - ``PineconeConfigurationError``
     - :class:`~pinecone.errors.exceptions.PineconeValueError`
     - Legacy configuration error
   * - ``PineconeProtocolError``
     - :class:`~pinecone.errors.exceptions.PineconeError`
     - Legacy protocol error
   * - ``PineconeApiTypeError``
     - :class:`~pinecone.errors.exceptions.PineconeTypeError`
     - Legacy type error
   * - ``PineconeApiValueError``
     - :class:`~pinecone.errors.exceptions.PineconeValueError`
     - Legacy value error
   * - ``PineconeApiAttributeError``
     - :class:`~pinecone.errors.exceptions.PineconeError`
     - Legacy attribute error
   * - ``PineconeApiKeyError``
     - :class:`~pinecone.errors.exceptions.PineconeError`
     - Legacy key error
   * - ``NotFoundException``
     - :class:`~pinecone.errors.exceptions.NotFoundError`
     - Legacy 404 error
   * - ``UnauthorizedException``
     - :class:`~pinecone.errors.exceptions.UnauthorizedError`
     - Legacy 401 error
   * - ``ForbiddenException``
     - :class:`~pinecone.errors.exceptions.ForbiddenError`
     - Legacy 403 error
   * - ``ServiceException``
     - :class:`~pinecone.errors.exceptions.ServiceError`
     - Legacy 5xx error
   * - ``ListConversionException``
     - :class:`~pinecone.errors.exceptions.PineconeError`
     - Legacy list conversion error
   * - ``ValidationError``
     - :class:`~pinecone.errors.exceptions.PineconeValueError`
     - Legacy validation alias
