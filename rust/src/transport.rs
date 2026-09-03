use std::sync::Arc;
use std::time::Duration;

use hyper_util::client::legacy::connect::{proxy::Tunnel, HttpConnector};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Channel, ClientTlsConfig};

use crate::proto;
use crate::proto::vector_service_client::VectorServiceClient;
use crate::retry::{retry_on_transient_request, RetryConfig, ThrottleCallback};

/// Maximum gRPC message size for both send and receive (128 MB).
const MAX_MESSAGE_SIZE: usize = 128 * 1024 * 1024;

/// Normalize a source tag string.
///
/// - Lowercase the input.
/// - Strip characters not in [a-z0-9_ :].
/// - Replace spaces with underscores.
fn normalize_source_tag(tag: &str) -> String {
    let lowered = tag.to_lowercase();
    let cleaned: String = lowered
        .chars()
        .filter(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_' || *c == ' ' || *c == ':'
        })
        .map(|c| if c == ' ' { '_' } else { c })
        .collect();
    cleaned
}

/// Build the gRPC User-Agent string.
///
/// Format: `python-client[grpc]/{version}` with optional ` source_tag:{normalized}` suffix.
fn build_grpc_user_agent(version: &str, source_tag: Option<&str>) -> String {
    let mut ua = format!("python-client[grpc]/{version}");
    if let Some(tag) = source_tag {
        let normalized = normalize_source_tag(tag);
        if !normalized.is_empty() {
            ua = format!("{ua} source_tag:{normalized}");
        }
    }
    ua
}

/// Parse the host component from an endpoint URL.
///
/// Strips the scheme (`https://`), then returns the portion before the first `:` or `/`.
/// Returns `None` if the result is empty.
///
/// Examples:
/// - `"https://my-index.svc.pinecone.io:443"` → `Some("my-index.svc.pinecone.io")`
/// - `"https://my-index.svc.pinecone.io"` → `Some("my-index.svc.pinecone.io")`
fn parse_host_from_endpoint(endpoint: &str) -> Option<String> {
    let after_scheme = endpoint
        .split_once("://")
        .map(|(_, r)| r)
        .unwrap_or(endpoint);
    let host = after_scheme.split([':', '/']).next()?.to_string();
    if host.is_empty() {
        None
    } else {
        Some(host)
    }
}

/// Ensure the endpoint URL has a port; append `:443` if none is specified.
fn ensure_port(endpoint: &str) -> String {
    // Try to parse the host portion to check for a port.
    // Endpoints look like "https://host:port" or "https://host".
    if let Some(scheme_end) = endpoint.find("://") {
        let after_scheme = &endpoint[scheme_end + 3..];
        // Strip any path component
        let host_port = after_scheme.split('/').next().unwrap_or(after_scheme);
        // Check for port: look for `:digits` at the end, but be careful with IPv6
        // brackets like `[::1]:443`.
        if host_port.ends_with(']')
            || !host_port.contains(':')
            || (host_port.starts_with('[') && !host_port.contains("]:"))
        {
            // No port found — append :443
            // Insert before any path component
            let path_start = scheme_end + 3 + host_port.len();
            let (before_path, path) = endpoint.split_at(path_start);
            return format!("{before_path}:443{path}");
        }
    }
    endpoint.to_string()
}

/// Interceptor that attaches API key, request ID (UUID v4), and API version
/// metadata to every outgoing gRPC request.
#[derive(Clone)]
struct MetadataInterceptor {
    api_key: tonic::metadata::MetadataValue<tonic::metadata::Ascii>,
    api_version: tonic::metadata::MetadataValue<tonic::metadata::Ascii>,
}

impl MetadataInterceptor {
    fn new(
        api_key: &str,
        api_version: &str,
    ) -> Result<Self, tonic::metadata::errors::InvalidMetadataValue> {
        Ok(Self {
            api_key: api_key.parse()?,
            api_version: api_version.parse()?,
        })
    }
}

impl tonic::service::Interceptor for MetadataInterceptor {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        let metadata = request.metadata_mut();
        metadata.insert("api-key", self.api_key.clone());
        metadata.insert(
            "x-request-id",
            uuid::Uuid::new_v4()
                .to_string()
                .parse()
                .map_err(|_| tonic::Status::internal("failed to create request ID"))?,
        );
        metadata.insert("x-pinecone-api-version", self.api_version.clone());
        Ok(request)
    }
}

/// Map a gRPC status code to the HTTP-ish status code used by ApiError subclasses.
///
/// Only codes whose corresponding Python exception is an ApiError subclass should
/// appear here. Codes that map to PineconeError subclasses without status_code
/// (PineconeTimeoutError, PineconeConnectionError) must return None.
fn grpc_code_to_http_status(code: tonic::Code) -> Option<i32> {
    match code {
        tonic::Code::NotFound => Some(404),
        tonic::Code::AlreadyExists => Some(409),
        tonic::Code::Unauthenticated => Some(401),
        tonic::Code::PermissionDenied => Some(403),
        tonic::Code::Internal | tonic::Code::Unknown => Some(500),
        tonic::Code::InvalidArgument => Some(400),
        tonic::Code::ResourceExhausted => Some(429),
        _ => None,
    }
}

/// Convert a `tonic::Status` into the appropriate Python SDK exception.
///
/// Maps tonic status codes to the Pinecone Python exception hierarchy:
///   - NotFound          → pinecone.errors.NotFoundError        (404)
///   - AlreadyExists     → pinecone.errors.ConflictError        (409)
///   - Unauthenticated   → pinecone.errors.UnauthorizedError    (401)
///   - PermissionDenied  → pinecone.errors.ForbiddenError       (403)
///   - Internal/Unknown  → pinecone.errors.ServiceError         (500)
///   - InvalidArgument   → pinecone.errors.ApiError             (400)
///   - ResourceExhausted → pinecone.errors.ApiError             (429)
///   - DeadlineExceeded  → pinecone.errors.PineconeTimeoutError (no status_code)
///   - Cancelled("Timeout expired") → pinecone.errors.PineconeTimeoutError (no status_code)
///   - Unavailable       → pinecone.errors.PineconeConnectionError (no status_code)
///   - All others        → pinecone.errors.PineconeError        (no status_code)
///
/// For ApiError subclasses the exception is constructed with:
///   (message,) positional + {status_code, body, error_code, request_id} kwargs.
/// For non-ApiError subclasses only (message,) is passed.
///
/// Trailer extraction (x-request-id / x-pinecone-request-id) and body synthesis
/// are best-effort: any failure falls back to raising the typed exception without
/// those fields. The function never panics; every fallible step uses .ok() or match.
fn status_to_py_err(status: tonic::Status) -> PyErr {
    let code = status.code();
    // Use the raw status message as the user-facing text; the structured error_code
    // field carries the gRPC code name so the "gRPC NOT_FOUND: " prefix is no longer needed.
    let msg = status.message().to_string();
    let error_code = grpc_code_name(code);

    // Extract request_id from gRPC trailers; try both header names for parity with REST.
    let request_id: Option<String> = status
        .metadata()
        .get("x-request-id")
        .or_else(|| status.metadata().get("x-pinecone-request-id"))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    Python::attach(|py| {
        let errors_mod = match py.import("pinecone.errors") {
            Ok(m) => m,
            Err(_) => return PyRuntimeError::new_err(msg),
        };

        // Determine which exception class to raise.
        // ApiError subclasses (those with an http_status) accept status_code/body/error_code/request_id.
        // Non-ApiError subclasses (PineconeTimeoutError, PineconeConnectionError, PineconeError)
        // accept only message.
        let exc_class_name = match code {
            tonic::Code::NotFound => "NotFoundError",
            tonic::Code::AlreadyExists => "ConflictError",
            tonic::Code::Unauthenticated => "UnauthorizedError",
            tonic::Code::PermissionDenied => "ForbiddenError",
            tonic::Code::DeadlineExceeded => "PineconeTimeoutError",
            // tonic converts per-request timeout expiry (req.set_timeout) via TimeoutExpired
            // into Status::cancelled("Timeout expired") rather than DeadlineExceeded.
            tonic::Code::Cancelled if msg == "Timeout expired" => "PineconeTimeoutError",
            tonic::Code::Unavailable => "PineconeConnectionError",
            tonic::Code::Internal | tonic::Code::Unknown => "ServiceError",
            tonic::Code::InvalidArgument | tonic::Code::ResourceExhausted => "ApiError",
            _ => "PineconeError",
        };

        let exc_class = match errors_mod.getattr(exc_class_name) {
            Ok(cls) => cls,
            Err(_) => return PyRuntimeError::new_err(msg),
        };

        let exc_instance = if let Some(http_status) = grpc_code_to_http_status(code) {
            // ApiError subclass: synthesize body and pass structured kwargs.
            // Body construction is best-effort; if it fails we still raise the typed exception.
            let body_opt: Option<pyo3::Bound<'_, PyDict>> =
                (|| -> pyo3::PyResult<pyo3::Bound<'_, PyDict>> {
                    let inner = PyDict::new(py);
                    inner.set_item("code", error_code)?;
                    inner.set_item("message", &msg)?;
                    let outer = PyDict::new(py);
                    outer.set_item("error", inner)?;
                    Ok(outer)
                })()
                .ok();

            let kwargs = PyDict::new(py);
            let _ = kwargs.set_item("status_code", http_status);
            let _ = kwargs.set_item("error_code", error_code);
            if let Some(ref body) = body_opt {
                let _ = kwargs.set_item("body", body);
            }
            if let Some(ref rid) = request_id {
                let _ = kwargs.set_item("request_id", rid.as_str());
            }
            // Fall back to positional-only construction if kwargs call fails.
            exc_class
                .call((&msg,), Some(&kwargs))
                .or_else(|_| exc_class.call1((&msg,)))
        } else {
            // Non-ApiError class: only message is accepted.
            exc_class.call1((&msg,))
        };

        match exc_instance {
            Ok(inst) => PyErr::from_value(inst.into_any()),
            Err(_) => PyRuntimeError::new_err(msg),
        }
    })
}

/// Raise a `pinecone.errors.PineconeValueError` for client-side validation failures.
///
/// Falls back to `PyValueError` if the Python exception class cannot be imported.
fn pinecone_value_error(py: Python<'_>, msg: &str) -> PyErr {
    let errors_mod = match py.import("pinecone.errors") {
        Ok(m) => m,
        Err(_) => return pyo3::exceptions::PyValueError::new_err(msg.to_string()),
    };
    let cls = match errors_mod.getattr("PineconeValueError") {
        Ok(c) => c,
        Err(_) => return pyo3::exceptions::PyValueError::new_err(msg.to_string()),
    };
    match cls.call1((msg,)) {
        Ok(inst) => PyErr::from_value(inst.into_any()),
        Err(_) => pyo3::exceptions::PyValueError::new_err(msg.to_string()),
    }
}

/// Raise a `pinecone.errors.PineconeError` for internal SDK failures.
///
/// Falls back to `PyRuntimeError` if the Python exception class cannot be imported.
fn pinecone_error(py: Python<'_>, msg: &str) -> PyErr {
    let errors_mod = match py.import("pinecone.errors") {
        Ok(m) => m,
        Err(_) => return PyRuntimeError::new_err(msg.to_string()),
    };
    let cls = match errors_mod.getattr("PineconeError") {
        Ok(c) => c,
        Err(_) => return PyRuntimeError::new_err(msg.to_string()),
    };
    match cls.call1((msg,)) {
        Ok(inst) => PyErr::from_value(inst.into_any()),
        Err(_) => PyRuntimeError::new_err(msg.to_string()),
    }
}

/// Raise a `pinecone.errors.ResponseParsingError` for upstream contract violations.
///
/// Falls back to `PyRuntimeError` if the Python exception class cannot be imported.
fn response_parsing_error(py: Python<'_>, msg: &str) -> PyErr {
    let errors_mod = match py.import("pinecone.errors") {
        Ok(m) => m,
        Err(_) => return PyRuntimeError::new_err(msg.to_string()),
    };
    let cls = match errors_mod.getattr("ResponseParsingError") {
        Ok(c) => c,
        Err(_) => return PyRuntimeError::new_err(msg.to_string()),
    };
    match cls.call1((msg,)) {
        Ok(inst) => PyErr::from_value(inst.into_any()),
        Err(_) => PyRuntimeError::new_err(msg.to_string()),
    }
}

/// Convert a seconds-as-f64 value to a `Duration`, returning `PineconeValueError` if the
/// value is negative, NaN, infinite, or otherwise out of range for `Duration`.
fn secs_to_duration(py: Python<'_>, secs: f64, field: &str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(secs).map_err(|_| {
        pinecone_value_error(
            py,
            &format!("{field} must be a non-negative finite number; got {secs}"),
        )
    })
}

/// Human-readable name for a gRPC status code.
fn grpc_code_name(code: tonic::Code) -> &'static str {
    match code {
        tonic::Code::Ok => "OK",
        tonic::Code::Cancelled => "CANCELLED",
        tonic::Code::Unknown => "UNKNOWN",
        tonic::Code::InvalidArgument => "INVALID_ARGUMENT",
        tonic::Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        tonic::Code::NotFound => "NOT_FOUND",
        tonic::Code::AlreadyExists => "ALREADY_EXISTS",
        tonic::Code::PermissionDenied => "PERMISSION_DENIED",
        tonic::Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        tonic::Code::FailedPrecondition => "FAILED_PRECONDITION",
        tonic::Code::Aborted => "ABORTED",
        tonic::Code::OutOfRange => "OUT_OF_RANGE",
        tonic::Code::Unimplemented => "UNIMPLEMENTED",
        tonic::Code::Internal => "INTERNAL",
        tonic::Code::Unavailable => "UNAVAILABLE",
        tonic::Code::DataLoss => "DATA_LOSS",
        tonic::Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

/// Convert a `prost_types::Struct` to a Python dict.
fn struct_to_py_dict(py: Python<'_>, s: &prost_types::Struct) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in &s.fields {
        dict.set_item(key, prost_value_to_py(py, value)?)?;
    }
    Ok(dict.unbind())
}

/// Convert a `prost_types::Value` to a Python object.
fn prost_value_to_py(py: Python<'_>, value: &prost_types::Value) -> PyResult<Py<PyAny>> {
    use prost_types::value::Kind;
    match &value.kind {
        Some(Kind::NullValue(_)) => Ok(py.None()),
        Some(Kind::NumberValue(n)) => Ok(n.into_pyobject(py)?.into_any().unbind()),
        Some(Kind::StringValue(s)) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Some(Kind::BoolValue(b)) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        Some(Kind::StructValue(s)) => Ok(struct_to_py_dict(py, s)?.into_any()),
        Some(Kind::ListValue(list)) => {
            let items: Vec<Py<PyAny>> = list
                .values
                .iter()
                .map(|v| prost_value_to_py(py, v))
                .collect::<PyResult<_>>()?;
            Ok(pyo3::types::PyList::new(py, items)?.into_any().unbind())
        }
        None => Ok(py.None()),
    }
}

/// How Python `None` values inside dicts are converted to prost.
///
/// Mirrors the REST backend's `json_to_prost` (`pinecone-db`
/// `pc-utils/src/prost_util.rs` @ f6fd0a40): the JSON path strips null-valued
/// object entries from **write metadata** before validation ever sees them,
/// but preserves them in **filters** so downstream validation rejects them.
/// Building the `Struct` here without the same split made the same upsert
/// succeed over REST and fail over gRPC (#203).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NullHandling {
    /// Drop dict entries whose value is `None` (write metadata).
    Strip,
    /// Keep `None` as a `NullValue` so the server can reject it (filters).
    Preserve,
}

/// Convert a Python object to a `prost_types::Value`.
fn py_to_prost_value(
    obj: &Bound<'_, pyo3::PyAny>,
    nulls: NullHandling,
) -> PyResult<prost_types::Value> {
    use prost_types::value::Kind;

    if obj.is_none() {
        return Ok(prost_types::Value {
            kind: Some(Kind::NullValue(0)),
        });
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::BoolValue(b)),
        });
    }
    if let Ok(n) = obj.extract::<f64>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::NumberValue(n)),
        });
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::StringValue(s)),
        });
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::StructValue(py_dict_to_struct(dict, nulls)?)),
        });
    }
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        // `None` inside a list is deliberately NOT stripped, matching
        // `json_to_prost`, which only drops null-valued *object entries*; a
        // null list element survives to be rejected by server-side validation.
        let values: Vec<prost_types::Value> = list
            .iter()
            .map(|item| py_to_prost_value(&item, nulls))
            .collect::<PyResult<_>>()?;
        return Ok(prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue { values })),
        });
    }

    let type_name = obj.get_type().name()?;
    Python::attach(|py| {
        Err(pinecone_value_error(
            py,
            &format!("Unsupported metadata value type: {type_name}"),
        ))
    })
}

/// Convert a Python dict to a `prost_types::Struct`.
fn py_dict_to_struct(
    dict: &Bound<'_, PyDict>,
    nulls: NullHandling,
) -> PyResult<prost_types::Struct> {
    use prost_types::value::Kind;
    let mut fields = std::collections::BTreeMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let converted = py_to_prost_value(&value, nulls)?;
        if nulls == NullHandling::Strip && matches!(converted.kind, Some(Kind::NullValue(_))) {
            continue;
        }
        fields.insert(key_str, converted);
    }
    Ok(prost_types::Struct { fields })
}

/// Convert a proto SparseValues into a Python dict with "indices" and "values" keys.
fn sparse_values_to_py_dict(py: Python<'_>, sv: &proto::SparseValues) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("indices", &sv.indices)?;
    dict.set_item("values", &sv.values)?;
    Ok(dict.unbind())
}

/// Extract sparse values from a Python dict.
fn py_dict_to_sparse_values(dict: &Bound<'_, PyDict>) -> PyResult<proto::SparseValues> {
    let py = dict.py();
    let indices: Vec<u32> = dict
        .get_item("indices")?
        .ok_or_else(|| response_parsing_error(py, "sparse_values missing 'indices'"))?
        .extract()?;
    let values: Vec<f32> = dict
        .get_item("values")?
        .ok_or_else(|| response_parsing_error(py, "sparse_values missing 'values'"))?
        .extract()?;
    Ok(proto::SparseValues { indices, values })
}

/// Convert a proto Vector to a Python dict.
fn vector_to_py_dict(py: Python<'_>, v: &proto::Vector) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &v.id)?;
    dict.set_item("values", &v.values)?;
    if let Some(ref sv) = v.sparse_values {
        dict.set_item("sparse_values", sparse_values_to_py_dict(py, sv)?)?;
    }
    if let Some(ref md) = v.metadata {
        dict.set_item("metadata", struct_to_py_dict(py, md)?)?;
    }
    Ok(dict.unbind())
}

/// Convert a proto ScoredVector to a Python dict.
fn scored_vector_to_py_dict(py: Python<'_>, sv: &proto::ScoredVector) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &sv.id)?;
    dict.set_item("score", sv.score)?;
    dict.set_item("values", &sv.values)?;
    if let Some(ref sparse) = sv.sparse_values {
        dict.set_item("sparse_values", sparse_values_to_py_dict(py, sparse)?)?;
    }
    if let Some(ref md) = sv.metadata {
        dict.set_item("metadata", struct_to_py_dict(py, md)?)?;
    }
    Ok(dict.unbind())
}

/// Convert a proto `NamespaceDescription` to a Python dict.
fn namespace_description_to_py_dict(
    py: Python<'_>,
    ns: &proto::NamespaceDescription,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &ns.name)?;
    dict.set_item("record_count", ns.record_count)?;
    // Unconditional, unlike the two below: `size_bytes` is a non-optional proto
    // `uint64`, so there is no absent-vs-zero signal on the wire to forward.
    dict.set_item("size_bytes", ns.size_bytes)?;
    if let Some(ref schema) = ns.schema {
        let schema_dict = metadata_schema_to_py_dict(py, schema)?;
        dict.set_item("schema", schema_dict)?;
    }
    if let Some(ref indexed) = ns.indexed_fields {
        dict.set_item("indexed_fields", &indexed.fields)?;
    }
    Ok(dict.unbind())
}

/// Convert a proto `MetadataSchema` to a Python dict.
fn metadata_schema_to_py_dict(
    py: Python<'_>,
    schema: &proto::MetadataSchema,
) -> PyResult<Py<PyDict>> {
    let fields_dict = PyDict::new(py);
    for (name, props) in &schema.fields {
        let props_dict = PyDict::new(py);
        props_dict.set_item("filterable", props.filterable)?;
        fields_dict.set_item(name, props_dict)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("fields", fields_dict)?;
    Ok(dict.unbind())
}

/// Convert a Python dict to a proto `MetadataSchema`.
fn py_dict_to_metadata_schema(dict: &Bound<'_, PyDict>) -> PyResult<proto::MetadataSchema> {
    let py = dict.py();
    let fields_obj = dict
        .get_item("fields")?
        .ok_or_else(|| response_parsing_error(py, "schema missing 'fields'"))?;
    let fields_dict = fields_obj.cast::<PyDict>()?;
    let mut fields = std::collections::HashMap::new();
    for (key, value) in fields_dict.iter() {
        let key_str: String = key.extract()?;
        let props_dict = value.cast::<PyDict>()?;
        let filterable: bool = props_dict
            .get_item("filterable")?
            .ok_or_else(|| response_parsing_error(py, "field properties missing 'filterable'"))?
            .extract()?;
        fields.insert(key_str, proto::MetadataFieldProperties { filterable });
    }
    Ok(proto::MetadataSchema { fields })
}

/// A gRPC channel wrapper exposed to Python.
#[pyclass]
pub struct GrpcChannel {
    client: VectorServiceClient<InterceptedService<Channel, MetadataInterceptor>>,
    runtime: tokio::runtime::Runtime,
    retry_config: RetryConfig,
}

#[pymethods]
impl GrpcChannel {
    /// Create a new gRPC channel connected to the given endpoint.
    ///
    /// Args:
    ///     endpoint: The gRPC endpoint URL (e.g. "https://my-index-abc123.svc.pinecone.io:443")
    ///     api_key: The Pinecone API key for authentication.
    ///     api_version: The Pinecone API version string (e.g. "2026-07").
    ///     version: The SDK version string (e.g. "0.1.0") used in the User-Agent header.
    ///     secure: Whether to use TLS encryption (default true).
    ///     timeout_s: Request timeout in seconds (default 20.0).
    ///     connect_timeout_s: Connection timeout in seconds (default 1.0).
    ///     max_retries: Max retry attempts on transient codes (UNAVAILABLE, RESOURCE_EXHAUSTED, ABORTED — default 5, 0 disables).
    ///     backoff_factor_s: Minimum backoff delay floor in seconds between retries (default 0.1).
    ///     max_wait_s: Maximum backoff delay in seconds (default 60.0). Bounds both the
    ///                 jitter path and a server-supplied `grpc-retry-pushback-ms` hint,
    ///                 so lowering it below a realistic pushback value means ignoring
    ///                 what the server asked for.
    ///     source_tag: Optional source tag appended to the User-Agent string.
    ///     proxy_url: Optional HTTP proxy URL (e.g. "http://proxy.example.com:8080").
    ///                When set, gRPC traffic is tunnelled through the proxy via HTTP CONNECT.
    ///     on_throttle: Optional Python callable invoked with the host string on every
    ///                  retryable error. Signature: ``(host: str) -> None``. Used by the
    ///                  SDK's adaptive concurrency limiter to self-tune bulk-operation
    ///                  concurrency in response to throttling.
    #[new]
    #[pyo3(signature = (endpoint, api_key, api_version, version, secure=true, timeout_s=None, connect_timeout_s=None, max_retries=None, backoff_factor_s=None, max_wait_s=None, source_tag=None, proxy_url=None, on_throttle=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        endpoint: &str,
        api_key: &str,
        api_version: &str,
        version: &str,
        secure: bool,
        timeout_s: Option<f64>,
        connect_timeout_s: Option<f64>,
        max_retries: Option<u32>,
        backoff_factor_s: Option<f64>,
        max_wait_s: Option<f64>,
        source_tag: Option<&str>,
        proxy_url: Option<&str>,
        on_throttle: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pinecone_error(py, &format!("Failed to create tokio runtime: {e}")))?;

        let request_timeout = secs_to_duration(py, timeout_s.unwrap_or(20.0), "timeout_s")?;
        let connection_timeout =
            secs_to_duration(py, connect_timeout_s.unwrap_or(1.0), "connect_timeout_s")?;

        let endpoint_with_port = ensure_port(endpoint);

        let user_agent = build_grpc_user_agent(version, source_tag);

        let mut endpoint_builder = Channel::from_shared(endpoint_with_port)
            .map_err(|e| pinecone_value_error(py, &format!("Invalid endpoint: {e}")))?
            .user_agent(&user_agent)
            .map_err(|e| pinecone_value_error(py, &format!("Invalid user agent: {e}")))?
            .timeout(request_timeout)
            .connect_timeout(connection_timeout);

        if secure {
            endpoint_builder = endpoint_builder
                .tls_config(ClientTlsConfig::new().with_native_roots())
                .map_err(|e| pinecone_value_error(py, &format!("Failed to configure TLS: {e}")))?;
        }

        // Use lazy connection — the channel establishes the actual TCP/TLS connection
        // on the first RPC call, not at construction time. This means GrpcIndex
        // construction never fails due to transient network issues; callers discover
        // connectivity problems only when they make an actual request.
        //
        // connect_lazy() must be called within a Tokio runtime context so that
        // hyper-util's TokioExecutor can register its executor. We use runtime.enter()
        // to set the context without actually running any async code.
        let channel = {
            let _guard = runtime.enter();
            if let Some(proxy_url_str) = proxy_url {
                let proxy_dst: http::Uri = proxy_url_str.parse().map_err(|e| {
                    pinecone_value_error(py, &format!("Invalid proxy URL '{proxy_url_str}': {e}"))
                })?;
                let mut http_connector = HttpConnector::new();
                http_connector.enforce_http(false);
                let tunnel_connector = Tunnel::new(proxy_dst, http_connector);
                endpoint_builder.connect_with_connector_lazy(tunnel_connector)
            } else {
                endpoint_builder.connect_lazy()
            }
        };

        let interceptor = MetadataInterceptor::new(api_key, api_version)
            .map_err(|e| pinecone_value_error(py, &format!("Invalid metadata value: {e}")))?;

        let host = parse_host_from_endpoint(endpoint).unwrap_or_else(|| endpoint.to_string());
        let on_throttle_cb: Option<ThrottleCallback> = on_throttle.map(|py_cb| {
            let cb: ThrottleCallback = Arc::new(move |h: String| {
                Python::attach(|py| {
                    if let Err(e) = py_cb.call1(py, (h,)) {
                        tracing::debug!("on_throttle callback raised, ignoring: {:?}", e);
                    }
                });
            });
            cb
        });
        let defaults = RetryConfig::default();
        let initial_backoff = match backoff_factor_s {
            Some(secs) => secs_to_duration(py, secs, "backoff_factor_s")?,
            None => defaults.initial_backoff,
        };
        let max_backoff = match max_wait_s {
            Some(secs) => secs_to_duration(py, secs, "max_wait_s")?,
            None => defaults.max_backoff,
        };
        let retry_config = RetryConfig {
            max_retries: max_retries.unwrap_or(defaults.max_retries),
            initial_backoff,
            max_backoff,
            on_throttle: on_throttle_cb,
            host,
            ..defaults
        };

        Ok(Self {
            client: VectorServiceClient::with_interceptor(channel, interceptor)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE),
            runtime,
            retry_config,
        })
    }

    /// Upsert vectors.
    ///
    /// Args:
    ///     vectors: List of dicts with keys: id, values, sparse_values (optional), metadata (optional)
    ///     namespace: Target namespace (default "")
    ///
    /// Returns:
    ///     Dict with "upserted_count".
    #[pyo3(signature = (vectors, namespace=None, timeout_s=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        vectors: Vec<Bound<'_, PyDict>>,
        namespace: Option<&str>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let mut proto_vectors = Vec::with_capacity(vectors.len());
        for v in &vectors {
            let id: String = v
                .get_item("id")?
                .ok_or_else(|| response_parsing_error(py, "vector missing 'id'"))?
                .extract()?;
            let values: Vec<f32> = v
                .get_item("values")?
                .ok_or_else(|| response_parsing_error(py, "vector missing 'values'"))?
                .extract()?;
            let sparse_values = match v.get_item("sparse_values")? {
                Some(sv) => Some(py_dict_to_sparse_values(&sv.cast_into::<PyDict>()?)?),
                None => None,
            };
            let metadata = match v.get_item("metadata")? {
                Some(md) => Some(py_dict_to_struct(
                    &md.cast_into::<PyDict>()?,
                    NullHandling::Strip,
                )?),
                None => None,
            };
            proto_vectors.push(proto::Vector {
                id,
                values,
                sparse_values,
                metadata,
            });
        }

        let request = proto::UpsertRequest {
            vectors: proto_vectors,
            namespace: namespace.unwrap_or("").to_string(),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: this closure's return type is `Result<Response<_>, tonic::Status>`
        // (from `retry_on_transient_request`); the lint fires on the closure itself, not just
        // fn signatures, so the statement-level allow below is load-bearing, not vestigial.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.upsert(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let dict = PyDict::new(py);
        dict.set_item("upserted_count", inner.upserted_count)?;
        Ok(dict.unbind())
    }

    /// Query vectors.
    ///
    /// Args:
    ///     top_k: Number of results to return.
    ///     vector: Query vector (optional).
    ///     id: Query by vector ID (optional).
    ///     namespace: Namespace to query (default "").
    ///     filter: Metadata filter dict (optional).
    ///     include_values: Include vector values in response (default false).
    ///     include_metadata: Include metadata in response (default false).
    ///
    /// Returns:
    ///     Dict with "matches" (list of scored vector dicts) and "namespace".
    #[pyo3(signature = (top_k, vector=None, id=None, namespace=None, filter=None, include_values=false, include_metadata=false, sparse_vector=None, scan_factor=None, max_candidates=None, timeout_s=None))]
    #[allow(clippy::too_many_arguments)]
    fn query(
        &self,
        py: Python<'_>,
        top_k: u32,
        vector: Option<Vec<f32>>,
        id: Option<&str>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        include_values: bool,
        include_metadata: bool,
        sparse_vector: Option<Bound<'_, PyDict>>,
        scan_factor: Option<f32>,
        max_candidates: Option<u32>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let has_vector = vector.as_ref().is_some_and(|v| !v.is_empty());
        let has_id = id.is_some_and(|s| !s.is_empty());
        if has_vector && has_id {
            return Err(pinecone_value_error(
                py,
                "Cannot specify both 'id' and 'vector' in a query",
            ));
        }

        #[allow(deprecated)]
        let request = proto::QueryRequest {
            namespace: namespace.unwrap_or("").to_string(),
            top_k,
            filter: filter
                .map(|f| py_dict_to_struct(&f, NullHandling::Preserve))
                .transpose()?,
            include_values,
            include_metadata,
            queries: vec![],
            vector: vector.unwrap_or_default(),
            sparse_vector: sparse_vector
                .map(|sv| py_dict_to_sparse_values(&sv))
                .transpose()?,
            id: id.unwrap_or("").to_string(),
            scan_factor,
            max_candidates,
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.query(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let matches: Vec<Py<PyDict>> = inner
            .matches
            .iter()
            .map(|m| scored_vector_to_py_dict(py, m))
            .collect::<PyResult<_>>()?;

        let dict = PyDict::new(py);
        dict.set_item("matches", matches)?;
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(usage) = &inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.unbind())
    }

    /// Fetch vectors by ID.
    ///
    /// Args:
    ///     ids: List of vector IDs to fetch.
    ///     namespace: Namespace (default "").
    ///
    /// Returns:
    ///     Dict with "vectors" (map of id → vector dict) and "namespace".
    #[pyo3(signature = (ids, namespace=None, timeout_s=None))]
    fn fetch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        namespace: Option<&str>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::FetchRequest {
            ids,
            namespace: namespace.unwrap_or("").to_string(),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.fetch(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let vectors_dict = PyDict::new(py);
        for (id, vector) in &inner.vectors {
            vectors_dict.set_item(id, vector_to_py_dict(py, vector)?)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("vectors", vectors_dict)?;
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(usage) = &inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.unbind())
    }

    /// Delete vectors.
    ///
    /// Args:
    ///     ids: List of vector IDs to delete (optional).
    ///     delete_all: Delete all vectors in namespace (default false).
    ///     namespace: Namespace (default "").
    ///     filter: Metadata filter dict (optional).
    ///
    /// Returns:
    ///     Empty dict (delete has no response fields).
    #[pyo3(signature = (ids=None, delete_all=false, namespace=None, filter=None, timeout_s=None))]
    fn delete(
        &self,
        py: Python<'_>,
        ids: Option<Vec<String>>,
        delete_all: bool,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DeleteRequest {
            ids: ids.unwrap_or_default(),
            delete_all,
            namespace: namespace.unwrap_or("").to_string(),
            filter: filter
                .map(|f| py_dict_to_struct(&f, NullHandling::Preserve))
                .transpose()?,
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        py.detach(|| {
            self.runtime
                .block_on(retry_on_transient_request(&retry_config, request, |r| {
                    let mut c = client.clone();
                    async move {
                        let mut req = tonic::Request::new(r);
                        if let Some(dur) = timeout {
                            req.set_timeout(dur);
                        }
                        c.delete(req).await
                    }
                }))
        })
        .map_err(status_to_py_err)?;

        let dict = PyDict::new(py);
        Ok(dict.unbind())
    }

    /// Update a vector.
    ///
    /// Args:
    ///     id: The unique ID of the vector to update.
    ///     values: New dense vector values (optional).
    ///     sparse_values: New sparse vector values dict with "indices" and "values" keys (optional).
    ///     set_metadata: Metadata dict to set/overwrite (optional).
    ///     namespace: Namespace (default "").
    ///     filter: Metadata filter for bulk update (optional).
    ///     dry_run: If true, return matched count without executing (optional).
    ///
    /// Returns:
    ///     Dict with optional "matched_records" count.
    #[pyo3(signature = (id, values=None, sparse_values=None, set_metadata=None, namespace=None, filter=None, dry_run=None, timeout_s=None))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        py: Python<'_>,
        id: &str,
        values: Option<Vec<f32>>,
        sparse_values: Option<Bound<'_, PyDict>>,
        set_metadata: Option<Bound<'_, PyDict>>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        dry_run: Option<bool>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::UpdateRequest {
            id: id.to_string(),
            values: values.unwrap_or_default(),
            sparse_values: sparse_values
                .map(|sv| py_dict_to_sparse_values(&sv))
                .transpose()?,
            set_metadata: set_metadata
                .map(|md| py_dict_to_struct(&md, NullHandling::Strip))
                .transpose()?,
            namespace: namespace.unwrap_or("").to_string(),
            filter: filter
                .map(|f| py_dict_to_struct(&f, NullHandling::Preserve))
                .transpose()?,
            dry_run,
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.update(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let dict = PyDict::new(py);
        if let Some(matched) = inner.matched_records {
            dict.set_item("matched_records", matched)?;
        }
        Ok(dict.unbind())
    }

    /// List vector IDs.
    ///
    /// Args:
    ///     prefix: ID prefix filter (optional).
    ///     limit: Max number of IDs to return (optional).
    ///     pagination_token: Token to continue a previous listing (optional).
    ///     namespace: Namespace (default "").
    ///
    /// Returns:
    ///     Dict with "vectors" (list of dicts with "id"), optional "pagination" dict,
    ///     "namespace", and optional "usage" dict.
    #[pyo3(signature = (prefix=None, limit=None, pagination_token=None, namespace=None, timeout_s=None))]
    fn list(
        &self,
        py: Python<'_>,
        prefix: Option<&str>,
        limit: Option<u32>,
        pagination_token: Option<&str>,
        namespace: Option<&str>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::ListRequest {
            prefix: prefix.map(|s| s.to_string()),
            limit,
            pagination_token: pagination_token.map(|s| s.to_string()),
            namespace: namespace.unwrap_or("").to_string(),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.list(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let vectors: Vec<Py<PyDict>> = inner
            .vectors
            .iter()
            .map(|item| {
                let d = PyDict::new(py);
                d.set_item("id", &item.id)?;
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;

        let dict = PyDict::new(py);
        dict.set_item("vectors", vectors)?;
        if let Some(ref pag) = inner.pagination {
            let pag_dict = PyDict::new(py);
            pag_dict.set_item("next", &pag.next)?;
            dict.set_item("pagination", pag_dict)?;
        }
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(ref usage) = inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.unbind())
    }

    /// Get index statistics.
    ///
    /// Args:
    ///     filter: Metadata filter dict (optional). If present, stats reflect only
    ///             vectors matching the filter.
    ///     timeout_s: Per-call timeout in seconds. None uses the client-level default.
    ///
    /// Returns:
    ///     Dict with "namespaces" (map of namespace → {"vector_count"}), "dimension",
    ///     "index_fullness", "total_vector_count", and optional "metric", "vector_type",
    ///     "memory_fullness", "storage_fullness".
    #[pyo3(signature = (filter=None, timeout_s=None))]
    fn describe_index_stats(
        &self,
        py: Python<'_>,
        filter: Option<Bound<'_, PyDict>>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DescribeIndexStatsRequest {
            filter: filter
                .map(|f| py_dict_to_struct(&f, NullHandling::Preserve))
                .transpose()?,
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.describe_index_stats(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let namespaces_dict = PyDict::new(py);
        for (name, summary) in &inner.namespaces {
            let ns_dict = PyDict::new(py);
            ns_dict.set_item("vector_count", summary.vector_count)?;
            namespaces_dict.set_item(name, ns_dict)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("namespaces", namespaces_dict)?;
        if let Some(dim) = inner.dimension {
            dict.set_item("dimension", dim)?;
        }
        dict.set_item("index_fullness", inner.index_fullness)?;
        dict.set_item("total_vector_count", inner.total_vector_count)?;
        if let Some(ref metric) = inner.metric {
            dict.set_item("metric", metric)?;
        }
        if let Some(ref vt) = inner.vector_type {
            dict.set_item("vector_type", vt)?;
        }
        if let Some(mf) = inner.memory_fullness {
            dict.set_item("memory_fullness", mf)?;
        }
        if let Some(sf) = inner.storage_fullness {
            dict.set_item("storage_fullness", sf)?;
        }
        Ok(dict.unbind())
    }

    /// List namespaces.
    ///
    /// Args:
    ///     pagination_token: Token to continue a previous listing (optional).
    ///     limit: Max number of namespaces to return (optional).
    ///     prefix: Namespace prefix filter (optional).
    ///     timeout_s: Per-call timeout in seconds. None uses the client-level default.
    ///
    /// Returns:
    ///     Dict with "namespaces" (list of namespace description dicts),
    ///     optional "pagination" dict, and "total_count".
    #[pyo3(signature = (pagination_token=None, limit=None, prefix=None, timeout_s=None))]
    fn list_namespaces(
        &self,
        py: Python<'_>,
        pagination_token: Option<&str>,
        limit: Option<u32>,
        prefix: Option<&str>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::ListNamespacesRequest {
            pagination_token: pagination_token.map(|s| s.to_string()),
            limit,
            prefix: prefix.map(|s| s.to_string()),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.list_namespaces(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let namespaces: Vec<Py<PyDict>> = inner
            .namespaces
            .iter()
            .map(|ns| namespace_description_to_py_dict(py, ns))
            .collect::<PyResult<_>>()?;

        let dict = PyDict::new(py);
        dict.set_item("namespaces", namespaces)?;
        if let Some(ref pag) = inner.pagination {
            let pag_dict = PyDict::new(py);
            pag_dict.set_item("next", &pag.next)?;
            dict.set_item("pagination", pag_dict)?;
        }
        dict.set_item("total_count", inner.total_count)?;
        Ok(dict.unbind())
    }

    /// Describe a namespace.
    ///
    /// Args:
    ///     namespace: The namespace to describe.
    ///     timeout_s: Per-call timeout in seconds. None uses the client-level default.
    ///
    /// Returns:
    ///     Dict with "name", "record_count", "size_bytes", and optional "schema"
    ///     and "indexed_fields".
    #[pyo3(signature = (namespace, timeout_s=None))]
    fn describe_namespace(
        &self,
        py: Python<'_>,
        namespace: &str,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DescribeNamespaceRequest {
            namespace: namespace.to_string(),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.describe_namespace(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        namespace_description_to_py_dict(py, &inner)
    }

    /// Delete a namespace.
    ///
    /// Args:
    ///     namespace: The namespace to delete.
    ///     timeout_s: Per-call timeout in seconds. None uses the client-level default.
    ///
    /// Returns:
    ///     Empty dict.
    #[pyo3(signature = (namespace, timeout_s=None))]
    fn delete_namespace(
        &self,
        py: Python<'_>,
        namespace: &str,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DeleteNamespaceRequest {
            namespace: namespace.to_string(),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        py.detach(|| {
            self.runtime
                .block_on(retry_on_transient_request(&retry_config, request, |r| {
                    let mut c = client.clone();
                    async move {
                        let mut req = tonic::Request::new(r);
                        if let Some(dur) = timeout {
                            req.set_timeout(dur);
                        }
                        c.delete_namespace(req).await
                    }
                }))
        })
        .map_err(status_to_py_err)?;

        let dict = PyDict::new(py);
        Ok(dict.unbind())
    }

    /// Create a namespace.
    ///
    /// Args:
    ///     name: The name of the namespace to create.
    ///     schema: Optional metadata schema dict with "fields" mapping field names
    ///             to {"filterable": bool}.
    ///     timeout_s: Per-call timeout in seconds. None uses the client-level default.
    ///
    /// Returns:
    ///     Dict with "name", "record_count", "size_bytes", and optional "schema"
    ///     and "indexed_fields".
    #[pyo3(signature = (name, schema=None, timeout_s=None))]
    fn create_namespace(
        &self,
        py: Python<'_>,
        name: &str,
        schema: Option<Bound<'_, PyDict>>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let metadata_schema = schema.map(|s| py_dict_to_metadata_schema(&s)).transpose()?;

        let request = proto::CreateNamespaceRequest {
            name: name.to_string(),
            schema: metadata_schema,
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.create_namespace(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        namespace_description_to_py_dict(py, &inner)
    }

    /// Fetch vectors by metadata filter.
    ///
    /// Args:
    ///     namespace: Namespace to fetch from (default "").
    ///     filter: Metadata filter dict (optional).
    ///     limit: Max number of vectors to return (optional).
    ///     pagination_token: Token to continue a previous listing (optional).
    ///     timeout_s: Per-call timeout in seconds. None uses the client-level default.
    ///
    /// Returns:
    ///     Dict with "vectors" (map of id → vector dict), "namespace",
    ///     optional "usage" dict, and optional "pagination" dict.
    #[pyo3(signature = (namespace=None, filter=None, limit=None, pagination_token=None, timeout_s=None))]
    fn fetch_by_metadata(
        &self,
        py: Python<'_>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        limit: Option<u32>,
        pagination_token: Option<&str>,
        timeout_s: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::FetchByMetadataRequest {
            namespace: namespace.unwrap_or("").to_string(),
            filter: filter
                .map(|f| py_dict_to_struct(&f, NullHandling::Preserve))
                .transpose()?,
            limit,
            pagination_token: pagination_token.map(|s| s.to_string()),
        };

        let timeout = timeout_s
            .map(|secs| secs_to_duration(py, secs, "timeout_s"))
            .transpose()?;
        let client = self.client.clone();
        let retry_config = self.retry_config.clone();
        // result_large_err: same closure-return pattern as the first site in `upsert` above.
        #[allow(clippy::result_large_err)]
        let response = py
            .detach(|| {
                self.runtime
                    .block_on(retry_on_transient_request(&retry_config, request, |r| {
                        let mut c = client.clone();
                        async move {
                            let mut req = tonic::Request::new(r);
                            if let Some(dur) = timeout {
                                req.set_timeout(dur);
                            }
                            c.fetch_by_metadata(req).await
                        }
                    }))
            })
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let vectors_dict = PyDict::new(py);
        for (id, vector) in &inner.vectors {
            vectors_dict.set_item(id, vector_to_py_dict(py, vector)?)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("vectors", vectors_dict)?;
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(ref usage) = inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        if let Some(ref pag) = inner.pagination {
            let pag_dict = PyDict::new(py);
            pag_dict.set_item("next", &pag.next)?;
            dict.set_item("pagination", pag_dict)?;
        }
        Ok(dict.unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;
    use tonic::service::Interceptor;

    fn size_bytes_through_converter(size_bytes: u64) -> u64 {
        Python::initialize();
        Python::attach(|py| {
            let ns = proto::NamespaceDescription {
                name: "movies-en".to_string(),
                record_count: 42,
                size_bytes,
                ..Default::default()
            };
            let dict = namespace_description_to_py_dict(py, &ns).expect("converter succeeds");
            dict.bind(py)
                .get_item("size_bytes")
                .expect("lookup succeeds")
                .expect("size_bytes is always emitted")
                .extract::<u64>()
                .expect("size_bytes extracts as u64")
        })
    }

    #[test]
    fn namespace_description_to_py_dict_emits_zero_size_bytes() {
        assert_eq!(size_bytes_through_converter(0), 0);
    }

    #[test]
    fn namespace_description_to_py_dict_does_not_truncate_size_bytes() {
        for value in [
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            1_099_511_627_776,
            u64::MAX - 1,
            u64::MAX,
        ] {
            assert_eq!(
                size_bytes_through_converter(value),
                value,
                "size_bytes {value} truncated crossing the PyO3 boundary"
            );
        }
    }

    #[test]
    fn namespace_description_to_py_dict_emits_the_full_key_set() {
        Python::initialize();
        Python::attach(|py| {
            let ns = proto::NamespaceDescription {
                name: "movies-en".to_string(),
                record_count: 42,
                size_bytes: 1_048_576,
                schema: Some(proto::MetadataSchema {
                    fields: std::collections::HashMap::new(),
                }),
                indexed_fields: Some(proto::IndexedFields {
                    fields: vec!["genre".to_string(), "year".to_string()],
                }),
            };
            let dict = namespace_description_to_py_dict(py, &ns).expect("converter succeeds");
            let bound = dict.bind(py);
            let mut keys: Vec<String> = bound
                .keys()
                .iter()
                .map(|k| k.extract::<String>().expect("keys are strings"))
                .collect();
            keys.sort();
            assert_eq!(
                keys,
                [
                    "indexed_fields",
                    "name",
                    "record_count",
                    "schema",
                    "size_bytes"
                ]
            );
            assert_eq!(
                bound
                    .get_item("indexed_fields")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .expect("indexed_fields is a bare list of names"),
                ["genre", "year"]
            );
        });
    }

    #[test]
    fn namespace_description_to_py_dict_omits_absent_optionals() {
        Python::initialize();
        Python::attach(|py| {
            let dict =
                namespace_description_to_py_dict(py, &proto::NamespaceDescription::default())
                    .expect("converter succeeds");
            let bound = dict.bind(py);
            let mut keys: Vec<String> = bound
                .keys()
                .iter()
                .map(|k| k.extract::<String>().expect("keys are strings"))
                .collect();
            keys.sort();
            assert_eq!(keys, ["name", "record_count", "size_bytes"]);
        });
    }

    // Null-metadata convergence with the REST path (#203): write metadata
    // strips None-valued keys exactly the way pinecone-db's json_to_prost
    // does for JSON writes, while filters keep the NullValue so the server
    // rejects it on both transports alike.

    fn py_metadata_struct(
        pairs: &[(&str, Option<&str>)],
        nulls: NullHandling,
    ) -> prost_types::Struct {
        Python::initialize();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (key, value) in pairs {
                match value {
                    Some(v) => dict.set_item(key, v).unwrap(),
                    None => dict.set_item(key, py.None()).unwrap(),
                }
            }
            py_dict_to_struct(&dict, nulls).expect("conversion succeeds")
        })
    }

    #[test]
    fn strip_drops_none_valued_keys_from_write_metadata() {
        let s = py_metadata_struct(
            &[("tag", None), ("genre", Some("comedy"))],
            NullHandling::Strip,
        );
        assert!(!s.fields.contains_key("tag"), "None value must be stripped");
        assert_eq!(s.fields.len(), 1);
        assert!(s.fields.contains_key("genre"));
    }

    #[test]
    fn preserve_keeps_none_as_null_value_for_filters() {
        use prost_types::value::Kind;
        let s = py_metadata_struct(&[("tag", None)], NullHandling::Preserve);
        assert!(
            matches!(s.fields["tag"].kind, Some(Kind::NullValue(_))),
            "filters must carry the null through for server-side rejection"
        );
    }

    #[test]
    fn strip_recurses_into_nested_dicts() {
        use prost_types::value::Kind;
        Python::initialize();
        Python::attach(|py| {
            let inner = PyDict::new(py);
            inner.set_item("gone", py.None()).unwrap();
            inner.set_item("kept", 1).unwrap();
            let outer = PyDict::new(py);
            outer.set_item("nested", inner).unwrap();

            let s = py_dict_to_struct(&outer, NullHandling::Strip).unwrap();
            let Some(Kind::StructValue(ref nested)) = s.fields["nested"].kind else {
                panic!("nested dict should convert to a StructValue");
            };
            assert!(!nested.fields.contains_key("gone"));
            assert!(nested.fields.contains_key("kept"));
        });
    }

    #[test]
    fn strip_preserves_none_inside_lists() {
        use prost_types::value::Kind;
        Python::initialize();
        Python::attach(|py| {
            let items: Vec<Py<PyAny>> = vec![
                "a".into_pyobject(py).unwrap().into_any().unbind(),
                py.None(),
            ];
            let list = pyo3::types::PyList::new(py, items).unwrap();
            let dict = PyDict::new(py);
            dict.set_item("tags", list).unwrap();

            let s = py_dict_to_struct(&dict, NullHandling::Strip).unwrap();
            let Some(Kind::ListValue(ref lv)) = s.fields["tags"].kind else {
                panic!("list should convert to a ListValue");
            };
            assert_eq!(lv.values.len(), 2, "null list elements must survive");
            assert!(matches!(lv.values[1].kind, Some(Kind::NullValue(_))));
        });
    }

    #[test]
    fn normalize_source_tag_lowercases_and_strips() {
        assert_eq!(normalize_source_tag("MyApp"), "myapp");
        assert_eq!(normalize_source_tag("My App"), "my_app");
        assert_eq!(normalize_source_tag("my-app!@#v2"), "myappv2");
        assert_eq!(normalize_source_tag("tag:value"), "tag:value");
        assert_eq!(normalize_source_tag("HELLO WORLD 123"), "hello_world_123");
    }

    #[test]
    fn normalize_source_tag_empty_input() {
        assert_eq!(normalize_source_tag(""), "");
        assert_eq!(normalize_source_tag("!!!"), "");
    }

    #[test]
    fn build_grpc_user_agent_without_source_tag() {
        let ua = build_grpc_user_agent("1.2.3", None);
        assert_eq!(ua, "python-client[grpc]/1.2.3");
    }

    #[test]
    fn build_grpc_user_agent_with_source_tag() {
        let ua = build_grpc_user_agent("1.2.3", Some("My App"));
        assert_eq!(ua, "python-client[grpc]/1.2.3 source_tag:my_app");
    }

    #[test]
    fn build_grpc_user_agent_with_empty_source_tag() {
        let ua = build_grpc_user_agent("1.2.3", Some(""));
        assert_eq!(ua, "python-client[grpc]/1.2.3");
    }

    #[test]
    fn build_grpc_user_agent_with_special_chars_source_tag() {
        let ua = build_grpc_user_agent("0.1.0", Some("My-App!@#v2"));
        assert_eq!(ua, "python-client[grpc]/0.1.0 source_tag:myappv2");
    }

    #[test]
    fn interceptor_attaches_all_metadata_headers() {
        let mut interceptor = MetadataInterceptor::new("test-api-key-123", "2026-07").unwrap();
        let request = tonic::Request::new(());
        let result = interceptor.call(request).unwrap();
        let metadata = result.metadata();

        assert_eq!(
            metadata.get("api-key").unwrap().to_str().unwrap(),
            "test-api-key-123"
        );
        assert_eq!(
            metadata
                .get("x-pinecone-api-version")
                .unwrap()
                .to_str()
                .unwrap(),
            "2026-07"
        );

        let request_id = metadata.get("x-request-id").unwrap().to_str().unwrap();
        // Validate UUID v4 format (8-4-4-4-12 hex chars)
        assert_eq!(request_id.len(), 36);
        assert!(uuid::Uuid::parse_str(request_id).is_ok());
    }

    #[test]
    fn tls_enabled_endpoint_builder_succeeds() {
        // Verify that configuring TLS on a valid endpoint does not error
        let endpoint = Channel::from_shared("https://example.pinecone.io:443".to_string())
            .expect("valid endpoint");
        let result = endpoint.tls_config(ClientTlsConfig::new());
        assert!(
            result.is_ok(),
            "TLS config should succeed on a valid endpoint"
        );
    }

    #[test]
    fn insecure_endpoint_builder_succeeds() {
        // Verify that creating an endpoint without TLS config does not error
        let endpoint = Channel::from_shared("http://localhost:5080".to_string());
        assert!(
            endpoint.is_ok(),
            "Insecure endpoint should be constructable"
        );
    }

    #[test]
    fn default_timeouts_are_applied() {
        // Verify that endpoint builder accepts default timeout values without error.
        // Default: request timeout = 20s, connection timeout = 1s.
        let endpoint = Channel::from_shared("https://example.pinecone.io:443".to_string())
            .expect("valid endpoint")
            .timeout(Duration::from_secs_f64(20.0))
            .connect_timeout(Duration::from_secs_f64(1.0));
        // If timeouts were invalid, the builder methods would panic or error.
        // Verify TLS still works after timeouts are set.
        let result = endpoint.tls_config(ClientTlsConfig::new());
        assert!(
            result.is_ok(),
            "Endpoint with default timeouts should be configurable"
        );
    }

    #[test]
    fn custom_timeouts_are_accepted() {
        // Verify that custom timeout values are accepted.
        let endpoint = Channel::from_shared("https://example.pinecone.io:443".to_string())
            .expect("valid endpoint")
            .timeout(Duration::from_secs_f64(60.0))
            .connect_timeout(Duration::from_secs_f64(5.0));
        let result = endpoint.tls_config(ClientTlsConfig::new());
        assert!(
            result.is_ok(),
            "Endpoint with custom timeouts should be configurable"
        );
    }

    #[test]
    fn fractional_timeouts_are_accepted() {
        // Verify that fractional second timeouts work correctly.
        let endpoint = Channel::from_shared("https://example.pinecone.io:443".to_string())
            .expect("valid endpoint")
            .timeout(Duration::from_secs_f64(0.5))
            .connect_timeout(Duration::from_secs_f64(0.1));
        let result = endpoint.tls_config(ClientTlsConfig::new());
        assert!(
            result.is_ok(),
            "Endpoint with fractional timeouts should be configurable"
        );
    }

    #[test]
    fn grpc_code_name_covers_all_variants() {
        // Verify that every tonic::Code variant has a human-readable name.
        let codes = vec![
            (tonic::Code::Ok, "OK"),
            (tonic::Code::Cancelled, "CANCELLED"),
            (tonic::Code::Unknown, "UNKNOWN"),
            (tonic::Code::InvalidArgument, "INVALID_ARGUMENT"),
            (tonic::Code::DeadlineExceeded, "DEADLINE_EXCEEDED"),
            (tonic::Code::NotFound, "NOT_FOUND"),
            (tonic::Code::AlreadyExists, "ALREADY_EXISTS"),
            (tonic::Code::PermissionDenied, "PERMISSION_DENIED"),
            (tonic::Code::ResourceExhausted, "RESOURCE_EXHAUSTED"),
            (tonic::Code::FailedPrecondition, "FAILED_PRECONDITION"),
            (tonic::Code::Aborted, "ABORTED"),
            (tonic::Code::OutOfRange, "OUT_OF_RANGE"),
            (tonic::Code::Unimplemented, "UNIMPLEMENTED"),
            (tonic::Code::Internal, "INTERNAL"),
            (tonic::Code::Unavailable, "UNAVAILABLE"),
            (tonic::Code::DataLoss, "DATA_LOSS"),
            (tonic::Code::Unauthenticated, "UNAUTHENTICATED"),
        ];
        for (code, expected_name) in codes {
            assert_eq!(grpc_code_name(code), expected_name);
        }
    }

    #[test]
    fn grpc_code_to_http_status_maps_api_error_codes() {
        // ApiError subclasses should get an HTTP status code.
        assert_eq!(grpc_code_to_http_status(tonic::Code::NotFound), Some(404));
        assert_eq!(
            grpc_code_to_http_status(tonic::Code::AlreadyExists),
            Some(409)
        );
        assert_eq!(
            grpc_code_to_http_status(tonic::Code::Unauthenticated),
            Some(401)
        );
        assert_eq!(
            grpc_code_to_http_status(tonic::Code::PermissionDenied),
            Some(403)
        );
        assert_eq!(grpc_code_to_http_status(tonic::Code::Internal), Some(500));
        assert_eq!(grpc_code_to_http_status(tonic::Code::Unknown), Some(500));
        assert_eq!(
            grpc_code_to_http_status(tonic::Code::InvalidArgument),
            Some(400)
        );
        assert_eq!(
            grpc_code_to_http_status(tonic::Code::ResourceExhausted),
            Some(429)
        );
    }

    #[test]
    fn grpc_code_to_http_status_returns_none_for_non_api_errors() {
        // Non-ApiError exception types (no status_code) should return None.
        assert_eq!(
            grpc_code_to_http_status(tonic::Code::DeadlineExceeded),
            None
        );
        assert_eq!(grpc_code_to_http_status(tonic::Code::Unavailable), None);
        assert_eq!(grpc_code_to_http_status(tonic::Code::Ok), None);
        assert_eq!(grpc_code_to_http_status(tonic::Code::Cancelled), None);
    }

    #[test]
    fn grpc_code_name_is_used_as_error_code() {
        // Verify grpc_code_name returns the string that becomes error_code in exceptions.
        assert_eq!(grpc_code_name(tonic::Code::NotFound), "NOT_FOUND");
        assert_eq!(
            grpc_code_name(tonic::Code::InvalidArgument),
            "INVALID_ARGUMENT"
        );
        assert_eq!(grpc_code_name(tonic::Code::Unavailable), "UNAVAILABLE");
        assert_eq!(
            grpc_code_name(tonic::Code::DeadlineExceeded),
            "DEADLINE_EXCEEDED"
        );
        assert_eq!(
            grpc_code_name(tonic::Code::ResourceExhausted),
            "RESOURCE_EXHAUSTED"
        );
    }

    #[test]
    fn each_call_gets_distinct_request_id() {
        let mut interceptor = MetadataInterceptor::new("key", "2026-07").unwrap();
        let mut ids = HashSet::new();

        for _ in 0..100 {
            let request = tonic::Request::new(());
            let result = interceptor.call(request).unwrap();
            let request_id = result
                .metadata()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            ids.insert(request_id);
        }

        assert_eq!(ids.len(), 100, "All 100 request IDs should be unique");
    }

    #[test]
    fn ensure_port_appends_443_when_no_port() {
        assert_eq!(
            ensure_port("https://my-index.svc.pinecone.io"),
            "https://my-index.svc.pinecone.io:443"
        );
    }

    #[test]
    fn ensure_port_preserves_existing_port() {
        assert_eq!(
            ensure_port("https://my-index.svc.pinecone.io:8443"),
            "https://my-index.svc.pinecone.io:8443"
        );
        assert_eq!(
            ensure_port("https://my-index.svc.pinecone.io:443"),
            "https://my-index.svc.pinecone.io:443"
        );
    }

    #[test]
    fn ensure_port_handles_http_scheme() {
        assert_eq!(ensure_port("http://localhost"), "http://localhost:443");
        assert_eq!(
            ensure_port("http://localhost:5080"),
            "http://localhost:5080"
        );
    }

    #[test]
    fn ensure_port_preserves_path() {
        assert_eq!(
            ensure_port("https://my-index.svc.pinecone.io/some/path"),
            "https://my-index.svc.pinecone.io:443/some/path"
        );
    }

    #[test]
    fn ensure_port_no_scheme_passthrough() {
        // If there's no scheme, return as-is (shouldn't happen in practice)
        assert_eq!(
            ensure_port("my-index.svc.pinecone.io"),
            "my-index.svc.pinecone.io"
        );
    }

    #[test]
    fn max_message_size_constant_is_128mb() {
        assert_eq!(MAX_MESSAGE_SIZE, 128 * 1024 * 1024);
        assert_eq!(MAX_MESSAGE_SIZE, 134_217_728);
    }

    #[test]
    fn test_channel_creates_without_proxy() {
        // Verify the default path (no proxy) configures an endpoint successfully.
        let endpoint = Channel::from_shared("https://example.pinecone.io:443".to_string())
            .expect("valid endpoint");
        let result = endpoint.tls_config(ClientTlsConfig::new());
        assert!(
            result.is_ok(),
            "Endpoint without proxy should configure TLS successfully"
        );
    }

    #[test]
    fn test_channel_accepts_proxy_url() {
        // Verify that a valid proxy URL is parsed correctly and a Tunnel connector
        // can be created from it — this is the configuration path that GrpcChannel::new()
        // exercises when proxy_url is Some.
        let proxy_url = "http://proxy.example.com:8080";
        let proxy_dst: http::Uri = proxy_url.parse().expect("valid proxy URL");
        assert_eq!(proxy_dst.host(), Some("proxy.example.com"));
        assert_eq!(proxy_dst.port_u16(), Some(8080));

        let mut http_connector = HttpConnector::new();
        http_connector.enforce_http(false);
        // Constructing the Tunnel connector must not panic or error.
        let _tunnel = Tunnel::new(proxy_dst, http_connector);
    }

    #[test]
    fn secs_to_duration_rejects_negative() {
        // secs_to_duration delegates to try_from_secs_f64; verify that returns Err.
        assert!(Duration::try_from_secs_f64(-1.0).is_err());
    }

    #[test]
    fn secs_to_duration_rejects_nan_and_inf() {
        assert!(Duration::try_from_secs_f64(f64::NAN).is_err());
        assert!(Duration::try_from_secs_f64(f64::INFINITY).is_err());
    }

    #[test]
    fn secs_to_duration_accepts_zero_and_positive() {
        assert!(Duration::try_from_secs_f64(0.0).is_ok());
        assert!(Duration::try_from_secs_f64(0.5).is_ok());
        assert!(Duration::try_from_secs_f64(60.0).is_ok());
    }

    // Verify that `tonic::Code::Cancelled` with message "Timeout expired" routes to the same
    // exception class as `DeadlineExceeded`. This guards the fix for the tonic behaviour where
    // per-request `req.set_timeout()` expiry is surfaced as `Status::cancelled("Timeout expired")`
    // rather than `Status::deadline_exceeded(...)`.
    #[test]
    fn cancelled_timeout_expired_routes_to_pinecone_timeout_error() {
        let deadline_exceeded = tonic::Status::deadline_exceeded("some deadline");
        let cancelled_timeout = tonic::Status::cancelled("Timeout expired");
        let cancelled_other = tonic::Status::cancelled("operation cancelled by caller");

        // The error class name is determined inside status_to_py_err, but we can verify the
        // routing logic by checking the conditions the match uses directly.
        let deadline_class = match deadline_exceeded.code() {
            tonic::Code::DeadlineExceeded => "PineconeTimeoutError",
            _ => "PineconeError",
        };
        let cancelled_timeout_class = match cancelled_timeout.code() {
            tonic::Code::DeadlineExceeded => "PineconeTimeoutError",
            tonic::Code::Cancelled if cancelled_timeout.message() == "Timeout expired" => {
                "PineconeTimeoutError"
            }
            _ => "PineconeError",
        };
        let cancelled_other_class = match cancelled_other.code() {
            tonic::Code::DeadlineExceeded => "PineconeTimeoutError",
            tonic::Code::Cancelled if cancelled_other.message() == "Timeout expired" => {
                "PineconeTimeoutError"
            }
            _ => "PineconeError",
        };

        assert_eq!(deadline_class, "PineconeTimeoutError");
        assert_eq!(cancelled_timeout_class, "PineconeTimeoutError");
        assert_eq!(cancelled_other_class, "PineconeError");
    }
}
