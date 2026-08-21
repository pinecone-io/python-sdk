use pyo3::prelude::*;

pub mod retry;
mod transport;

/// Generated protobuf types for the Pinecone data plane.
// result_large_err: every generated client method returns `Result<_, tonic::Status>`,
// and `Status` is over clippy's error-size threshold. The bodies are tonic output we
// do not author, so the lint has nothing actionable to say about them.
#[allow(clippy::result_large_err)]
pub mod proto {
    tonic::include_proto!("_");
}

/// Pinecone gRPC extension module, importable as `pinecone._grpc`.
#[pymodule]
fn _grpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<transport::GrpcChannel>()?;
    Ok(())
}
