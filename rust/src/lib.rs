use pyo3::prelude::*;

pub mod retry;
mod transport;

/// Generated protobuf types for the Pinecone data plane.
pub mod proto {
    tonic::include_proto!("_");
}

/// Pinecone gRPC extension module, importable as `pinecone._grpc`.
#[pymodule]
fn _grpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<transport::GrpcChannel>()?;
    Ok(())
}
