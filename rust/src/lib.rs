use pyo3::prelude::*;

pub mod retry;
mod transport;

/// Generated protobuf types for the Pinecone data plane.
// result_large_err: tonic-build generates `Result<_, tonic::Status>` (176 bytes); not ours to box.
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

#[cfg(test)]
mod proto_schema_tests {
    use prost::Message;
    use prost_types::field_descriptor_proto::Type;
    use prost_types::{DescriptorProto, FileDescriptorSet};

    const DESCRIPTOR_SET: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/db_data_descriptor.bin"));

    fn message(name: &str) -> DescriptorProto {
        let fds = FileDescriptorSet::decode(DESCRIPTOR_SET).expect("descriptor set decodes");
        fds.file
            .iter()
            .flat_map(|file| &file.message_type)
            .find(|msg| msg.name() == name)
            .unwrap_or_else(|| panic!("{name} missing from the compiled descriptor"))
            .clone()
    }

    #[test]
    fn compiled_descriptor_has_namespace_description_size_bytes() {
        let msg = message("NamespaceDescription");
        let field = msg
            .field
            .iter()
            .find(|field| field.name() == "size_bytes")
            .expect("NamespaceDescription.size_bytes missing — vendored proto is stale");

        assert_eq!(field.number(), 5);
        assert_eq!(field.r#type(), Type::Uint64);
    }

    #[test]
    fn generated_namespace_description_exposes_size_bytes() {
        let ns = crate::proto::NamespaceDescription {
            size_bytes: 4096,
            ..Default::default()
        };

        assert_eq!(ns.size_bytes, 4096);
    }

    #[test]
    fn compiled_descriptor_declares_all_twelve_rpcs() {
        let fds = FileDescriptorSet::decode(DESCRIPTOR_SET).expect("descriptor set decodes");
        let service = fds
            .file
            .iter()
            .flat_map(|file| &file.service)
            .find(|svc| svc.name() == "VectorService")
            .expect("VectorService missing from the compiled descriptor");
        let mut methods: Vec<&str> = service.method.iter().map(|m| m.name()).collect();
        methods.sort_unstable();

        assert_eq!(
            methods,
            [
                "CreateNamespace",
                "Delete",
                "DeleteNamespace",
                "DescribeIndexStats",
                "DescribeNamespace",
                "Fetch",
                "FetchByMetadata",
                "List",
                "ListNamespaces",
                "Query",
                "Update",
                "Upsert",
            ]
        );
    }
}
