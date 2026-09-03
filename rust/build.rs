fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let proto_root = manifest_dir.join("proto");
    let google_include = manifest_dir.join("proto");
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    tonic_build::configure()
        .build_server(false)
        // Emitted so tests can assert on the compiled schema itself and catch a
        // stale vendored proto.
        .file_descriptor_set_path(out_dir.join("db_data_descriptor.bin"))
        .compile_protos(
            &[proto_root.join("db_data_2026-07.proto")],
            &[&proto_root, &google_include],
        )?;

    Ok(())
}
