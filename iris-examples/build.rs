// The focused inference workspace does not build historical custom-op examples
// or download models at compile time. Runtime examples fetch explicitly requested
// model artifacts through hf-hub instead.
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
}
