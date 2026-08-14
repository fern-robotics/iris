.PHONY: clean-ptx clean test

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > iris-kernels/src/lib.rs
	touch iris-kernels/build.rs
	touch iris-examples/build.rs
	touch iris-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
