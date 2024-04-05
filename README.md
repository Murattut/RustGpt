# RustGpt

RustGpt is a variant of OpenAI's GPT-2 language model implemented in Rust using the tch (Torch for Rust) library. It allows for generating human-like text based on input prompts.

## Features

- **GPT-2 Variant**: This project implements a variant of OpenAI's GPT-2 language model.
- **Rust Implementation**: Utilizes the Rust programming language for efficiency and performance.
- **Torch Integration**: Leverages the tch library, a Rust binding for PyTorch, for neural network computations.
- **Text Generation**: Generates human-like text based on input prompts using the trained GPT-2 model.

## Installation

To use RustGpt, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/Murattut/RustGpt.git
    ```

2. Navigate to the project directory:

    ```bash
    cd RustGpt
    ```

3. Build the project using Cargo:

    ```bash
    cargo build --release
    ```

4. Once built, you can run the application:

    ```bash
    cargo run --release
    ```

## Usage (Under Development)

After building and running the project, you can interact with the RustGpt variant through a command-line interface (CLI). Simply provide a prompt, and the model will generate text based on it.
Example:

```bash
./RustGpt "Once upon a time, in a faraway land"
```

## Contributing

Contributions are welcome! If you'd like to contribute to RustGpt, please open an issue to discuss the changes you'd like to make or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://openai.com) for providing the GPT-2 model and inspiration for this project.
- [PyTorch](https://pytorch.org) for the underlying deep learning framework.
- [tch (Torch for Rust)](https://github.com/LaurentMazare/tch-rs) for the Rust binding to PyTorch.

## Contact

For questions, suggestions, or other inquiries, feel free to reach out to [Your Name](mailto:your_email@example.com).

---

Feel free to customize this README file with additional information or features specific to your implementation!
