from utils import load_data
from embeddings import generate_embeddings

def main():
    data = load_data()
    chunk_sizes = [512, 1024, 2048]

    for chunk_size in chunk_sizes:
        generate_embeddings(data, chunk_size)

if __name__ == "__main__":
    main()
