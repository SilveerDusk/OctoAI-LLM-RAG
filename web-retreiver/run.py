import os

if __name__ == "__main__":
    from main import predict

    print(predict(data_source="octoai_docs", prompt="how to reduce cold starts?"))
