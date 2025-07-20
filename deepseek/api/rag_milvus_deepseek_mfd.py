import os
from openai import OpenAI
from pymilvus import model as milvus_model
from pymilvus import MilvusClient
from tqdm import tqdm
import json

from sentence_transformers import SentenceTransformer

api_key = os.getenv("DEEPSEEK_API_KEY")

text_lines = []

with open("mfd.md", "r") as f:
    file_text = f.read()
    text_lines += file_text.split("\n")
    # text_lines += file_text.split("\n")

print(text_lines)
print(len(text_lines))
print(api_key)

deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

# embedding_model = milvus_model.DefaultEmbeddingFunction()
model_name = "Qwen/Qwen3-Embedding-0.6B"
model = SentenceTransformer(model_name)
embedding_model = SentenceTransformer(model_name)
# test_embedding = embedding_model.encode_queries(["这是一个测试句子", "这是另一个示例文本"])[0]
test_embedding = embedding_model.encode(["这是一个测试句子", "这是另一个示例文本"])[0]
embedding_dim = len(test_embedding)
print(test_embedding)
print(test_embedding[:10])

milvus_client = MilvusClient(uri="./mfd.db")
collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",
    consistency_level="Strong"
)

data = []

# doc_embeddings = embedding_model.encode_documents(text_lines)
doc_embeddings = embedding_model.encode(text_lines)

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

milvus_client.insert(collection_name=collection_name, data=data)

print(milvus_client)

SYSTEM_PROMPT = """
Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
"""

USER_PROMPT = """
请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。
<context>
{context}
</context>
<question>
{question}
</question>
"""


def search_and_question(q):
    search_res = milvus_client.search(collection_name=collection_name,
                                      data=embedding_model.encode([q]),
                                      limit=10,
                                      search_params={"metric_type": "IP", "params": {}},
                                      output_fields=["text"]
                                      )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    print(json.dumps(retrieved_lines_with_distances, indent=4))

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    print(context)
    print(q)

    user_prompt = USER_PROMPT.format(context=context, question=q)
    print(user_prompt)

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


question = "宅基地能卖吗？"
print(search_and_question(question))
question_2 = "高空砸物怎么办？"
print(search_and_question(question_2))
