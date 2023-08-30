import torch

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util


PRETRAINED_PATH = 'sentence-transformers/LaBSE'

# todo !!!
# import torch
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
# model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
# sentences = ["Hello World", "Привет Мир"]
# encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
# with torch.no_grad():
#     model_output = model(**encoded_input)
# embeddings = model_output.pooler_output
# embeddings = torch.nn.functional.normalize(embeddings)
# print(embeddings)


def get_device():
    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
        device = torch.device('cuda')
    else:
        print('No GPU available, using the GPU instead.')
        device = torch.device('cpu')
    return device


def run1():
    # https://sbert.net/examples/applications/semantic-search/README.html#approximate-nearest-neighbor
    embedder = SentenceTransformer(PRETRAINED_PATH)
    embedder.to(get_device())
    # todo ? to gpu
    print(type(embedder))
    # corpus = ['A man is eating food.',
    #           'A man is eating a piece of bread.',
    #           'The girl is carrying a baby.',
    #           'A man is riding a horse.',
    #           'A woman is playing violin.',
    #           'Two men pushed carts through the woods.',
    #           'A man is riding a white horse on an enclosed ground.',
    #           'A monkey is playing drums.',
    #           'A cheetah is running behind its prey.'
    #           ]

    corpus = ['A man is eating food.',
              'Мужчина ест еду',
              'A man is eating a piece of bread.',
              'Мужчина ест кусок хлеба',
              'The girl is carrying a baby.',
              'Девушка ухаживает за ребенком',
              'A man is riding a horse.',
              'Мужчина скачет на лошади',
              'A woman is playing violin.',
              'Женщина играет на скрипке',
              'Two men pushed carts through the woods.',
              'Двое мужчин толкали тележки через лес.',
              'A man is riding a white horse on an enclosed ground.',
              'Мужчина едет на белой лошади по огороженной территории.',
              'A monkey is playing drums.',
              'Обезьяна играет на барабанах.',
              'A cheetah is running behind its prey.'
              'Гепард бежит за своей добычей.'
              ]

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    print(corpus_embeddings.shape)

    queries = [
        'A man is eating pasta.',
        'Мужчина ест пасту',
        'Someone in a gorilla costume is playing a set of drums.',
        'Кто-то в костюме гориллы играет на барабанной установке.',
        'A cheetah chases prey on across a field.'
        'Гепард преследует добычу по полю.'
    ]

    top_k = min(5, len(corpus))
    print(f'top_k: {top_k}')

    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))
#
#     """
#     # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
#     hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
#     hits = hits[0]      #Get the hits for the first query
#     for hit in hits:
#         print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
#     """



def run0():
    s = [
        'Привет, я C++ программист',
        'Привет, я C++ программист',
        'Привет, я C программист',
        'Привет, я Java программист'
    ]
    model = SentenceTransformer(PRETRAINED_PATH)
    print(type(model))
    embeddings = model.encode(s)
    print(embeddings.shape)
    print(type(embeddings))
    print(embeddings)


if __name__ == '__main__':
    # run0()
    run1()
