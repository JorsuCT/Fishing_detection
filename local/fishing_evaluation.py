from sentence_transformers import SentenceTransformer, util
import torch

fishing_descriptions = [
    "people holding a fishing tool to fish",
    "a person fishing with a tool",
    "a person fishing in a boat",
    "a fishing tool is used",
    "a boat with fishing tools",
    "a person engaged in fishing activities",
    "people fishing from a boat",
    "a fishing net is used",
    "a fishing rod is used",
    "a fishing pole is used",
    "a fishing equipment is used",
    "fishing boat with people trying to fish"
]

no_fishing_descriptions = [
    "people standing on a boat",
    "people lying on a boat",
    "ship with people inside",
    "yacht with people inside",
    "boat with people inside",
    "boat shipping in the water",
    "people abroad a boat",
    "people on the deck of a boat",
    "boat is equipped with a flag"
]

model_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

def fishing_evaluation(caption):
    fish_emb = model_embeddings.encode(fishing_descriptions, convert_to_tensor=True)
    no_fish_emb = model_embeddings.encode(no_fishing_descriptions, convert_to_tensor=True)

    emb_caption = model_embeddings.encode(caption, convert_to_tensor=True)

    fish_similarity = util.cos_sim(emb_caption, fish_emb)[0]
    no_fish_similarity = util.cos_sim(emb_caption, no_fish_emb)[0]

    max_fish_sim = torch.max(fish_similarity).item()
    max_no_fish_sim = torch.max(no_fish_similarity).item()

    fishin_prediction = 1 if max_fish_sim > max_no_fish_sim else 0
    
    if fishin_prediction == 1:
        return "Fishing"
    
    else:
        return "No Fishing"
