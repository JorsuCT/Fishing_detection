from fishing_evaluation import fishing_evaluation
import json

def evaluation(caption):
    fishing = fishing_evaluation(caption)
    fishing_json = {
        'caption' : caption,
        'is fishing' : fishing,
    }

    return json.dumps(fishing_json, ensure_ascii=False)
