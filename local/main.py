import json
import sys
from PIL import Image
from server_controler import get_remote_caption
from jetson_controler import JetsonController
from evaluation import evaluation

def main(image, model_id="openai/clip-vit-large-patch14", machine="DeepLearning2", prompt_type="yes_no"):
    """Main function that orchestrates the entire process:"""

    if prompt_type == "yes_no":
        prompt = "Do you see someone fishing in the image? Answer only 'yes' or 'no'."
    elif prompt_type == "multiple_choice":
        prompt = "Which of the following activities describe better what it is happening in the image? a) Fishing b) Reading c) Not Fishing d) Running. Answer only with the letter of the most correct option."
    elif prompt_type == "focus":
        prompt = "Describe this image concisely (maximum 2 sentences). - If fishing-related elements are present (e.g., anglers, rods, boats, fish, gear, waterbodies), focus your description primarily on those elements. - If the image does not contain any fishing context, provide a brief, objective description of the scene. Do not mention or imply fishing if it is not explicitly depicted."
    else:
        prompt = "Describe the image."

    if machine == "Orin":
        controller = JetsonController()

        if not controller.start_server(model_id):
            print("[FATAL ERROR] Could not start the server on the Jetson. Exiting.")
            sys.exit(1)
        
        try:
            img = Image.open(image).convert("RGB")
            caption = controller.image_inference(img, model_id, prompt)

        finally:
            controller.shutting_server()

    elif machine == "DeepLearning2":
        img = Image.open(image).convert("RGB")
        caption = get_remote_caption(img, model_id, machine, prompt)
    
    if isinstance(caption, dict) and "answer" in caption:
        caption = caption["answer"]

    if prompt_type == "yes_no":
        is_fishing_pred = "Fishing" if caption.startswith("yes") or "yes" in f" {caption} " else "Not Fishing"
        json_result = {
            "caption": caption, 
            "is_fishing_pred": is_fishing_pred
        }
        json_result = json.dumps(json_result, ensure_ascii=False)

    elif prompt_type == "multiple_choice":
        is_fishing_pred = "Fishing" if "a)" in caption else "Not Fishing"
        json_result = {
            "caption": caption, 
            "is_fishing_pred": is_fishing_pred
        }
        json_result = json.dumps(json_result, ensure_ascii=False)
    
    else:
        json_result = evaluation(caption)

    print(json_result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [model] [machine] [prompt_type]")
        sys.exit(1)

    path = sys.argv[1]
    
    selected_model = sys.argv[2] if len(sys.argv) > 2 else "openai/clip-vit-large-patch14"
    selected_machine = sys.argv[3] if len(sys.argv) > 3 else "DeepLearning2"
    
    selected_prompt = sys.argv[4] if len(sys.argv) > 4 else "yes_no"

    main(path, selected_model, selected_machine, selected_prompt)