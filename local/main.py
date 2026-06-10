import sys
from PIL import Image
from server_controler import get_remote_caption
from jetson_controler import JetsonController
from evaluation import evaluation

def main(image, model_id="Qwen/Qwen3-VL-2B-Instruct", machine="DeepLearning2"):
    """Main function that orchestrates the entire process:"""

    if machine == "Orin":
        controller = JetsonController()

        if not controller.start_server(model_id):
            print("[FATAL ERROR] Could not start the server on the Jetson. Exiting.")
            sys.exit(1)
        
        try:
            img = Image.open(image).convert("RGB")
            caption = controller.image_inference(img, model_id)

        finally:
            controller.shutting_server()

    elif machine == "DeepLearning2":
        img = Image.open(image).convert("RGB")
        caption = get_remote_caption(img, model_id, machine)
    
    if isinstance(caption, dict) and "answer" in caption:
        caption = caption["answer"]

    final_result = evaluation(caption)
    
    print(final_result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [model] [machine]")
        sys.exit(1)

    path = sys.argv[1]
    
    selected_model = sys.argv[2] if len(sys.argv) > 2 else "google/gemma-4-E2B-it"
    selected_machine = sys.argv[3] if len(sys.argv) > 3 else "Orin"

    main(path, selected_model, selected_machine)
