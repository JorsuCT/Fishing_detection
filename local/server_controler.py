import sys
import requests
import io
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "DeepLearning2": {
        "url": os.getenv("DEEPLEARNING_IP"),
        "timeout": 300
    }
}

def get_remote_caption(image, model_id, machine, prompt="Describe this image concisely (maximum 2 sentences). - If fishing-related elements are present (e.g., anglers, rods, boats, fish, gear, waterbodies), focus your description primarily on those elements. - If the image does not contain any fishing context, provide a brief, objective description of the scene. Do not mention or imply fishing if it is not explicitly depicted."):
    conf = CONFIG.get(machine)
    if not conf:
        print(f"[ERROR] Machine {machine} not found in the configuration.")
        sys.exit(1)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    prompt = {
        "model": model_id,
        "prompt": prompt
    }
    
    files = {
        "file": ("frame.jpg", buffered, "image/jpeg")
    }
    
    try:
        response = requests.post(
            conf["url"], 
            data=prompt, 
            files=files, 
            timeout=conf["timeout"]
        )
        
        if response.status_code != 200:
            print(f"\n[ERROR IN SERVER] Code {response.status_code}: {response.text}")
            sys.exit(1)
            
        caption = response.json().get("caption", "")
        return caption
        
    except requests.exceptions.ConnectionError:
        print("\n[FATAL ERROR] Could not connect to the server. Is the Docker running on port 8000?")
        sys.exit(1)
    except requests.exceptions.ReadTimeout:
        print("\n[ERROR] Timeout exceeded. The server took too long to load the model in VRAM.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Something went wrong during the request: {str(e)}")
        sys.exit(1)
