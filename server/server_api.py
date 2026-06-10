from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration, Qwen3VLForConditionalGeneration, PreTrainedModel, InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch
import io
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.enabled = False

app = FastAPI()

current_model_name = None
current_model = None
current_processor = None

def load_model(model_name: str):
    """Download and load the specified model into VRAM."""

    global current_model_name, current_model, current_processor

    if current_model_name == model_name and current_model is not None:
        return 
        
    if current_model is not None or current_processor is not None:
        try:
            del current_model
        except NameError:
            pass
        try:
            del current_processor
        except NameError:
            pass
            
        current_model = None
        current_processor = None
        current_model_name = None
        gc.collect()
        torch.cuda.empty_cache()
        
    
    if model_name == "openai/clip-vit-large-patch14":
        current_processor = CLIPProcessor.from_pretrained(model_name)
        current_model = CLIPModel.from_pretrained(model_name).to("cuda")
        
    elif model_name == "Salesforce/blip2-opt-2.7b":
        current_processor = Blip2Processor.from_pretrained(model_name)
        current_model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        
    elif model_name == "Qwen/Qwen3-VL-2B-Instruct":
        current_processor = AutoProcessor.from_pretrained(model_name)
        current_model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    
    elif model_name == "Salesforce/instructblip-flan-t5-xl":
        current_processor = InstructBlipProcessor.from_pretrained(model_name)
        current_model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16
        )

    else:
        raise ValueError(f"model {model_name} not supported.")
        
    current_model_name = model_name


@app.post("/infer")
async def infer(model: str = Form(...), prompt: str = Form("Describe this image"), file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        load_model(model)
        
        if model == "Salesforce/blip2-opt-2.7b":
            blip_prompt = f"Question: {prompt} Answer:"
            
            inputs = current_processor(
                images=image, 
                text=blip_prompt, 
                return_tensors="pt"
            ).to("cuda", torch.float16)
            
            generated_ids = current_model.generate(**inputs, max_new_tokens=100)
            result = current_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if "Answer:" in result:
                result = result.split("Answer:")[-1].strip()
            else:
                result = result.strip()
            
        elif model == "openai/clip-vit-large-patch14":
            text_to_compare = ["a person of fishing", "a person not fishing"]
            inputs = current_processor(text=text_to_compare, images=image, return_tensors="pt", padding=True).to("cuda")
            outputs = current_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            best_class = text_to_compare[probs.argmax().item()]
            result = f"CLIP Clasification: {best_class}"
            
        elif model == "Qwen/Qwen3-VL-2B-Instruct":
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = current_processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to("cuda")
            
            generated_ids = current_model.generate(**inputs, max_new_tokens=100)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = current_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        elif model == "Salesforce/instructblip-flan-t5-xl":
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            with torch.no_grad():
                inputs = current_processor(images=image, text=prompt, return_tensors="pt").to(current_model.device)
                
                outputs = current_model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1, 
                    max_length=100,
                    min_length=10
                )
                
                result = current_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        return {"caption": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
