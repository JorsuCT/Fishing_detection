import os
import sys
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score

from server_controler import get_remote_caption
from jetson_controler import JetsonController
from fishing_evaluation import fishing_evaluation

def run_benchmark(model_id="Qwen/Qwen3-VL-2B-Instruct", machine="DeepLearning2"):
    print(f"\n[STARTING BENCHMARK] Model: {model_id} | Machine: {machine}")
    
    base_folder = Path("dataset")
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    os.makedirs(f"codigo/metrics/{machine}", exist_ok=True)
    
    controller = None
    
    if machine == "Orin":
        controller = JetsonController()
        if not controller.start_server(model_id):
            print("[FATAL ERROR] Can not start the server on the Jetson. Exiting.")
            sys.exit(1)
            
    results_list = []
    y_true = []
    y_pred = []
    total_time = 0
    image_count = 0

    # prompt = "Do you see someone fishing in the image? Answer only 'yes' or 'no'."
    # prompt = "Describe this image concisely (maximum 2 sentences). - If fishing-related elements are present (e.g., anglers, rods, boats, fish, gear, waterbodies), focus your description primarily on those elements. - If the image does not contain any fishing context, provide a brief, objective description of the scene. Do not mention or imply fishing if it is not explicitly depicted."
    prompt = "Which of the following activities describe better what it is happening in the image? a) Fishing b) Reading c) Not Fishing d)Running. Answer only with the letter of the most correct option."

    try:
        for img_path in base_folder.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in extensions:
                        try:
                            parent_folder_name = img_path.parent.name
                            is_fishing_gt = 1 if "fishing_" in parent_folder_name else 0

                            img = Image.open(img_path).convert("RGB")
                            
                            start_time = time.time()
                            
                            if machine == "Orin":
                                caption = controller.image_inference(img, model_id, prompt)
    
                            elif machine == "DeepLearning2":
                                caption = get_remote_caption(img, model_id, machine, prompt)
                                
                            if isinstance(caption, dict) and "caption" in caption:
                                caption = caption["caption"]
                            
                            caption = str(caption).lower()
                            
                            if "answer:" in caption:
                                caption = caption.split("answer:")[-1].strip()
                            
                            if "clip" in model_id.lower():
                                is_fishing_pred = 1 if "a person of fishing" in caption else 0
                            else:
                                eval_result = fishing_evaluation(caption)
                                # is_fishing_pred = 1 if eval_result == "Fishing" else 0 
                                # is_fishing_pred = 1 if caption.startswith("yes") or "yes" in f" {caption} " else 0
                                is_fishing_pred = 1 if  "Fishing" in caption or "a)" in caption else 0
                            
                            proc_time = time.time() - start_time
                            
                            y_true.append(is_fishing_gt)
                            y_pred.append(is_fishing_pred)
                            total_time += proc_time
                            image_count += 1
                            
                            results_list.append({
                                "Folder": parent_folder_name,
                                "Image": img_path.name,
                                "Caption": str(caption),
                                "Ground_Truth": is_fishing_gt,
                                "Model_Prediction": is_fishing_pred,
                                "Time_s": round(proc_time, 4)
                            })
                            
                            print(f"Completed: {img_path.name} | GT: {is_fishing_gt} | Pred: {is_fishing_pred} | Time: {proc_time:.2f}s")
                            
                        except Exception as e:
                            print(f"Error processing image {img_path.name}: {e}")
                            
    finally:
        if machine == "Orin" and controller:
            controller.shutting_server()
            
    if image_count > 0:
        safe_model_name = model_id.replace("/", "_")
        detailed_csv = f"codigo/metrics/{machine}/detailed_results_{safe_model_name}.csv"
        
        df_detailed = pd.DataFrame(results_list)
        df_detailed.to_csv(detailed_csv, sep=';', index=False)
        print(f"\nDetailed results saved in: {detailed_csv}")
        
        avg_time = total_time / image_count
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        summary_csv = f"codigo/metrics/{machine}/model_performance_summary_multiple.csv"
        summary_data = pd.DataFrame({
            "Model": [model_id],
            "Machine": [machine],
            "Average_Time_s": [round(avg_time, 4)],
            "F1_Score_Macro": [round(macro_f1, 4)]
        })
        
        if os.path.exists(summary_csv):
            df_summary = pd.read_csv(summary_csv, sep=';')
            mask = (df_summary['Model'] == model_id) & (df_summary['Machine'] == machine)
            if mask.any():
                df_summary.loc[mask, ['Average_Time_s', 'F1_Score_Macro']] = [round(avg_time, 4), round(macro_f1, 4)]
            else:
                df_summary = pd.concat([df_summary, summary_data], ignore_index=True)
        else:
            df_summary = summary_data
            
        df_summary.to_csv(summary_csv, sep=';', index=False)
        
        print(f"\n[BENCHMARK COMPLETED SUCCESSFULLY]")
        print(f"Model: {model_id}")
        print(f"F1-Score Macro: {macro_f1:.4f}")
        print(f"Average Time: {avg_time:.4f}s")
        print(f"Metrics updated in {summary_csv}")
        
    else:
        print("\nNo image was processed successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <model> [machine]")
        sys.exit(1)
        
    selected_model = sys.argv[1]
    selected_machine = sys.argv[2] if len(sys.argv) > 2 else "DeepLearning2"
    
    run_benchmark(selected_model, selected_machine)
