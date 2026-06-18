from ast import If
import os
import time
import requests
import paramiko
import base64
import io
from dotenv import load_dotenv

load_dotenv()

class JetsonController:
    def __init__(self):
        self.ip = os.getenv('JETSON_IP')
        self.user = os.getenv('JETSON_USER')
        self.password = os.getenv('JETSON_PASS')
        self.port = 8080

    def _connect_ssh(self):
        """Create and return a secure SSH connection."""

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.ip, username=self.user, password=self.password)
        return ssh

    def start_server(self, model):
        """Run the model on the Jetson in an optimized way and wait for it to be ready."""

        print(f"[JETSON] Starting server for: {model}.")

        model_name_lower = model.lower()
        if "blip" in model_name_lower or "opt" in model_name_lower:
            max_len = 2048
        else:
            max_len = 4096

        ssh = self._connect_ssh()
        
        try:
            ssh.exec_command("docker rm -f vllm_server 2>/dev/null")
            time.sleep(2)
            
            cmd_base = (
                "docker run -d --name vllm_server --runtime=nvidia --network host "
                "--shm-size=4g "
                "-v $HOME/.cache/huggingface:/root/.cache/huggingface "
            )

            if "google" in model_name_lower:
                cmd = cmd_base + (
                    "ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin "
                    f"vllm serve {model} "
                    f"--port {self.port} "
                    "--gpu-memory-utilization 0.4 " 
                    f"--max-model-len {max_len} "
                    "--enable-auto-tool-choice "
                    "--reasoning-parser gemma4 "
                    "--tool-call-parser gemma4 "
                    "--enforce-eager"
                )
            else:
                cmd = cmd_base + (
                    "ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin "
                    f"vllm serve {model} "
                    f"--port {self.port} "
                    "--gpu-memory-utilization 0.3 "
                    f"--max-model-len {max_len} "
                    "--limit-mm-per-prompt '{\"image\": 1}' "
                    "--enforce-eager"
                )

            _, _, stderr = ssh.exec_command(cmd)
            error_ssh = stderr.read().decode().strip()

            if error_ssh and not error_ssh.startswith("WARNING"):
                print(f"[FATAL ERROR SSH]: {error_ssh}")
                return False
            
            print("[JETSON] Waiting for the vLLM engine to load in VRAM.")

            for tries in range(60):
                time.sleep(10) 
                try:
                    res = requests.get(f"http://{self.ip}:{self.port}/v1/models", timeout=2)
                    if res.status_code == 200:
                        print(f"[JETSON] Engine ready in aprox {tries * 10} seconds.")
                        return True
                    
                except requests.exceptions.RequestException:
                    pass
                    
            print("[ERROR] The server did not respond in time. Check the logs on the Jetson.")
            return False
            
        finally:
            ssh.close()

    def shutting_server(self):
        """Shutdown the server and destroy the container, freeing up RAM and VRAM."""

        print("[JETSON] Shutting down the server and freeing resources.")
        ssh = self._connect_ssh()
        try:
            ssh.exec_command("docker stop vllm_server")
            print("[JETSON] Cleanup completed. Zero impact.")

        finally:
            ssh.close()

    def image_inference(self, image, model, prompt="Describe this image."):
        """Converts the image to base64 and makes a request to the vLLM API."""
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        url = f"http://{self.ip}:{self.port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        }
        
        print(f"[JETSON] Sending image to {model}.")
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
