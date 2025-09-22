from pydantic import BaseModel
import modal

docker_image = modal.Image.from_dockerfile("Dockerfile.serverless")
app = modal.App("coco-ai", image=docker_image)

class CodeReq(BaseModel):
    code: str

snapshot_key = "v1"

with docker_image.imports():
    from reference.wpc import WPC

@app.cls(
    gpu="T4", 
    enable_memory_snapshot=True, 
    experimental_options={"enable_gpu_snapshot": True}
)
class Inference:
    @modal.enter(snap=True)
    def load_module(self):
        print("loading model")
        self.wpc = WPC()
        print(f"snapshotting {snapshot_key}")

    @modal.fastapi_endpoint(method="POST")
    def run(self, p_id:str, code_req:CodeReq):
        try:
            fixed_code,generalized_code = self.wpc.process(code_req.code,p_id)
        except Exception as e:
            return {"error":0}
        return {'fixed_code':fixed_code,'bug_code':generalized_code}