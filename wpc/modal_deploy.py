from pydantic import BaseModel
import modal

docker_image = modal.Image.from_dockerfile("Dockerfile.serverless")
app = modal.App("coco-ai", image=docker_image)

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

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, HTTPException

        web_app = FastAPI()

        class CodeReq(BaseModel):
            code: str

        @web_app.post("/process")
        def run(p_id:str, code_req:CodeReq):
            try:
                fixed_code,generalized_code = self.wpc.process(code_req.code,p_id)
            except Exception as e:
                raise HTTPException(500,e)
            return {'fixed_code':fixed_code,'bug_code':generalized_code}
        
        @web_app.get('/hello')
        def test():
            """
            모델일 정상적으로 동작하는지 테스트

            returns
            - 정상동작시 True, 비정상동작시 False
            """

            try:
                self.wpc.process('a,b,c=map(int,input().split())\nsound=0\nfor i in range(c):\n    b-=a\nif a<0:\n    break\nelse:\n    sound+=1\nprint(sound)',"p03105")
            except Exception as e:
                print(str(e))
                return False
            return True
        return web_app