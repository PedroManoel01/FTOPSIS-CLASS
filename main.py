from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ftopsis_class import run_ftopsis  # Sua função aqui
import uvicorn

app = FastAPI()

# Permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # substitua por seu domínio em produção
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run-ftopsis")
async def process_json(request: Request):
    data = await request.json()
    output = run_ftopsis(data)
    return output

if __name__ == "__main__":
    uvicorn.run("main:app", host="https://ftopsis-class-eqglhyoep-pedromanoel01s-projects.vercel.app 
