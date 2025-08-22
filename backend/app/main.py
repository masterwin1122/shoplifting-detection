from fastapi import FastAPI
app = FastAPI(title="RetailGuard Pro API (minimal)")
@app.get("/v1/healthz")
def healthz():
    return {"status": "ok"}
