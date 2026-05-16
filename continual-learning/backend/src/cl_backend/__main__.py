import uvicorn
from cl_backend.config import ServerConfig

cfg = ServerConfig()
uvicorn.run("cl_backend.main:app", host="0.0.0.0", port=cfg.port, reload=False)
