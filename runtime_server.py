"""Existing backend API plus a separate runtime frontend."""
from pathlib import Path
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from crowd_reliability.api import app
from crowd_reliability.runtime import router
app.include_router(router)
from crowd_reliability.live_provider import router as world_router
app.include_router(world_router)
app.mount('/dashboard', StaticFiles(directory=Path(__file__).parent/'runtime_frontend',html=True),name='runtime_dashboard')
@app.get('/runtime')
def runtime():
    return RedirectResponse('/dashboard/')
