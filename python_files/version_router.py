from fastapi import APIRouter, Request

async def _version(request: Request):
    return {"version": request.app.version}

class VersionRouter(APIRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_api_route(path="/", endpoint=_version, methods=["GET"],)

router = VersionRouter(prefix="/version",)