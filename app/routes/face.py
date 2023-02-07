from fastapi import APIRouter


router = APIRouter()


@router.get("/face")
async def face():
    return "face"
