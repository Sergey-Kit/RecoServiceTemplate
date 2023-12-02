import random
import typing
from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import ModelNotFound, UserNotFound
from service.log import app_logger
from service.models import popular, user_knn


models = ("test_model", "top", "random", "popular", "user_knn")


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class Message(BaseModel):
    message: str


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@typing.no_type_check
@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {
            "description": "Successful recommendation",
            "model": RecoResponse
        },
        404: {
            "description": "Wrong inputs",
            "model": Message
        },
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFound(error_message=f"User {user_id} not found")

    if model_name not in models:
        raise ModelNotFound(error_message=f"Model {model_name} not found")

    if model_name == "test_model":
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
    elif model_name == "top":
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
    elif model_name == "random":
        reco = [random.randint(0, 1000) for _ in range(10)]
    elif model_name == "popular":
        reco = popular.predict()
    elif model_name == "user_knn":
        reco = user_knn.predict(user_id)
    else:
        raise ModelNotFound(error_message=f"Model {model_name} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
