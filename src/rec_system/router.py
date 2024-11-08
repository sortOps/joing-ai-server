from fastapi import APIRouter
from rec_system.schemas import CreatorRecommendRequest, ItemRecommendRequest
from rec_system.service import random_for_item, random_for_creator

router = APIRouter()


# Creator recommend Endpoint
@router.post("/ai/recommendation/creator")
def creator_recommendation_router(request: CreatorRecommendRequest):
    return random_for_creator(request)


# Item Recommend Endpoint
@router.post("/ai/recommendation/item")
def item_recommendation_router(request: ItemRecommendRequest):
    return random_for_item(request)
