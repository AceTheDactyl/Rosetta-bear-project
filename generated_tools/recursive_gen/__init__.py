"""Recursive Generated Tools - Tools That Create Tools."""

from .recursive_recommendation_00 import RecursiveRecommendation00
from .recursive_recommendation_01 import RecursiveRecommendation01
from .recursive_recommendation_02 import RecursiveRecommendation02
from .recursive_recommendation_03 import RecursiveRecommendation03

RECURSIVE_TOOLS = {
    "recursive_recommendation_00": RecursiveRecommendation00,
    "recursive_recommendation_01": RecursiveRecommendation01,
    "recursive_recommendation_02": RecursiveRecommendation02,
    "recursive_recommendation_03": RecursiveRecommendation03,
}

__all__ = [
    "RECURSIVE_TOOLS",
]
