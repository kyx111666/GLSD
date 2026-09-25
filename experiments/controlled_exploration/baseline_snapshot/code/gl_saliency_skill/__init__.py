"""Backbone-agnostic, training-free historical Global-Local saliency skill."""

from .evidence import Config
from .skill import GLSaliencySkill

__all__ = ["Config", "GLSaliencySkill"]
